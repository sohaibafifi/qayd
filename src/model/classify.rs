//! Backend selection: model classification and engine choice.

use super::*;
use crate::model::list;

type ItemSumBound = (usize, Vec<(i32, i64)>, i64, i64);
type ItemSumSignature = (Vec<(i32, i64)>, i64, i64);
const MAX_INTEGER_ROUTING_NODES: usize = 32;

// `auto` is a performance policy, not only a capability check. Ordered-list
// enumeration grows factorially, assignment enumeration grows exponentially,
// and the current exact scheduling backend is intended for compact models. The
// explicit exact path may use larger models, but its classifier is still capped
// by `MAX_CLASSIFICATION_WORK` so model inspection itself can never become an
// unbounded pre-search phase.
const AUTO_ORDERED_EXACT_ITEMS: usize = 10;
const AUTO_ASSIGNMENT_EXACT_ITEMS: usize = 24;
const AUTO_ASSIGNMENT_EXACT_CELLS: usize = 192;
const AUTO_SCHEDULE_EXACT_INTERVALS: usize = 48;
const AUTO_SCHEDULE_EXACT_MODES: usize = 96;
const MAX_CLASSIFICATION_WORK: u128 = 1_000_000;
const MAX_SCHEDULE_CLASSIFICATION_OBJECTS: usize = 100_000;
const MAX_EXACT_CONSTRUCTION_CELLS: usize = 100_000;

/// High-level model shape used to choose a solver engine.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModelClass {
    /// Finite-domain integer model.
    PureInteger,
    /// List model without edge costs or sequence-only reductions.
    ListAssignmentPacking,
    /// List model with edge reductions, usually routing.
    Routing,
    /// List model with scan, window, or pair reductions.
    Sequencing,
    /// Interval scheduling model.
    Schedule,
}

impl ModelClass {
    /// Stable display name for verbose output.
    pub fn name(self) -> &'static str {
        match self {
            Self::PureInteger => "integer",
            Self::ListAssignmentPacking => "list assignment / packing",
            Self::Routing => "routing",
            Self::Sequencing => "sequencing",
            Self::Schedule => "schedule",
        }
    }
}

/// Solver engine family selected for a model.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    /// Exact finite-domain integer search.
    IntegerExact,
    /// Exact list / interval search.
    DomainExact,
    /// Heuristic list local search.
    ListLocalSearch,
    /// Heuristic interval schedule search.
    ScheduleLocalSearch,
}

impl Backend {
    /// Stable display name for verbose output.
    pub fn name(self) -> &'static str {
        match self {
            Self::IntegerExact => "integer exact",
            Self::DomainExact => "domain exact",
            Self::ListLocalSearch => "list local search",
            Self::ScheduleLocalSearch => "schedule local search",
        }
    }

    /// Whether this backend is exact.
    pub fn is_exact(self) -> bool {
        matches!(self, Self::IntegerExact | Self::DomainExact)
    }

    /// Whether this backend is heuristic.
    pub fn is_heuristic(self) -> bool {
        !self.is_exact()
    }

    /// Stable role description for architecture checks and diagnostics.
    pub fn role(self) -> &'static str {
        match self {
            Self::IntegerExact | Self::DomainExact => "semantic exact solver",
            Self::ListLocalSearch => "fallback / incumbent / routing heuristic",
            Self::ScheduleLocalSearch => "fallback schedule heuristic",
        }
    }
}

/// Routing-related features detected in a list model.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct RoutingShape {
    /// At least one edge reduction is present.
    pub has_edges: bool,
    /// The model has multiple ordered lists.
    pub multi_route: bool,
    /// Per-list item-sum constraints are present.
    pub has_capacity_like_constraints: bool,
    /// Scan reductions are present.
    pub has_scan: bool,
    /// Same-list globals are present.
    pub has_same_list: bool,
    /// Item-precedence globals are present.
    pub has_item_precedence: bool,
    /// A used-list objective tier is present.
    pub has_fleet_objective: bool,
}

/// Diagnostic result for backend selection.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BackendSelection {
    /// Selected model class.
    pub class: ModelClass,
    /// Selected backend.
    pub backend: Backend,
    /// Short reason suitable for verbose logs.
    pub reason: &'static str,
    /// Routing features, if relevant.
    pub routing: Option<RoutingShape>,
}

impl BackendSelection {
    /// Select a backend for a shared Rust model.
    pub fn for_model(model: &Model) -> Self {
        if !model.intervals.is_empty() {
            let all_fixed = model.intervals.iter().all(|interval| interval.modes.is_empty());
            // Generic interval model, recognised by structure
            // Fixed intervals: precedence + no-overlap + cumulative. Moded
            // intervals (per-operation machine choice): precedence + machine
            // no-overlap, lowered to optional mode-intervals + an `alternative` per
            // operation. (Job-shop, flexible job-shop and RCPSP all compose from these.)
            let constraints_ok = if all_fixed {
                model.constraints.iter().all(|constraint| {
                    matches!(
                        constraint,
                        Constraint::IntervalPrecedence { .. }
                            | Constraint::IntervalResource(list::Resource::NoOverlap(_))
                            | Constraint::IntervalResource(list::Resource::Cumulative { .. })
                    )
                })
            } else {
                model.constraints.iter().all(|constraint| {
                    matches!(
                        constraint,
                        Constraint::IntervalPrecedence { .. } | Constraint::IntervalResource(list::Resource::MachineNoOverlap)
                    )
                })
            };
            let interval_schedule =
                constraints_ok && model.objectives.iter().all(|objective| matches!(objective, Objective::Makespan { minimize: true, .. }));
            if interval_schedule {
                return Self {
                    class: ModelClass::Schedule,
                    backend: Backend::DomainExact,
                    reason: if all_fixed {
                        "fixed intervals: precedence, no-overlap, cumulative"
                    } else {
                        "machine choice: precedence + per-machine no-overlap via alternative"
                    },
                    routing: None,
                };
            }
            return Self {
                class: ModelClass::Schedule,
                backend: Backend::ScheduleLocalSearch,
                reason: "schedule uses resources, modes, objectives, or unsupported constraints",
                routing: None,
            };
        }

        if !model.lists.is_empty() {
            let summary = ListSummary::for_model(model);
            let interval_constraints = model.constraints.iter().all(|constraint| {
                matches!(
                    constraint,
                    Constraint::ListPartition { .. }
                        | Constraint::SameList { .. }
                        | Constraint::ItemPrecedence { .. }
                        | Constraint::CollectionGlobal(_)
                        | Constraint::ListLength { .. }
                        | Constraint::ListItemSum { .. }
                        | Constraint::ListReduction(list::Constraint {
                            reduction: list::Reduction { iterable: list::Iterable::SetItems(_), .. },
                            ..
                        })
                )
            });

            if summary.has_edges {
                if routing_integer_lowering_supported(model) {
                    return Self {
                        class: ModelClass::Routing,
                        backend: Backend::IntegerExact,
                        reason: "closed route edge-sums lower to integer circuit",
                        routing: summary.routing_shape(model.lists.len()),
                    };
                }
                // Exact list enumeration handles a max_of objective, a linear edge
                // objective over an optional list model (its free pool list only
                // enumeration resolves soundly), and an edge objective mixed with a
                // scan term that integer edge-lowering cannot represent.
                if objectives_are_list_terms(model)
                    && constraints_domain_exact_checkable(model)
                    && (summary.has_max_objective || has_free_pool_list(model) || objective_has_scan_reduction(model))
                {
                    return Self {
                        class: ModelClass::Routing,
                        backend: Backend::DomainExact,
                        reason: "edge list objective uses exact list enumeration",
                        routing: summary.routing_shape(model.lists.len()),
                    };
                }
                return Self {
                    class: ModelClass::Routing,
                    backend: Backend::ListLocalSearch,
                    reason: "edge reductions require the list local-search heuristic today",
                    routing: summary.routing_shape(model.lists.len()),
                };
            }

            if summary.has_sequence_reduction {
                // A scan/window objective is scored and checked exactly by
                // enumeration. A scan appearing only in a constraint keeps its
                // established local-search fallback unless the objective needs it.
                if objectives_are_list_terms(model)
                    && constraints_domain_exact_checkable(model)
                    && (summary.has_max_objective || has_free_pool_list(model) || objective_has_scan_reduction(model))
                {
                    return Self {
                        class: ModelClass::Sequencing,
                        backend: Backend::DomainExact,
                        reason: "sequence list objective uses exact list enumeration",
                        routing: None,
                    };
                }
                return Self {
                    class: ModelClass::Sequencing,
                    backend: Backend::ListLocalSearch,
                    reason: "scan, window, or pair reductions require the list local-search heuristic today",
                    routing: None,
                };
            }

            let list_exact = interval_constraints && (model.objectives.is_empty() || objectives_are_list_simple(model));

            if list_exact {
                return Self {
                    class: ModelClass::ListAssignmentPacking,
                    backend: Backend::DomainExact,
                    reason: if model.objectives.is_empty() {
                        "list model uses only partition, length bounds, item-sum bounds, same-list, and item precedence"
                    } else {
                        "list model uses list constraints and simple list objectives"
                    },
                    routing: summary.routing_shape(model.lists.len()),
                };
            }

            return Self {
                class: ModelClass::ListAssignmentPacking,
                backend: Backend::ListLocalSearch,
                reason: "constraints or objectives are not yet covered by domain exact search",
                routing: None,
            };
        }

        Self { class: ModelClass::PureInteger, backend: Backend::IntegerExact, reason: "integer-only model", routing: None }
    }

    /// Select a backend for the current collection model shape.
    pub fn for_collection(model: &list::CollectionModel) -> Self {
        Self::for_model(&Model::from_collection(model))
    }

    /// Select the fastest appropriate backend for an automatic collection
    /// solve. This performs cheap shape and size checks directly on the
    /// collection IR before building the shared-model mirror. Large routing,
    /// sequencing, packing, and scheduling models therefore reach local search
    /// without paying exact-backend classification.
    pub fn for_collection_auto(model: &list::CollectionModel) -> Self {
        if let Some(selection) = auto_collection_size_fallback(model) {
            return selection;
        }
        if let Some(selection) = bounded_classification_fallback(model) {
            return selection;
        }
        let selection = Self::for_collection(model);
        // Some ordered shapes only become visible after the full classifier.
        // Keep the exponential enumerator confined to deliberately small auto
        // models even when the cheap scan above was inconclusive.
        if selection.backend == Backend::DomainExact
            && matches!(selection.class, ModelClass::Routing | ModelClass::Sequencing)
            && model.items.len() > AUTO_ORDERED_EXACT_ITEMS
        {
            return heuristic_selection(model, "ordered-list size exceeds the auto exact threshold");
        }
        selection
    }

    /// Automatic selection with a caller-provided exact-backend memory ceiling.
    /// The estimate is computed from the compact IR and therefore runs before
    /// any CP variables, disjunctive pairs, or propagators are allocated.
    pub fn for_collection_auto_with_memory(model: &list::CollectionModel, max_bytes: u64) -> Self {
        let selection = Self::for_collection_auto(model);
        if selection.backend.is_exact() && estimated_exact_backend_bytes(model) > max_bytes {
            heuristic_selection(model, "estimated exact backend exceeds the memory limit")
        } else {
            selection
        }
    }

    /// Capability-oriented selection for an explicit exact request, with a
    /// deterministic upper bound on classifier work. Returning a heuristic
    /// backend tells the caller that exact classification was intentionally not
    /// attempted within the bounded preprocessing envelope.
    pub fn for_collection_exact_bounded(model: &list::CollectionModel) -> Self {
        bounded_classification_fallback(model).unwrap_or_else(|| Self::for_collection(model))
    }
}

/// Conservative allocation estimate for lowering a compact collection IR to an
/// exact backend. It is a guardrail, not a live RSS prediction. Schedule pairs
/// dominate the estimate because every disjunction creates state, propagation,
/// and search metadata in addition to its two endpoints.
pub fn estimated_exact_backend_bytes(model: &list::CollectionModel) -> u64 {
    let mut bytes = 64u128 * 1024;
    if let Some(schedule) = &model.schedule {
        let intervals = schedule.intervals.len() as u128;
        let modes = schedule.intervals.iter().map(|interval| interval.modes.len().max(1) as u128).sum::<u128>();
        bytes = bytes
            .saturating_add(intervals.saturating_mul(4 * 1024))
            .saturating_add(modes.saturating_mul(6 * 1024))
            .saturating_add((schedule.precedences.len() as u128).saturating_mul(2 * 1024));
        for resource in &schedule.resources {
            match resource {
                list::Resource::NoOverlap(group) => {
                    let size = group.len() as u128;
                    let pairs = size.saturating_mul(size.saturating_sub(1)) / 2;
                    bytes = bytes.saturating_add(pairs.saturating_mul(8 * 1024));
                }
                list::Resource::MachineNoOverlap => {
                    let mut by_machine = std::collections::HashMap::<usize, u128>::new();
                    for interval in &schedule.intervals {
                        for mode in &interval.modes {
                            *by_machine.entry(mode.machine).or_default() += 1;
                        }
                    }
                    for size in by_machine.into_values() {
                        let pairs = size.saturating_mul(size.saturating_sub(1)) / 2;
                        bytes = bytes.saturating_add(pairs.saturating_mul(8 * 1024));
                    }
                }
                list::Resource::Cumulative { demands, .. } => {
                    bytes = bytes.saturating_add((demands.len() as u128).saturating_mul(4 * 1024));
                }
            }
        }
    } else {
        let cells = (model.items.len() as u128).saturating_mul(model.lists.max(1) as u128);
        let reductions = collection_reductions(model).count().max(1) as u128;
        bytes = bytes
            .saturating_add(cells.saturating_mul(128))
            .saturating_add((model.items.len() as u128).saturating_mul(reductions).saturating_mul(256));
    }
    u64::try_from(bytes).unwrap_or(u64::MAX)
}

fn collection_reductions(model: &list::CollectionModel) -> impl Iterator<Item = &list::Reduction> {
    model.objectives.iter().flat_map(|tier| tier.reductions()).chain(model.constraints.iter().map(|constraint| &constraint.reduction))
}

#[derive(Clone, Copy, Default)]
struct CollectionShape {
    has_edges: bool,
    has_sequence: bool,
    has_scan: bool,
    reductions: usize,
}

fn collection_shape(model: &list::CollectionModel) -> CollectionShape {
    let mut shape = CollectionShape::default();
    for reduction in collection_reductions(model) {
        shape.reductions = shape.reductions.saturating_add(1);
        match &reduction.iterable {
            list::Iterable::Edges { .. } => shape.has_edges = true,
            list::Iterable::Pairs(_) | list::Iterable::Windows { .. } => shape.has_sequence = true,
            list::Iterable::Scan { .. } => {
                shape.has_sequence = true;
                shape.has_scan = true;
            }
            list::Iterable::Items(_) | list::Iterable::SetItems(_) => {}
        }
    }
    shape
}

fn heuristic_selection(model: &list::CollectionModel, reason: &'static str) -> BackendSelection {
    if model.schedule.is_some() {
        return BackendSelection { class: ModelClass::Schedule, backend: Backend::ScheduleLocalSearch, reason, routing: None };
    }
    let shape = collection_shape(model);
    let class = if shape.has_edges {
        ModelClass::Routing
    } else if shape.has_sequence {
        ModelClass::Sequencing
    } else {
        ModelClass::ListAssignmentPacking
    };
    let routing = shape.has_edges.then_some(RoutingShape {
        has_edges: true,
        multi_route: model.lists > 1,
        has_capacity_like_constraints: model.constraints.iter().any(|constraint| {
            matches!(constraint.op, list::Op::Le)
                && matches!(constraint.reduction.op, list::ReduceOp::Sum)
                && matches!(constraint.reduction.iterable, list::Iterable::Items(_) | list::Iterable::SetItems(_))
        }),
        has_scan: shape.has_scan,
        has_same_list: model.globals.iter().any(|global| matches!(global, list::GlobalConstraint::SameList { .. })),
        has_item_precedence: model.globals.iter().any(|global| matches!(global, list::GlobalConstraint::ListLe { .. })),
        has_fleet_objective: collection_reductions(model).any(|reduction| matches!(reduction.op, list::ReduceOp::Used)),
    });
    BackendSelection { class, backend: Backend::ListLocalSearch, reason, routing }
}

fn auto_collection_size_fallback(model: &list::CollectionModel) -> Option<BackendSelection> {
    if let Some(schedule) = &model.schedule {
        let modes = schedule.intervals.iter().map(|interval| interval.modes.len().max(1)).sum::<usize>();
        if schedule.intervals.len() > AUTO_SCHEDULE_EXACT_INTERVALS || modes > AUTO_SCHEDULE_EXACT_MODES {
            return Some(heuristic_selection(model, "schedule size exceeds the auto exact threshold"));
        }
        return None;
    }

    let shape = collection_shape(model);
    if shape.has_edges {
        // The integer routing lowering already has a hard node cap. Above it,
        // neither its classifier nor factorial list enumeration is appropriate
        // for an automatic solve.
        if model.items.len().saturating_add(model.lists) > MAX_INTEGER_ROUTING_NODES {
            return Some(heuristic_selection(model, "routing size exceeds the auto exact threshold"));
        }
        return None;
    }
    if shape.has_sequence && model.items.len() > AUTO_ORDERED_EXACT_ITEMS {
        return Some(heuristic_selection(model, "ordered-list size exceeds the auto exact threshold"));
    }
    if !shape.has_sequence
        && (model.items.len() > AUTO_ASSIGNMENT_EXACT_ITEMS || model.items.len().saturating_mul(model.lists) > AUTO_ASSIGNMENT_EXACT_CELLS)
    {
        return Some(heuristic_selection(model, "list assignment size exceeds the auto exact threshold"));
    }
    None
}

fn bounded_classification_fallback(model: &list::CollectionModel) -> Option<BackendSelection> {
    if let Some(schedule) = &model.schedule {
        let objects = schedule
            .intervals
            .len()
            .saturating_add(schedule.precedences.len())
            .saturating_add(schedule.resources.len())
            .saturating_add(schedule.intervals.iter().map(|interval| interval.modes.len()).sum::<usize>());
        return (objects > MAX_SCHEDULE_CLASSIFICATION_OBJECTS)
            .then(|| heuristic_selection(model, "schedule classification exceeds the bounded work threshold"));
    }

    let shape = collection_shape(model);
    let n = model.items.len() as u128;
    let reductions = shape.reductions.max(1) as u128;
    // Scan signature derivation computes an n-step accumulator fixpoint over
    // all (current, previous) node pairs. Other reductions are linear in the
    // item universe during shared-model construction and validation.
    let per_reduction = if shape.has_scan { n.saturating_mul(n).saturating_mul(n) } else { n.max(1) };
    let work = reductions.saturating_mul(per_reduction).saturating_add((model.lists as u128).saturating_mul(n));
    let construction_cells = model.items.len().saturating_mul(model.lists.max(1));
    (work > MAX_CLASSIFICATION_WORK || construction_cells > MAX_EXACT_CONSTRUCTION_CELLS)
        .then(|| heuristic_selection(model, "classification work exceeds the bounded exact threshold"))
}

#[derive(Clone, Copy, Default)]
struct ListSummary {
    has_edges: bool,
    has_scan: bool,
    has_sequence_reduction: bool,
    has_max_objective: bool,
    has_capacity_like_constraints: bool,
    has_fleet_objective: bool,
    has_same_list: bool,
    has_item_precedence: bool,
}

impl ListSummary {
    fn for_model(model: &Model) -> Self {
        let mut summary = Self::default();
        for objective in &model.objectives {
            if let Objective::ListTerms { terms, max_terms, .. } = objective {
                summary.has_max_objective |= max_terms.as_ref().is_some_and(|terms| !terms.is_empty());
                for reduction in terms.iter().chain(
                    max_terms.iter().flat_map(|terms| terms.iter().flat_map(|term| term.groups.iter().flat_map(|group| group.iter()))),
                ) {
                    summary.visit_reduction(reduction);
                    summary.has_fleet_objective |= matches!(reduction.op, list::ReduceOp::Used);
                }
            }
        }
        for constraint in &model.constraints {
            match constraint {
                Constraint::ListReduction(constraint) => {
                    summary.visit_reduction(&constraint.reduction);
                    summary.has_capacity_like_constraints |= is_item_sum_upper_bound(constraint);
                }
                Constraint::SameList { .. } => summary.has_same_list = true,
                Constraint::ItemPrecedence { .. } => summary.has_item_precedence = true,
                Constraint::CollectionGlobal(_) => {}
                Constraint::ListItemSum { .. } => summary.has_capacity_like_constraints = true,
                Constraint::ListPartition { .. }
                | Constraint::ListLength { .. }
                | Constraint::IntervalPrecedence { .. }
                | Constraint::IntervalResource(_)
                | Constraint::Intension(_) => {}
            }
        }
        summary
    }

    fn visit_reduction(&mut self, reduction: &list::Reduction) {
        match &reduction.iterable {
            list::Iterable::Edges { .. } => self.has_edges = true,
            list::Iterable::Pairs(_) | list::Iterable::Scan { .. } | list::Iterable::Windows { .. } => {
                self.has_sequence_reduction = true;
                self.has_scan |= matches!(&reduction.iterable, list::Iterable::Scan { .. });
            }
            list::Iterable::Items(_) | list::Iterable::SetItems(_) => {}
        }
    }

    fn routing_shape(self, list_count: usize) -> Option<RoutingShape> {
        self.has_edges.then_some(RoutingShape {
            has_edges: true,
            multi_route: list_count > 1,
            has_capacity_like_constraints: self.has_capacity_like_constraints,
            has_scan: self.has_scan,
            has_same_list: self.has_same_list,
            has_item_precedence: self.has_item_precedence,
            has_fleet_objective: self.has_fleet_objective,
        })
    }
}

fn objectives_are_list_terms(model: &Model) -> bool {
    !model.objectives.is_empty() && model.objectives.iter().all(|objective| matches!(objective, Objective::ListTerms { .. }))
}

/// Every constraint can be enforced by the domain-exact engine: the standard
/// list constraints post into the solver, and a `Scan` reduction constraint is
/// checked on the completed ordered lists during enumeration. A scan is admitted
/// only when it also passes the routing signature, so classify and the engines
/// agree on which scans are sound (weirder scans still fall back to local search).
fn constraints_domain_exact_checkable(model: &Model) -> bool {
    let items = model.lists.first().map_or(&[][..], |list| list.universe.as_slice());
    model.constraints.iter().all(|constraint| match constraint {
        Constraint::ListPartition { .. }
        | Constraint::SameList { .. }
        | Constraint::ItemPrecedence { .. }
        | Constraint::CollectionGlobal(_)
        | Constraint::ListLength { .. }
        | Constraint::ListItemSum { .. } => true,
        Constraint::ListReduction(constraint) => {
            matches!(constraint.reduction.iterable, list::Iterable::Scan { .. })
                && list::scan::scan_routing_signature(constraint, items).is_some()
        }
        Constraint::IntervalPrecedence { .. } | Constraint::IntervalResource(_) | Constraint::Intension(_) => false,
    })
}

/// True when an objective tier scores a scan, window, or pair reduction that
/// integer edge-lowering cannot represent, so it needs exact list enumeration.
///
/// Only a `Scan` objective is diverted here: `Windows`/`Pairs` objectives keep
/// their established local-search fallback (exact enumeration is exponential, so
/// diverting them would regress large sequencing/QAP models).
fn objective_has_scan_reduction(model: &Model) -> bool {
    model.objectives.iter().any(|objective| match objective {
        Objective::ListTerms { terms, max_terms, .. } => terms
            .iter()
            .chain(max_terms.iter().flat_map(|terms| terms.iter().flat_map(|term| term.groups.iter().flatten())))
            .any(|reduction| matches!(reduction.iterable, list::Iterable::Scan { .. })),
        Objective::IntExpr { .. } | Objective::Makespan { .. } => false,
    })
}

/// True when some list is a free "pool": referenced by no objective term and no
/// per-list constraint. An optional list model carries such a pool that silently
/// absorbs unvisited items, and only exact enumeration handles it soundly.
fn has_free_pool_list(model: &Model) -> bool {
    let n = model.lists.len();
    if n == 0 {
        return false;
    }
    let mut referenced = vec![false; n];
    for objective in &model.objectives {
        if let Objective::ListTerms { terms, max_terms, .. } = objective {
            for reduction in
                terms.iter().chain(max_terms.iter().flat_map(|terms| terms.iter().flat_map(|term| term.groups.iter().flatten())))
            {
                let list = reduction.iterable.list();
                if list < n {
                    referenced[list] = true;
                }
            }
        }
    }
    for constraint in &model.constraints {
        match constraint {
            Constraint::ListLength { list, .. } | Constraint::ListItemSum { list, .. } => referenced[list.0] = true,
            Constraint::ListReduction(constraint) => {
                let list = constraint.reduction.iterable.list();
                if list < n {
                    referenced[list] = true;
                }
            }
            _ => {}
        }
    }
    referenced.iter().any(|seen| !seen)
}

fn objectives_are_list_simple(model: &Model) -> bool {
    model.objectives.iter().all(|objective| match objective {
        Objective::ListTerms { terms, max_terms, .. } => terms
            .iter()
            .chain(max_terms.iter().flat_map(|terms| terms.iter().flat_map(|term| term.groups.iter().flat_map(|group| group.iter()))))
            .all(is_list_simple_objective_reduction),
        Objective::IntExpr { .. } | Objective::Makespan { .. } => false,
    })
}

fn is_list_simple_objective_reduction(reduction: &list::Reduction) -> bool {
    matches!(&reduction.iterable, list::Iterable::Items(_) | list::Iterable::SetItems(_))
        && match reduction.op {
            list::ReduceOp::Sum | list::ReduceOp::Count => expr_uses_at_most_first_arg(&reduction.arena.exprs, reduction.body),
            list::ReduceOp::Used => true,
            list::ReduceOp::Min | list::ReduceOp::Max | list::ReduceOp::SelectKth(_) => false,
        }
}

fn routing_integer_lowering_supported(model: &Model) -> bool {
    if !(model.int_vars.is_empty() && model.intervals.is_empty() && !model.lists.is_empty() && model.objectives.len() == 1) {
        return false;
    }
    let items = &model.lists[0].universe;
    if model.lists.len().saturating_add(items.len()) > MAX_INTEGER_ROUTING_NODES
        || !model.lists.iter().all(|list| list.universe == *items)
        || !items_are_unique(items)
    {
        return false;
    }
    let Objective::ListTerms { minimize: true, terms, max_terms } = &model.objectives[0] else {
        return false;
    };
    if max_terms.as_ref().is_some_and(|terms| !terms.is_empty()) {
        return false;
    }

    // The objective tier is closed-edge terms (one per real route) plus optional
    // reified `Sum`-scan terms (one per real route); anything else is unsupported.
    let mut edge_terms = Vec::new();
    let mut scan_terms = Vec::new();
    for term in terms {
        match &term.iterable {
            list::Iterable::Edges { .. } => edge_terms.push(term),
            // A reified `Sum` scan, or a per-item `Sum` reward normalising to one
            // (kept in lockstep with `RoutingSpec::from_model` via the shared
            // `scan_routing_signature` parser both funnel through).
            list::Iterable::Scan { .. } => scan_terms.push(term),
            list::Iterable::Items(_) if matches!(term.op, list::ReduceOp::Sum) => scan_terms.push(term),
            _ => return false,
        }
    }

    // A list referenced by no objective term and no per-list constraint is a free
    // pool. Exactly one => optional model with `k = lists - 1` real routes; zero =>
    // today's non-optional path; more than one => unsupported (fall to enumeration).
    let n_lists = model.lists.len();
    let Some(pool_list) = single_free_pool_list(model, terms, n_lists) else {
        // `single_free_pool_list` returns `None` for zero or >1 free lists; only >1
        // is unsupported. Distinguish by recomputing the free count.
        return free_pool_count(model, terms, n_lists) == 0
            && edges_and_side_supported(model, &edge_terms, &scan_terms, items, n_lists, None);
    };
    edges_and_side_supported(model, &edge_terms, &scan_terms, items, n_lists, Some(pool_list))
}

/// Number of lists referenced by no objective term and no per-list constraint.
fn free_pool_count(model: &Model, terms: &[list::Reduction], n_lists: usize) -> usize {
    pool_referenced(model, terms, n_lists).iter().filter(|&&seen| !seen).count()
}

/// The single free-pool list index, or `None` when the free count is not exactly 1.
fn single_free_pool_list(model: &Model, terms: &[list::Reduction], n_lists: usize) -> Option<usize> {
    let referenced = pool_referenced(model, terms, n_lists);
    let mut free = (0..n_lists).filter(|&list| !referenced[list]);
    let first = free.next()?;
    free.next().is_none().then_some(first)
}

fn pool_referenced(model: &Model, terms: &[list::Reduction], n_lists: usize) -> Vec<bool> {
    let mut referenced = vec![false; n_lists];
    for term in terms {
        let list = term.iterable.list();
        if list < n_lists {
            referenced[list] = true;
        }
    }
    for constraint in &model.constraints {
        match constraint {
            Constraint::ListLength { list, .. } | Constraint::ListItemSum { list, .. } => referenced[list.0] = true,
            Constraint::ListReduction(constraint) => {
                let list = constraint.reduction.iterable.list();
                if list < n_lists {
                    referenced[list] = true;
                }
            }
            _ => {}
        }
    }
    referenced
}

/// Shared tail of the routing gate once the pool structure is known: the `k` real
/// routes carry a homogeneous closed-edge objective (and optional homogeneous scan
/// objective), and the side constraints are supported. `pool` is the free-pool
/// list index when the model is optional, or `None` for the non-optional path.
fn edges_and_side_supported(
    model: &Model,
    edge_terms: &[&list::Reduction],
    scan_terms: &[&list::Reduction],
    items: &[i32],
    n_lists: usize,
    pool: Option<usize>,
) -> bool {
    let k = if pool.is_some() { n_lists - 1 } else { n_lists };
    if k == 0 || !edges_homogeneous(edge_terms, k, items) {
        return false;
    }
    if !scan_terms.is_empty() {
        // A reified scan objective is only lowered alongside a pool (a non-pool
        // scan objective keeps its established enumeration path).
        if pool.is_none() || scan_terms.len() != k || !scans_homogeneous_objective(scan_terms, k, items) {
            return false;
        }
    }
    match pool {
        Some(pool_list) => pool_side_constraints_supported(model, items, pool_list, n_lists),
        None => list_constraints_are_integer_routing_supported(model, items),
    }
}

/// A homogeneous closed-edge objective over exactly `k` real routes: `k` distinct
/// lists, identical depot and matrix. The lists need not be `0..k` (a pool model's
/// real lists are the objective-referenced subset).
fn edges_homogeneous(edge_terms: &[&list::Reduction], k: usize, items: &[i32]) -> bool {
    if edge_terms.len() != k {
        return false;
    }
    let mut seen: Vec<usize> = Vec::with_capacity(k);
    let mut base: Option<EdgeSignature<'_>> = None;
    for term in edge_terms {
        let Some((list, signature)) = direct_closed_edge_signature(term, items) else {
            return false;
        };
        if seen.contains(&list) {
            return false;
        }
        seen.push(list);
        match &base {
            None => base = Some(signature),
            Some(base) => {
                if signature.depot != base.depot || signature.reversed != base.reversed || signature.matrix != base.matrix {
                    return false;
                }
            }
        }
    }
    true
}

/// A homogeneous reified `Sum`-scan objective over `k` distinct real routes.
fn scans_homogeneous_objective(scan_terms: &[&list::Reduction], k: usize, items: &[i32]) -> bool {
    if scan_terms.len() != k {
        return false;
    }
    let mut seen: Vec<usize> = Vec::with_capacity(k);
    let mut base: Option<list::scan::ScanRoutingSpec> = None;
    for term in scan_terms {
        let list = term.iterable.list();
        if seen.contains(&list) {
            return false;
        }
        seen.push(list);
        let synth = list::Constraint { reduction: (*term).clone(), op: list::Op::Le, rhs: 0 };
        let Some(spec) = list::scan::scan_routing_signature(&synth, items).filter(|s| s.end.is_none()) else {
            return false;
        };
        match &base {
            None => base = Some(spec),
            Some(base) => {
                if &spec != base {
                    return false;
                }
            }
        }
    }
    true
}

/// A pool model admits only homogeneous `Sum`-scan side constraints over the real
/// routes (the pool list carries none). Capacity, length, item-sum, same-list, and
/// precedence are rejected (they would wrongly cap or count the free pool route).
fn pool_side_constraints_supported(model: &Model, items: &[i32], pool_list: usize, n_lists: usize) -> bool {
    let mut scans: Vec<Vec<list::scan::ScanRoutingSpec>> = (0..n_lists).map(|_| Vec::new()).collect();
    for constraint in &model.constraints {
        match constraint {
            Constraint::ListPartition { .. } => {}
            Constraint::ListReduction(reduction) if list::scan::scan_constraint_list(reduction).is_some() => {
                let (Some(list), Some(spec)) = (
                    list::scan::scan_constraint_list(reduction),
                    list::scan::scan_routing_signature(reduction, items).filter(|s| s.end.is_none()),
                ) else {
                    return false;
                };
                if list >= n_lists || list == pool_list {
                    return false;
                }
                scans[list].push(spec);
            }
            _ => return false,
        }
    }
    let mut real = (0..n_lists).filter(|&list| list != pool_list).map(|list| &scans[list]);
    match real.next() {
        None => true,
        Some(first) => real.all(|list_scans| list_scans == first),
    }
}

struct EdgeSignature<'a> {
    depot: i32,
    reversed: bool,
    matrix: &'a [Vec<i64>],
}

fn direct_closed_edge_signature<'a>(reduction: &'a list::Reduction, items: &[i32]) -> Option<(usize, EdgeSignature<'a>)> {
    if !matches!(reduction.op, list::ReduceOp::Sum) || reduction.coeff != 1 {
        return None;
    }
    let list::Iterable::Edges { list: reduction_list, start, end } = &reduction.iterable else {
        return None;
    };
    if start != end || items.contains(start) {
        return None;
    }
    match reduction.arena.exprs.get(reduction.body.0 as usize) {
        Some(list::Expr::Matrix(matrix, row, col)) => {
            let direct = expr_is_arg(&reduction.arena.exprs, *row, 0) && expr_is_arg(&reduction.arena.exprs, *col, 1);
            let reversed = expr_is_arg(&reduction.arena.exprs, *row, 1) && expr_is_arg(&reduction.arena.exprs, *col, 0);
            if !(direct || reversed) || !matrix_covers_nodes(matrix, *start, items) || !matrix_values_fit_i32(matrix, *start, items) {
                return None;
            }
            Some((*reduction_list, EdgeSignature { depot: *start, reversed, matrix: matrix.as_ref() }))
        }
        _ => None,
    }
}

fn list_constraints_are_integer_routing_supported(model: &Model, items: &[i32]) -> bool {
    let mut capacities: Vec<Option<(Vec<i64>, i64)>> = (0..model.lists.len()).map(|_| None).collect();
    let mut lengths: Vec<Option<(usize, usize)>> = (0..model.lists.len()).map(|_| None).collect();
    let mut item_sums: Vec<Option<ItemSumSignature>> = (0..model.lists.len()).map(|_| None).collect();
    let mut scans: Vec<Vec<list::scan::ScanRoutingSpec>> = (0..model.lists.len()).map(|_| Vec::new()).collect();
    for constraint in &model.constraints {
        match constraint {
            // A `Sum` scan lowers to a per-customer resource extension; anything
            // else on `ListReduction` stays with local search. A route may carry
            // several scans (e.g. one time window per customer).
            Constraint::ListReduction(reduction) if list::scan::scan_constraint_list(reduction).is_some() => {
                let (Some(list), Some(spec)) = (
                    list::scan::scan_constraint_list(reduction),
                    list::scan::scan_routing_signature(reduction, items).filter(|s| s.end.is_none()),
                ) else {
                    return false;
                };
                if list >= scans.len() {
                    return false;
                }
                scans[list].push(spec);
            }
            Constraint::ListPartition { .. } => {}
            Constraint::SameList { a, b, .. } => {
                if !items.contains(a) || !items.contains(b) {
                    return false;
                }
            }
            Constraint::ListLength { list, min, max } => {
                if list.0 >= lengths.len() || lengths[list.0].is_some() || min > max || *max > items.len() {
                    return false;
                }
                lengths[list.0] = Some((*min, *max));
            }
            Constraint::ListItemSum { list, weights, min, max } => {
                if list.0 >= capacities.len() {
                    return false;
                }
                if let Some(capacity) = capacity_signature_from_item_sum(weights, *min, *max, items) {
                    let slot = &mut capacities[list.0];
                    if slot.is_some() {
                        return false;
                    }
                    *slot = Some(capacity);
                } else {
                    let Some(item_sum) = item_sum_signature(weights, *min, *max, items) else {
                        return false;
                    };
                    let slot = &mut item_sums[list.0];
                    if slot.is_some() {
                        return false;
                    }
                    *slot = Some(item_sum);
                }
            }
            Constraint::ItemPrecedence { .. }
            | Constraint::CollectionGlobal(_)
            | Constraint::ListReduction(_)
            | Constraint::IntervalPrecedence { .. }
            | Constraint::IntervalResource(_)
            | Constraint::Intension(_) => return false,
        }
    }
    homogeneous_present(&capacities) && homogeneous_present(&lengths) && homogeneous_present(&item_sums) && scan_lists_homogeneous(&scans)
}

/// Every route must carry the identical (ordered) set of scans, or none -- a
/// route-specific scan is unsafe under depot-copy symmetry. Empty everywhere is
/// fine. Mirrors the engine's `homogeneous(&scan_by_list)`.
fn scan_lists_homogeneous(by_list: &[Vec<list::scan::ScanRoutingSpec>]) -> bool {
    match by_list.first() {
        None => true,
        Some(first) => by_list.iter().all(|scans| scans == first),
    }
}

fn homogeneous_present<T: PartialEq>(by_list: &[Option<T>]) -> bool {
    let present = by_list.iter().filter(|x| x.is_some()).count();
    if present == 0 {
        return true;
    }
    if present != by_list.len() {
        return false;
    }
    let first = by_list[0].as_ref().expect("all present");
    by_list.iter().all(|item| item.as_ref().expect("all present") == first)
}

fn capacity_signature_from_item_sum(weights: &[(i32, i64)], min: i64, max: i64, items: &[i32]) -> Option<(Vec<i64>, i64)> {
    if min > 0 || max < 0 || i32::try_from(max).is_err() {
        return None;
    }
    let values = capacity_values_for_items(weights, items)?;
    if values.iter().any(|&value| value < 0 || value > max || i32::try_from(value).is_err()) {
        return None;
    }
    Some((values, max))
}

fn item_sum_signature(weights: &[(i32, i64)], min: i64, max: i64, items: &[i32]) -> Option<ItemSumSignature> {
    let values = capacity_values_for_items(weights, items)?;
    let weights = items.iter().copied().zip(values).collect::<Vec<_>>();
    Some((weights, min, max))
}

fn capacity_values_for_items(weights: &[(i32, i64)], items: &[i32]) -> Option<Vec<i64>> {
    let mut values = Vec::with_capacity(items.len());
    for &item in items {
        values.push(weights.iter().find_map(|(known, value)| (*known == item).then_some(*value))?);
    }
    Some(values)
}

fn expr_is_arg(exprs: &[list::Expr], id: list::ExprId, arg: u8) -> bool {
    matches!(exprs.get(id.0 as usize), Some(list::Expr::Arg(found)) if *found == arg)
}

fn items_are_unique(items: &[i32]) -> bool {
    for (i, &item) in items.iter().enumerate() {
        if items[i + 1..].contains(&item) {
            return false;
        }
    }
    true
}

fn matrix_covers_nodes(matrix: &[Vec<i64>], depot: i32, items: &[i32]) -> bool {
    let mut nodes = Vec::with_capacity(items.len() + 1);
    nodes.push(depot);
    nodes.extend_from_slice(items);
    let Some(depot) = usize::try_from(depot).ok() else {
        return false;
    };
    if depot >= matrix.len() {
        return false;
    }
    for &from in &nodes {
        let Some(from) = usize::try_from(from).ok() else {
            return false;
        };
        let Some(row) = matrix.get(from) else {
            return false;
        };
        if nodes.iter().any(|&to| usize::try_from(to).ok().is_none_or(|to| row.len() <= to)) {
            return false;
        }
    }
    true
}

fn matrix_values_fit_i32(matrix: &[Vec<i64>], depot: i32, items: &[i32]) -> bool {
    let mut nodes = Vec::with_capacity(items.len() + 1);
    nodes.push(depot);
    nodes.extend_from_slice(items);
    nodes.iter().all(|&from| {
        let from = from as usize;
        nodes.iter().all(|&to| i32::try_from(matrix[from][to as usize]).is_ok())
    })
}

fn expr_uses_at_most_first_arg(arena: &[list::Expr], id: list::ExprId) -> bool {
    let Some(expr) = arena.get(id.0 as usize) else {
        return false;
    };
    match expr {
        list::Expr::Const(_) | list::Expr::Arg(0) => true,
        list::Expr::Arg(_) | list::Expr::Matrix(_, _, _) => false,
        list::Expr::Array(_, index)
        | list::Expr::Abs(index)
        | list::Expr::Pow(index, _)
        | list::Expr::PiecewiseLinear { input: index, .. } => expr_uses_at_most_first_arg(arena, *index),
        list::Expr::Add(a, b)
        | list::Expr::Sub(a, b)
        | list::Expr::Mul(a, b)
        | list::Expr::Mod(a, b)
        | list::Expr::MulScaled(a, b, _)
        | list::Expr::DivScaled(a, b, _)
        | list::Expr::Min(a, b)
        | list::Expr::Max(a, b)
        | list::Expr::Div(a, b)
        | list::Expr::Lt(a, b)
        | list::Expr::Le(a, b)
        | list::Expr::Eq(a, b)
        | list::Expr::Ne(a, b) => expr_uses_at_most_first_arg(arena, *a) && expr_uses_at_most_first_arg(arena, *b),
        list::Expr::IfThenElse(c, a, b) => {
            expr_uses_at_most_first_arg(arena, *c) && expr_uses_at_most_first_arg(arena, *a) && expr_uses_at_most_first_arg(arena, *b)
        }
        list::Expr::External { args, .. } => args.iter().all(|id| expr_uses_at_most_first_arg(arena, *id)),
    }
}

pub(crate) fn length_bound_from_collection_constraint(constraint: &list::Constraint, item_count: usize) -> Option<(usize, usize, usize)> {
    let list::Iterable::Items(list) = &constraint.reduction.iterable else {
        return None;
    };
    if matches!(constraint.reduction.op, list::ReduceOp::Used) {
        if constraint.reduction.coeff != 1 {
            return None;
        }
        return used_bound_as_length(*list, constraint.op, constraint.rhs, item_count);
    }
    if !matches!(constraint.reduction.op, list::ReduceOp::Count) {
        return None;
    }
    if constraint.reduction.coeff != 1 {
        return None;
    }
    if !matches!(
        constraint.reduction.arena.exprs.get(constraint.reduction.body.0 as usize),
        Some(list::Expr::Const(value)) if *value != 0
    ) {
        return None;
    }

    let n = item_count as i64;
    let (min, max) = match constraint.op {
        list::Op::Le => (0, constraint.rhs),
        list::Op::Ge => (constraint.rhs, n),
        list::Op::Eq => (constraint.rhs, constraint.rhs),
    };
    if max < 0 || min > n || min > max {
        return Some((*list, 1, 0));
    }
    Some((*list, min.max(0) as usize, max.min(n) as usize))
}

fn used_bound_as_length(list: usize, op: list::Op, rhs: i64, item_count: usize) -> Option<(usize, usize, usize)> {
    let n = item_count;
    let (min, max) = match op {
        list::Op::Le if rhs < 0 => return Some((list, 1, 0)),
        list::Op::Le if rhs == 0 => (0, 0),
        list::Op::Le => (0, n),
        list::Op::Ge if rhs <= 0 => (0, n),
        list::Op::Ge if rhs == 1 => (1, n),
        list::Op::Ge => return Some((list, 1, 0)),
        list::Op::Eq if rhs == 0 => (0, 0),
        list::Op::Eq if rhs == 1 => (1, n),
        list::Op::Eq => return Some((list, 1, 0)),
    };
    Some((list, min, max))
}

fn is_item_sum_upper_bound(constraint: &list::Constraint) -> bool {
    matches!(constraint.op, list::Op::Le)
        && matches!(constraint.reduction.op, list::ReduceOp::Sum)
        && matches!(&constraint.reduction.iterable, list::Iterable::Items(_))
}

pub(crate) fn item_sum_bound_from_collection_constraint(constraint: &list::Constraint, items: &[i32]) -> Option<ItemSumBound> {
    if !matches!(constraint.reduction.op, list::ReduceOp::Sum) {
        return None;
    }
    let list::Iterable::Items(list) = &constraint.reduction.iterable else {
        return None;
    };
    let mut weights = Vec::with_capacity(items.len());
    for &item in items {
        weights.push((
            item,
            eval_collection_expr_one(&constraint.reduction.arena.exprs, constraint.reduction.body, i64::from(item))?
                .saturating_mul(constraint.reduction.coeff),
        ));
    }
    let (min, max) = match constraint.op {
        list::Op::Le => (i64::MIN / 4, constraint.rhs),
        list::Op::Ge => (constraint.rhs, i64::MAX / 4),
        list::Op::Eq => (constraint.rhs, constraint.rhs),
    };
    Some((*list, weights, min, max))
}

pub(crate) fn eval_collection_expr_one(arena: &[list::Expr], id: list::ExprId, arg0: i64) -> Option<i64> {
    list::eval_expr_checked(arena, id, &[arg0])
}
