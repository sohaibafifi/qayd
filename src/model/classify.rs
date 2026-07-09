//! Backend selection: model classification and engine choice.

use super::*;
use crate::model::list;

type ItemSumBound = (usize, Vec<(i32, i64)>, i64, i64);
type ItemSumSignature = (Vec<(i32, i64)>, i64, i64);
const MAX_INTEGER_ROUTING_NODES: usize = 32;

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
                        | Constraint::ListLength { .. }
                        | Constraint::ListItemSum { .. }
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
            list::Iterable::Items(_) => {}
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
    matches!(&reduction.iterable, list::Iterable::Items(_))
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
        return free_pool_count(model, terms, n_lists) == 0 && edges_and_side_supported(model, &edge_terms, &scan_terms, items, n_lists, None);
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
        let Some(spec) = list::scan::scan_routing_signature(&synth, items) else {
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
                let (Some(list), Some(spec)) =
                    (list::scan::scan_constraint_list(reduction), list::scan::scan_routing_signature(reduction, items))
                else {
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
                let (Some(list), Some(spec)) = (list::scan::scan_constraint_list(reduction), list::scan::scan_routing_signature(reduction, items))
                else {
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
        list::Expr::Array(_, index) | list::Expr::Abs(index) => expr_uses_at_most_first_arg(arena, *index),
        list::Expr::Add(a, b)
        | list::Expr::Sub(a, b)
        | list::Expr::Mul(a, b)
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
    let node = arena.get(id.0 as usize)?;
    Some(match node {
        list::Expr::Const(value) => *value,
        list::Expr::Arg(0) => arg0,
        list::Expr::Arg(_) => return None,
        list::Expr::Array(values, index) => {
            let index = eval_collection_expr_one(arena, *index, arg0)?;
            *values.get(usize::try_from(index).ok()?)?
        }
        list::Expr::Matrix(_, _, _) => return None,
        list::Expr::Add(a, b) => eval_collection_expr_one(arena, *a, arg0)?.checked_add(eval_collection_expr_one(arena, *b, arg0)?)?,
        list::Expr::Sub(a, b) => eval_collection_expr_one(arena, *a, arg0)?.checked_sub(eval_collection_expr_one(arena, *b, arg0)?)?,
        list::Expr::Mul(a, b) => eval_collection_expr_one(arena, *a, arg0)?.checked_mul(eval_collection_expr_one(arena, *b, arg0)?)?,
        list::Expr::Min(a, b) => eval_collection_expr_one(arena, *a, arg0)?.min(eval_collection_expr_one(arena, *b, arg0)?),
        list::Expr::Max(a, b) => eval_collection_expr_one(arena, *a, arg0)?.max(eval_collection_expr_one(arena, *b, arg0)?),
        list::Expr::Div(a, b) => {
            let numerator = eval_collection_expr_one(arena, *a, arg0)?;
            let denominator = eval_collection_expr_one(arena, *b, arg0)?;
            if denominator == 0 {
                0
            } else {
                numerator.checked_div(denominator)?
            }
        }
        list::Expr::Abs(a) => eval_collection_expr_one(arena, *a, arg0)?.saturating_abs(),
        list::Expr::Lt(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? < eval_collection_expr_one(arena, *b, arg0)?),
        list::Expr::Le(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? <= eval_collection_expr_one(arena, *b, arg0)?),
        list::Expr::Eq(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? == eval_collection_expr_one(arena, *b, arg0)?),
        list::Expr::Ne(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? != eval_collection_expr_one(arena, *b, arg0)?),
        list::Expr::IfThenElse(c, a, b) => {
            if eval_collection_expr_one(arena, *c, arg0)? != 0 {
                eval_collection_expr_one(arena, *a, arg0)?
            } else {
                eval_collection_expr_one(arena, *b, arg0)?
            }
        }
    })
}
