//! Shared Rust model surface.
//!
//! This module is the target home for solver-level model semantics. Frontends
//! should eventually build this representation first, then backend selection
//! can decide whether to run integer exact search, structured exact search, or
//! a heuristic engine.
//!
//! The list-reduction aliases deliberately point at `collection` for now. That
//! keeps this first step small while making the migration boundary explicit:
//! the data shape is public here, and the implementation can move out of
//! `collection` incrementally.

use crate::collection;
use crate::expr::Expr;

/// Reference to an integer variable declaration inside [`Model`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct IntVarRef(pub usize);

/// Reference to a structured list declaration inside [`Model`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ListVarRef(pub usize);

/// Reference to a structured interval declaration inside [`Model`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct IntervalVarRef(pub usize);

/// Integer variable domain declaration.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum IntDomain {
    /// Contiguous integer range.
    Range { lo: i32, hi: i32 },
    /// Explicit finite set.
    Set(Vec<i32>),
}

/// Structured list declaration.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ListDecl {
    /// Immutable item universe for this list.
    pub universe: Vec<i32>,
}

/// Structured interval declaration.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct IntervalDecl {
    /// Earliest start.
    pub start_min: i64,
    /// Latest start.
    pub start_max: i64,
    /// Fixed duration.
    pub duration: i64,
    /// Alternative execution modes. Empty means fixed duration and no machine.
    pub modes: Vec<IntervalMode>,
    /// Whether the interval may be absent.
    pub optional: bool,
}

/// Alternative interval execution mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IntervalMode {
    /// Machine index.
    pub machine: usize,
    /// Mode duration.
    pub duration: i64,
}

/// Current list reduction representation during the migration.
pub type ListReduction = collection::Reduction;

/// Current list constraint representation during the migration.
pub type ListReductionConstraint = collection::Constraint;

/// Current list reduction operator representation during the migration.
pub type ListReduceOp = collection::ReduceOp;

/// Current list iterable representation during the migration.
pub type ListIterable = collection::Iterable;

type ItemSumBound = (usize, Vec<(i32, i64)>, i64, i64);

/// Constraint declarations owned by the shared model.
#[derive(Clone)]
pub enum Constraint {
    /// Each item is assigned to exactly one of the listed lists.
    ListPartition { lists: Vec<ListVarRef>, items: Vec<i32> },
    /// Two items must be assigned to the same list.
    SameList { lists: Vec<ListVarRef>, a: i32, b: i32 },
    /// `before` must be assigned to a list no later than `after`.
    ItemPrecedence { lists: Vec<ListVarRef>, before: i32, after: i32 },
    /// Bounds a list length.
    ListLength { list: ListVarRef, min: usize, max: usize },
    /// Bounds a weighted sum over items in a list.
    ListItemSum { list: ListVarRef, weights: Vec<(i32, i64)>, min: i64, max: i64 },
    /// Current reduction constraint bridge.
    ListReduction(ListReductionConstraint),
    /// Fixed-duration interval precedence.
    IntervalPrecedence { before: IntervalVarRef, after: IntervalVarRef },
    /// Current scheduling resource bridge during migration.
    IntervalResource(collection::Resource),
    /// Integer expression constraint.
    Intension(Expr),
}

/// Objective declarations owned by the shared model.
#[derive(Clone)]
pub enum Objective {
    /// Integer expression objective.
    IntExpr { minimize: bool, expr: Expr },
    /// One tier over list reductions.
    ListTerms { minimize: bool, terms: Vec<ListReduction> },
    /// Minimize or maximize the latest interval end.
    Makespan { minimize: bool, intervals: Vec<IntervalVarRef> },
}

/// Structured exact objective term over one list.
#[derive(Clone)]
pub enum StructuredListObjectiveTerm {
    /// Sum of item weights in a list.
    Sum { list: usize, weights: Vec<(i32, i64)> },
    /// Count of items whose weight/predicate value is non-zero.
    Count { list: usize, weights: Vec<(i32, i64)> },
    /// `1` when the list is non-empty, else `0`.
    Used { list: usize },
}

/// One structured exact lexicographic objective tier.
#[derive(Clone)]
pub struct StructuredListObjectiveTier {
    /// Whether this tier is minimized.
    pub minimize: bool,
    /// Terms summed to form the tier value.
    pub terms: Vec<StructuredListObjectiveTerm>,
}

/// Shared Rust model declaration.
#[derive(Clone, Default)]
pub struct Model {
    /// Integer variables.
    pub int_vars: Vec<IntDomain>,
    /// Structured list variables.
    pub lists: Vec<ListDecl>,
    /// Structured interval variables.
    pub intervals: Vec<IntervalDecl>,
    /// Constraints over any supported model object.
    pub constraints: Vec<Constraint>,
    /// Lexicographic objective tiers in declaration order.
    pub objectives: Vec<Objective>,
}

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
    /// Exact structured list / interval search.
    StructuredExact,
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
            Self::StructuredExact => "structured exact",
            Self::ListLocalSearch => "list local search",
            Self::ScheduleLocalSearch => "schedule local search",
        }
    }

    /// Whether this backend is exact.
    pub fn is_exact(self) -> bool {
        matches!(self, Self::IntegerExact | Self::StructuredExact)
    }

    /// Whether this backend is heuristic.
    pub fn is_heuristic(self) -> bool {
        !self.is_exact()
    }

    /// Stable role description for architecture checks and diagnostics.
    pub fn role(self) -> &'static str {
        match self {
            Self::IntegerExact | Self::StructuredExact => "semantic exact solver",
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

impl Model {
    /// Create an empty model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare an integer variable with domain `[lo, hi]`.
    pub fn int_range(&mut self, lo: i32, hi: i32) -> IntVarRef {
        let id = IntVarRef(self.int_vars.len());
        self.int_vars.push(IntDomain::Range { lo, hi });
        id
    }

    /// Declare an integer variable with an explicit finite domain.
    pub fn int_set(&mut self, values: Vec<i32>) -> IntVarRef {
        let id = IntVarRef(self.int_vars.len());
        self.int_vars.push(IntDomain::Set(values));
        id
    }

    /// Declare a structured list variable.
    pub fn list(&mut self, universe: Vec<i32>) -> ListVarRef {
        let id = ListVarRef(self.lists.len());
        self.lists.push(ListDecl { universe });
        id
    }

    /// Declare a mandatory fixed-duration interval.
    pub fn interval(&mut self, start_min: i64, start_max: i64, duration: i64) -> IntervalVarRef {
        self.interval_with_presence(start_min, start_max, duration, false)
    }

    /// Declare an optional fixed-duration interval.
    pub fn optional_interval(&mut self, start_min: i64, start_max: i64, duration: i64) -> IntervalVarRef {
        self.interval_with_presence(start_min, start_max, duration, true)
    }

    /// Add a constraint declaration.
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Add a lexicographic objective tier.
    pub fn add_objective(&mut self, objective: Objective) {
        self.objectives.push(objective);
    }

    /// Mirror the current collection model into the shared model surface.
    pub fn from_collection(collection_model: &collection::CollectionModel) -> Self {
        let mut model = Self::new();
        if let Some(schedule) = &collection_model.schedule {
            let intervals = schedule
                .intervals
                .iter()
                .map(|interval| {
                    let modes = interval
                        .modes
                        .iter()
                        .map(|mode| IntervalMode { machine: mode.machine, duration: mode.duration })
                        .collect::<Vec<_>>();
                    let duration = if modes.is_empty() {
                        interval.duration
                    } else {
                        modes.iter().map(|mode| mode.duration).min().unwrap_or(interval.duration)
                    };
                    let start_max = interval.horizon.saturating_sub(duration);
                    let id = IntervalVarRef(model.intervals.len());
                    model.intervals.push(IntervalDecl { start_min: 0, start_max, duration: interval.duration, modes, optional: false });
                    id
                })
                .collect::<Vec<_>>();

            for &(before, after) in &schedule.precedences {
                model.add_constraint(Constraint::IntervalPrecedence { before: intervals[before], after: intervals[after] });
            }
            for resource in &schedule.resources {
                model.add_constraint(Constraint::IntervalResource(resource.clone()));
            }
            if schedule.minimize_makespan {
                model.add_objective(Objective::Makespan { minimize: true, intervals });
            }
            return model;
        }

        let lists = (0..collection_model.lists).map(|_| model.list(collection_model.items.clone())).collect::<Vec<_>>();
        if !lists.is_empty() {
            model.add_constraint(Constraint::ListPartition { lists: lists.clone(), items: collection_model.items.clone() });
        }
        for constraint in &collection_model.constraints {
            if let Some((list, min, max)) = length_bound_from_collection_constraint(constraint, collection_model.items.len()) {
                model.add_constraint(Constraint::ListLength { list: lists[list], min, max });
            } else if let Some((list, weights, min, max)) = item_sum_bound_from_collection_constraint(constraint, &collection_model.items) {
                model.add_constraint(Constraint::ListItemSum { list: lists[list], weights, min, max });
            } else {
                model.add_constraint(Constraint::ListReduction(constraint.clone()));
            }
        }
        for global in &collection_model.globals {
            match *global {
                collection::GlobalConstraint::ListLe { before, after } => {
                    model.add_constraint(Constraint::ItemPrecedence { lists: lists.clone(), before, after });
                }
                collection::GlobalConstraint::SameList { a, b } => {
                    model.add_constraint(Constraint::SameList { lists: lists.clone(), a, b });
                }
            }
        }
        for tier in &collection_model.objectives {
            model.add_objective(Objective::ListTerms { minimize: tier.minimize, terms: tier.terms.clone() });
        }
        model
    }

    fn interval_with_presence(&mut self, start_min: i64, start_max: i64, duration: i64, optional: bool) -> IntervalVarRef {
        let id = IntervalVarRef(self.intervals.len());
        self.intervals.push(IntervalDecl { start_min, start_max, duration, modes: Vec::new(), optional });
        id
    }
}

impl From<&collection::CollectionModel> for Model {
    fn from(collection_model: &collection::CollectionModel) -> Self {
        Self::from_collection(collection_model)
    }
}

impl BackendSelection {
    /// Select a backend for a shared Rust model.
    pub fn for_model(model: &Model) -> Self {
        if !model.intervals.is_empty() {
            let structured_schedule =
                model.constraints.iter().all(|constraint| matches!(constraint, Constraint::IntervalPrecedence { .. }))
                    && model.objectives.iter().all(|objective| matches!(objective, Objective::Makespan { minimize: true, .. }))
                    && model.intervals.iter().all(|interval| interval.modes.is_empty());
            if structured_schedule {
                return Self {
                    class: ModelClass::Schedule,
                    backend: Backend::StructuredExact,
                    reason: "fixed intervals with precedences only",
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
            let structured_constraints = model.constraints.iter().all(|constraint| {
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
                return Self {
                    class: ModelClass::Routing,
                    backend: Backend::ListLocalSearch,
                    reason: "edge reductions require the list local-search heuristic today",
                    routing: summary.routing_shape(model.lists.len()),
                };
            }

            if summary.has_sequence_reduction {
                return Self {
                    class: ModelClass::Sequencing,
                    backend: Backend::ListLocalSearch,
                    reason: "scan, window, or pair reductions require the list local-search heuristic today",
                    routing: None,
                };
            }

            let structured_list = structured_constraints && (model.objectives.is_empty() || objectives_are_structured_simple(model));

            if structured_list {
                return Self {
                    class: ModelClass::ListAssignmentPacking,
                    backend: Backend::StructuredExact,
                    reason: if model.objectives.is_empty() {
                        "list model uses only partition, length bounds, item-sum bounds, same-list, and item precedence"
                    } else {
                        "list model uses structured constraints and simple list objectives"
                    },
                    routing: summary.routing_shape(model.lists.len()),
                };
            }

            return Self {
                class: ModelClass::ListAssignmentPacking,
                backend: Backend::ListLocalSearch,
                reason: "constraints or objectives are not yet covered by structured exact search",
                routing: None,
            };
        }

        Self { class: ModelClass::PureInteger, backend: Backend::IntegerExact, reason: "integer-only model", routing: None }
    }

    /// Select a backend for the current collection model shape.
    pub fn for_collection(model: &collection::CollectionModel) -> Self {
        Self::for_model(&Model::from_collection(model))
    }
}

/// Parse list objective tiers that structured exact search can evaluate.
pub fn structured_list_objective_tiers(tiers: &[collection::ObjectiveTier], items: &[i32]) -> Option<Vec<StructuredListObjectiveTier>> {
    let mut parsed = Vec::with_capacity(tiers.len());
    for tier in tiers {
        let mut terms = Vec::with_capacity(tier.terms.len());
        for reduction in &tier.terms {
            terms.push(structured_list_objective_term(reduction, items)?);
        }
        parsed.push(StructuredListObjectiveTier { minimize: tier.minimize, terms });
    }
    Some(parsed)
}

/// Evaluate parsed structured objective tiers on concrete list contents.
pub fn evaluate_structured_list_objectives(tiers: &[StructuredListObjectiveTier], lists: &[Vec<i32>]) -> Vec<i64> {
    tiers
        .iter()
        .map(|tier| {
            tier.terms.iter().fold(0i64, |acc, term| {
                acc.saturating_add(match term {
                    StructuredListObjectiveTerm::Sum { list, weights } => lists
                        .get(*list)
                        .map_or(0, |items| items.iter().fold(0i64, |sum, item| sum.saturating_add(structured_weight(weights, *item)))),
                    StructuredListObjectiveTerm::Count { list, weights } => lists.get(*list).map_or(0, |items| {
                        items.iter().filter(|item| structured_weight(weights, **item) != 0).fold(0i64, |count, _| count.saturating_add(1))
                    }),
                    StructuredListObjectiveTerm::Used { list } => i64::from(lists.get(*list).is_some_and(|items| !items.is_empty())),
                })
            })
        })
        .collect()
}

/// Lexicographic comparison of structured objective vectors.
pub fn structured_list_objectives_better(candidate: &[i64], incumbent: &[i64], tiers: &[StructuredListObjectiveTier]) -> bool {
    for ((candidate, incumbent), tier) in candidate.iter().zip(incumbent).zip(tiers) {
        if candidate == incumbent {
            continue;
        }
        return if tier.minimize { candidate < incumbent } else { candidate > incumbent };
    }
    false
}

fn structured_list_objective_term(reduction: &collection::Reduction, items: &[i32]) -> Option<StructuredListObjectiveTerm> {
    let collection::Iterable::Items(list) = &reduction.iterable else {
        return None;
    };
    match reduction.op {
        collection::ReduceOp::Sum => {
            Some(StructuredListObjectiveTerm::Sum { list: *list, weights: structured_item_values(reduction, items)? })
        }
        collection::ReduceOp::Count => {
            Some(StructuredListObjectiveTerm::Count { list: *list, weights: structured_item_values(reduction, items)? })
        }
        collection::ReduceOp::Used => Some(StructuredListObjectiveTerm::Used { list: *list }),
        collection::ReduceOp::Min | collection::ReduceOp::Max | collection::ReduceOp::SelectKth(_) => None,
    }
}

fn structured_item_values(reduction: &collection::Reduction, items: &[i32]) -> Option<Vec<(i32, i64)>> {
    let mut values = Vec::with_capacity(items.len());
    for &item in items {
        values.push((item, eval_collection_expr_one(&reduction.arena.exprs, reduction.body, i64::from(item))?));
    }
    Some(values)
}

fn structured_weight(weights: &[(i32, i64)], item: i32) -> i64 {
    debug_assert!(weights.iter().any(|(known, _)| *known == item));
    weights.iter().find_map(|(known, weight)| (*known == item).then_some(*weight)).unwrap_or(0)
}

#[derive(Clone, Copy, Default)]
struct ListSummary {
    has_edges: bool,
    has_scan: bool,
    has_sequence_reduction: bool,
    has_capacity_like_constraints: bool,
    has_fleet_objective: bool,
    has_same_list: bool,
    has_item_precedence: bool,
}

impl ListSummary {
    fn for_model(model: &Model) -> Self {
        let mut summary = Self::default();
        for objective in &model.objectives {
            if let Objective::ListTerms { terms, .. } = objective {
                for reduction in terms {
                    summary.visit_reduction(reduction);
                    summary.has_fleet_objective |= matches!(reduction.op, collection::ReduceOp::Used);
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

    fn visit_reduction(&mut self, reduction: &collection::Reduction) {
        match &reduction.iterable {
            collection::Iterable::Edges { .. } => self.has_edges = true,
            collection::Iterable::Pairs(_) | collection::Iterable::Scan { .. } | collection::Iterable::Windows { .. } => {
                self.has_sequence_reduction = true;
                self.has_scan |= matches!(&reduction.iterable, collection::Iterable::Scan { .. });
            }
            collection::Iterable::Items(_) => {}
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

fn objectives_are_structured_simple(model: &Model) -> bool {
    model.objectives.iter().all(|objective| match objective {
        Objective::ListTerms { terms, .. } => terms.iter().all(is_structured_simple_objective_reduction),
        Objective::IntExpr { .. } | Objective::Makespan { .. } => false,
    })
}

fn is_structured_simple_objective_reduction(reduction: &collection::Reduction) -> bool {
    matches!(&reduction.iterable, collection::Iterable::Items(_))
        && match reduction.op {
            collection::ReduceOp::Sum | collection::ReduceOp::Count => expr_uses_at_most_first_arg(&reduction.arena.exprs, reduction.body),
            collection::ReduceOp::Used => true,
            collection::ReduceOp::Min | collection::ReduceOp::Max | collection::ReduceOp::SelectKth(_) => false,
        }
}

fn expr_uses_at_most_first_arg(arena: &[collection::Expr], id: collection::ExprId) -> bool {
    let Some(expr) = arena.get(id.0 as usize) else {
        return false;
    };
    match expr {
        collection::Expr::Const(_) | collection::Expr::Arg(0) => true,
        collection::Expr::Arg(_) | collection::Expr::Matrix(_, _, _) => false,
        collection::Expr::Array(_, index) | collection::Expr::Abs(index) => expr_uses_at_most_first_arg(arena, *index),
        collection::Expr::Add(a, b)
        | collection::Expr::Sub(a, b)
        | collection::Expr::Mul(a, b)
        | collection::Expr::Min(a, b)
        | collection::Expr::Max(a, b)
        | collection::Expr::Div(a, b)
        | collection::Expr::Lt(a, b)
        | collection::Expr::Le(a, b)
        | collection::Expr::Eq(a, b)
        | collection::Expr::Ne(a, b) => expr_uses_at_most_first_arg(arena, *a) && expr_uses_at_most_first_arg(arena, *b),
        collection::Expr::IfThenElse(c, a, b) => {
            expr_uses_at_most_first_arg(arena, *c) && expr_uses_at_most_first_arg(arena, *a) && expr_uses_at_most_first_arg(arena, *b)
        }
    }
}

fn length_bound_from_collection_constraint(constraint: &collection::Constraint, item_count: usize) -> Option<(usize, usize, usize)> {
    let collection::Iterable::Items(list) = &constraint.reduction.iterable else {
        return None;
    };
    if matches!(constraint.reduction.op, collection::ReduceOp::Used) {
        return used_bound_as_length(*list, constraint.op, constraint.rhs, item_count);
    }
    if !matches!(constraint.reduction.op, collection::ReduceOp::Count) {
        return None;
    }
    if !matches!(
        constraint.reduction.arena.exprs.get(constraint.reduction.body.0 as usize),
        Some(collection::Expr::Const(value)) if *value != 0
    ) {
        return None;
    }

    let n = item_count as i64;
    let (min, max) = match constraint.op {
        collection::Op::Le => (0, constraint.rhs),
        collection::Op::Ge => (constraint.rhs, n),
        collection::Op::Eq => (constraint.rhs, constraint.rhs),
    };
    if max < 0 || min > n || min > max {
        return Some((*list, 1, 0));
    }
    Some((*list, min.max(0) as usize, max.min(n) as usize))
}

fn used_bound_as_length(list: usize, op: collection::Op, rhs: i64, item_count: usize) -> Option<(usize, usize, usize)> {
    let n = item_count;
    let (min, max) = match op {
        collection::Op::Le if rhs < 0 => return Some((list, 1, 0)),
        collection::Op::Le if rhs == 0 => (0, 0),
        collection::Op::Le => (0, n),
        collection::Op::Ge if rhs <= 0 => (0, n),
        collection::Op::Ge if rhs == 1 => (1, n),
        collection::Op::Ge => return Some((list, 1, 0)),
        collection::Op::Eq if rhs == 0 => (0, 0),
        collection::Op::Eq if rhs == 1 => (1, n),
        collection::Op::Eq => return Some((list, 1, 0)),
    };
    Some((list, min, max))
}

fn is_item_sum_upper_bound(constraint: &collection::Constraint) -> bool {
    matches!(constraint.op, collection::Op::Le)
        && matches!(constraint.reduction.op, collection::ReduceOp::Sum)
        && matches!(&constraint.reduction.iterable, collection::Iterable::Items(_))
}

fn item_sum_bound_from_collection_constraint(constraint: &collection::Constraint, items: &[i32]) -> Option<ItemSumBound> {
    if !matches!(constraint.reduction.op, collection::ReduceOp::Sum) {
        return None;
    }
    let collection::Iterable::Items(list) = &constraint.reduction.iterable else {
        return None;
    };
    let mut weights = Vec::with_capacity(items.len());
    for &item in items {
        weights.push((item, eval_collection_expr_one(&constraint.reduction.arena.exprs, constraint.reduction.body, i64::from(item))?));
    }
    let (min, max) = match constraint.op {
        collection::Op::Le => (i64::MIN / 4, constraint.rhs),
        collection::Op::Ge => (constraint.rhs, i64::MAX / 4),
        collection::Op::Eq => (constraint.rhs, constraint.rhs),
    };
    Some((*list, weights, min, max))
}

fn eval_collection_expr_one(arena: &[collection::Expr], id: collection::ExprId, arg0: i64) -> Option<i64> {
    let node = arena.get(id.0 as usize)?;
    Some(match node {
        collection::Expr::Const(value) => *value,
        collection::Expr::Arg(0) => arg0,
        collection::Expr::Arg(_) => return None,
        collection::Expr::Array(values, index) => {
            let index = eval_collection_expr_one(arena, *index, arg0)?;
            *values.get(usize::try_from(index).ok()?)?
        }
        collection::Expr::Matrix(_, _, _) => return None,
        collection::Expr::Add(a, b) => {
            eval_collection_expr_one(arena, *a, arg0)?.checked_add(eval_collection_expr_one(arena, *b, arg0)?)?
        }
        collection::Expr::Sub(a, b) => {
            eval_collection_expr_one(arena, *a, arg0)?.checked_sub(eval_collection_expr_one(arena, *b, arg0)?)?
        }
        collection::Expr::Mul(a, b) => {
            eval_collection_expr_one(arena, *a, arg0)?.checked_mul(eval_collection_expr_one(arena, *b, arg0)?)?
        }
        collection::Expr::Min(a, b) => eval_collection_expr_one(arena, *a, arg0)?.min(eval_collection_expr_one(arena, *b, arg0)?),
        collection::Expr::Max(a, b) => eval_collection_expr_one(arena, *a, arg0)?.max(eval_collection_expr_one(arena, *b, arg0)?),
        collection::Expr::Div(a, b) => {
            let numerator = eval_collection_expr_one(arena, *a, arg0)?;
            let denominator = eval_collection_expr_one(arena, *b, arg0)?;
            if denominator == 0 {
                0
            } else {
                numerator.checked_div(denominator)?
            }
        }
        collection::Expr::Abs(a) => eval_collection_expr_one(arena, *a, arg0)?.saturating_abs(),
        collection::Expr::Lt(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? < eval_collection_expr_one(arena, *b, arg0)?),
        collection::Expr::Le(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? <= eval_collection_expr_one(arena, *b, arg0)?),
        collection::Expr::Eq(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? == eval_collection_expr_one(arena, *b, arg0)?),
        collection::Expr::Ne(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? != eval_collection_expr_one(arena, *b, arg0)?),
        collection::Expr::IfThenElse(c, a, b) => {
            if eval_collection_expr_one(arena, *c, arg0)? != 0 {
                eval_collection_expr_one(arena, *a, arg0)?
            } else {
                eval_collection_expr_one(arena, *b, arg0)?
            }
        }
    })
}
