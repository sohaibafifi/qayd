//! Shared Rust model surface: the canonical IR frontends build and backend
//! compilers read. Declarations split by concern into [`int`], [`list`],
//! [`interval`], and [`objective`]. The orchestrator selects only among plans
//! that have already compiled successfully.

mod compile;
mod cp_compile;
mod decompose;
pub mod expression;
pub mod int;
pub mod interval;
pub mod list;
pub mod objective;
pub mod package;
pub mod set;
mod validate;

pub(crate) use compile::*;
pub(crate) use cp_compile::*;
pub(crate) use decompose::*;
pub use expression::*;
pub use int::*;
pub use interval::*;
pub use objective::*;
pub use package::*;
pub use set::*;
// The list-reduction IR (CollectionModel, Reduction, Expr, ...) stays under
// `list::` so its `Expr` does not clash with the intension `Expr`; only the
// model-level list surface is re-exported at the model root.
#[cfg(test)]
pub(crate) use list::list_objective_tiers;
pub(crate) use list::{evaluate_list_objectives, list_objectives_better};
pub use list::{
    ListDecl, ListIterable, ListMaxTerm, ListObjectiveTerm, ListObjectiveTier, ListOrdering, ListReduceOp, ListReduction,
    ListReductionConstraint, ListRole, ListVarRef,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstraintRef(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectiveRef(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionCoverage {
    Exact,
    Partial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntervalEndpoint {
    Start,
    End,
}

/// Constraint declarations owned by the shared model.
#[derive(Clone)]
pub enum Constraint {
    /// Each item is assigned to exactly one of the listed lists.
    ListPartition { lists: Vec<ListVarRef>, items: Vec<i32> },
    /// Explicit partition semantics used by new frontends.
    ListPartitionWithCoverage { lists: Vec<ListVarRef>, items: Vec<i32>, coverage: PartitionCoverage },
    /// Two items must be assigned to the same list.
    SameList { lists: Vec<ListVarRef>, a: i32, b: i32 },
    /// `before` must be assigned to a list no later than `after`.
    ItemPrecedence { lists: Vec<ListVarRef>, before: i32, after: i32 },
    /// Cross-list relation represented directly by the collection IR.
    CollectionGlobal(list::GlobalConstraint),
    /// Bounds a list length.
    ListLength { list: ListVarRef, min: usize, max: usize },
    /// Bounds a weighted sum over items in a list.
    ListItemSum { list: ListVarRef, weights: Vec<(i32, i64)>, min: i64, max: i64 },
    /// Current reduction constraint bridge.
    ListReduction(ListReductionConstraint),
    /// Fixed-duration interval precedence.
    IntervalPrecedence { before: IntervalVarRef, after: IntervalVarRef },
    /// One master interval is realized by exactly one present member.
    IntervalAlternative { master: IntervalVarRef, members: Vec<IntervalVarRef> },
    /// Relation between interval endpoints with an optional offset.
    IntervalEndpointRelation {
        left: IntervalVarRef,
        left_endpoint: IntervalEndpoint,
        relation: Relation,
        right: IntervalVarRef,
        right_endpoint: IntervalEndpoint,
        offset: i64,
    },
    /// Current scheduling resource bridge during migration.
    IntervalResource(list::Resource),
    /// Integer expression constraint.
    Intension(IntExpr),
    /// A constraint enabled only when its semantic Boolean selector is `1`.
    /// This keeps soft-group ownership in the model while allowing MUS services
    /// to activate candidate groups through ordinary semantic assumptions.
    Selected { selector: IntVarRef, constraint: Box<Constraint> },
    /// Normalized linear relation.
    Linear { terms: Vec<(i64, IntVarRef)>, relation: Relation, rhs: i64 },
    /// Boolean clause. An empty clause is a valid contradiction.
    Clause(Vec<BoolLiteral>),
    /// Normalized integer global constraint.
    IntegerGlobal(IntGlobalConstraint),
    /// Set inclusion.
    SetSubset { subset: SetVarRef, superset: SetVarRef },
    /// Set disjointness.
    SetDisjoint { left: SetVarRef, right: SetVarRef },
    /// Set cardinality bounds.
    SetCardinality { set: SetVarRef, min: usize, max: usize },
}

/// Shared Rust model declaration.
#[derive(Clone, Default)]
pub struct Model {
    /// Integer variables.
    pub(crate) int_vars: Vec<IntDomain>,
    /// Unordered finite-set variables.
    pub(crate) sets: Vec<SetDecl>,
    /// Structured list variables.
    pub(crate) lists: Vec<ListDecl>,
    /// Structured interval variables.
    pub(crate) intervals: Vec<IntervalDecl>,
    /// Stable interval execution modes.
    pub(crate) interval_modes: Vec<IntervalMode>,
    /// Constraints over any supported model object.
    pub(crate) constraints: Vec<Constraint>,
    /// Lexicographic objective tiers in declaration order.
    pub(crate) objectives: Vec<Objective>,
}

impl Model {
    /// Create an empty model.
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(feature = "python")]
    pub(crate) fn clear_list_family(&mut self) {
        self.lists.clear();
        self.constraints.retain(|constraint| {
            !matches!(
                constraint,
                Constraint::ListPartition { .. }
                    | Constraint::ListPartitionWithCoverage { .. }
                    | Constraint::SameList { .. }
                    | Constraint::ItemPrecedence { .. }
                    | Constraint::CollectionGlobal(_)
                    | Constraint::ListLength { .. }
                    | Constraint::ListItemSum { .. }
                    | Constraint::ListReduction(_)
            )
        });
        self.objectives.retain(|objective| !matches!(objective, Objective::ListTerms { .. }));
    }

    #[cfg(feature = "python")]
    pub(crate) fn clear_interval_family(&mut self) {
        self.intervals.clear();
        self.interval_modes.clear();
        self.constraints.retain(|constraint| {
            !matches!(
                constraint,
                Constraint::IntervalPrecedence { .. }
                    | Constraint::IntervalAlternative { .. }
                    | Constraint::IntervalEndpointRelation { .. }
                    | Constraint::IntervalResource(_)
            )
        });
        self.objectives.retain(|objective| !matches!(objective, Objective::Makespan { .. }));
    }

    /// Declare an integer variable with domain `[lo, hi]`.
    pub fn int_range(&mut self, lo: i32, hi: i32) -> IntVarRef {
        let id = IntVarRef(self.int_vars.len());
        self.int_vars.push(IntDomain::Range { lo, hi });
        id
    }

    /// Declare a Boolean variable.
    pub fn bool_var(&mut self) -> IntVarRef {
        let id = IntVarRef(self.int_vars.len());
        self.int_vars.push(IntDomain::Bool);
        id
    }

    /// Declare an integer variable with an explicit finite domain.
    pub fn int_set(&mut self, values: Vec<i32>) -> IntVarRef {
        let id = IntVarRef(self.int_vars.len());
        self.int_vars.push(IntDomain::Set(values));
        id
    }

    /// Declare an unordered set with lower and upper bounds.
    pub fn set_bounds(
        &mut self,
        required: impl IntoIterator<Item = i32>,
        possible: impl IntoIterator<Item = i32>,
    ) -> Result<SetVarRef, String> {
        let declaration = SetDecl::new(required, possible)?;
        let id = SetVarRef(self.sets.len());
        self.sets.push(declaration);
        Ok(id)
    }

    /// Declare an unconstrained subset of the given finite universe.
    pub fn set(&mut self, universe: impl IntoIterator<Item = i32>) -> SetVarRef {
        let id = SetVarRef(self.sets.len());
        self.sets.push(SetDecl::new([], universe).expect("an empty lower set is always valid"));
        id
    }

    /// Declare a list variable.
    pub fn list(&mut self, universe: Vec<i32>) -> ListVarRef {
        self.list_with(universe, ListOrdering::Ordered, ListRole::Decision)
    }

    pub fn unordered_list(&mut self, universe: Vec<i32>) -> ListVarRef {
        self.list_with(universe, ListOrdering::Unordered, ListRole::Decision)
    }

    pub fn remainder_list(&mut self, universe: Vec<i32>, ordering: ListOrdering) -> ListVarRef {
        self.list_with(universe, ordering, ListRole::HiddenRemainder)
    }

    fn list_with(&mut self, universe: Vec<i32>, ordering: ListOrdering, role: ListRole) -> ListVarRef {
        let id = ListVarRef(self.lists.len());
        self.lists.push(ListDecl { universe, ordering, role });
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
    pub fn add_constraint(&mut self, constraint: Constraint) -> ConstraintRef {
        let id = ConstraintRef(self.constraints.len());
        self.constraints.push(constraint);
        id
    }

    /// Add a lexicographic objective tier.
    pub fn add_objective(&mut self, objective: Objective) -> ObjectiveRef {
        let id = ObjectiveRef(self.objectives.len());
        self.objectives.push(objective);
        id
    }

    pub fn int_vars(&self) -> &[IntDomain] {
        &self.int_vars
    }

    pub fn sets(&self) -> &[SetDecl] {
        &self.sets
    }

    pub fn lists(&self) -> &[ListDecl] {
        &self.lists
    }

    pub fn intervals(&self) -> &[IntervalDecl] {
        &self.intervals
    }

    pub fn interval_modes(&self) -> &[IntervalMode] {
        &self.interval_modes
    }

    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    pub fn objectives(&self) -> &[Objective] {
        &self.objectives
    }

    pub fn add_interval_mode(
        &mut self,
        interval: IntervalVarRef,
        machine: usize,
        duration: i64,
        start_window: Option<(i64, i64)>,
    ) -> Result<IntervalModeRef, String> {
        if interval.0 >= self.intervals.len() {
            return Err("mode references an unknown interval".to_string());
        }
        let mode = IntervalModeRef(self.interval_modes.len());
        self.interval_modes.push(IntervalMode { interval, machine, duration, start_window });
        self.intervals[interval.0].modes.push(mode);
        Ok(mode)
    }

    /// Mirror the current collection model into the shared model surface.
    #[cfg(test)]
    pub(crate) fn from_collection(collection_model: &list::CollectionModel) -> Self {
        let mut model = Self::new();
        if let Some(schedule) = &collection_model.schedule {
            let intervals = schedule
                .intervals
                .iter()
                .map(|interval| {
                    let duration = if interval.modes.is_empty() {
                        interval.duration
                    } else {
                        interval.modes.iter().map(|mode| mode.duration).min().unwrap_or(interval.duration)
                    };
                    let start_max = interval.horizon.saturating_sub(duration);
                    let id = IntervalVarRef(model.intervals.len());
                    model.intervals.push(IntervalDecl {
                        start_min: 0,
                        start_max,
                        duration: interval.duration,
                        modes: Vec::new(),
                        optional: interval.optional,
                    });
                    for mode in &interval.modes {
                        model
                            .add_interval_mode(id, mode.machine, mode.duration, Some(mode.start_window))
                            .expect("collection schedule mode references its owning interval");
                    }
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
            if let Some((list, min, max)) = compile::length_bound_from_collection_constraint(constraint, collection_model.items.len()) {
                model.add_constraint(Constraint::ListLength { list: lists[list], min, max });
            } else if let Some((list, weights, min, max)) =
                compile::item_sum_bound_from_collection_constraint(constraint, &collection_model.items)
            {
                model.add_constraint(Constraint::ListItemSum { list: lists[list], weights, min, max });
            } else {
                model.add_constraint(Constraint::ListReduction(constraint.clone()));
            }
        }
        for global in &collection_model.globals {
            match global {
                list::GlobalConstraint::ListLe { before, after } => {
                    model.add_constraint(Constraint::ItemPrecedence { lists: lists.clone(), before: *before, after: *after });
                }
                list::GlobalConstraint::SameList { a, b } => {
                    model.add_constraint(Constraint::SameList { lists: lists.clone(), a: *a, b: *b });
                }
                other => {
                    model.add_constraint(Constraint::CollectionGlobal(other.clone()));
                }
            }
        }
        for tier in &collection_model.objectives {
            model.add_objective(Objective::ListTerms {
                minimize: tier.minimize,
                terms: tier.terms.clone(),
                max_terms: tier.max_terms.clone(),
            });
        }
        model
    }

    fn interval_with_presence(&mut self, start_min: i64, start_max: i64, duration: i64, optional: bool) -> IntervalVarRef {
        let id = IntervalVarRef(self.intervals.len());
        self.intervals.push(IntervalDecl { start_min, start_max, duration, modes: Vec::new(), optional });
        id
    }
}
