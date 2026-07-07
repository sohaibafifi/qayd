//! Shared Rust model surface: the canonical IR frontends build and engines
//! read. Declarations split by concern into [`int`], [`list`], [`interval`],
//! [`objective`]; [`classify`] picks a backend. The list-reduction IR lives
//! under [`list`].

use crate::expr::Expr;

pub mod classify;
pub mod int;
pub mod interval;
pub mod list;
pub mod objective;

pub use classify::*;
pub use int::*;
pub use interval::*;
pub use objective::*;
// The list-reduction IR (CollectionModel, Reduction, Expr, ...) stays under
// `list::` so its `Expr` does not clash with the intension `Expr`; only the
// model-level list surface is re-exported at the model root.
pub use list::{
    evaluate_list_objectives, list_objective_tiers, list_objectives_better, ListDecl, ListIterable, ListObjectiveTerm, ListObjectiveTier,
    ListReduceOp, ListReduction, ListReductionConstraint, ListVarRef,
};

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
    IntervalResource(list::Resource),
    /// Integer expression constraint.
    Intension(Expr),
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

    /// Declare a list variable.
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
    pub fn from_collection(collection_model: &list::CollectionModel) -> Self {
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
            if let Some((list, min, max)) = classify::length_bound_from_collection_constraint(constraint, collection_model.items.len()) {
                model.add_constraint(Constraint::ListLength { list: lists[list], min, max });
            } else if let Some((list, weights, min, max)) =
                classify::item_sum_bound_from_collection_constraint(constraint, &collection_model.items)
            {
                model.add_constraint(Constraint::ListItemSum { list: lists[list], weights, min, max });
            } else {
                model.add_constraint(Constraint::ListReduction(constraint.clone()));
            }
        }
        for global in &collection_model.globals {
            match *global {
                list::GlobalConstraint::ListLe { before, after } => {
                    model.add_constraint(Constraint::ItemPrecedence { lists: lists.clone(), before, after });
                }
                list::GlobalConstraint::SameList { a, b } => {
                    model.add_constraint(Constraint::SameList { lists: lists.clone(), a, b });
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

impl From<&list::CollectionModel> for Model {
    fn from(collection_model: &list::CollectionModel) -> Self {
        Self::from_collection(collection_model)
    }
}
