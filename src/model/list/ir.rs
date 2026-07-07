//! List-reduction IR: a model partitions a universe of item ids among ordered
//! list variables, with objectives and constraints expressed as lambda
//! reductions over a list (an aggregate of a body evaluated at each item, or at
//! each edge of a route). One representation covers routing, sequencing,
//! assignment, and packing by changing only the reductions.

use std::sync::Arc;

/// Maximum number of lexicographic objective tiers a model may declare.
pub const MAX_TIERS: usize = 4;

/// How a reduction aggregates its body values over the iterable.
#[derive(Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
    /// Number of steps whose body value is non-zero (so `count` with a body of
    /// `1` is the length).
    Count,
    /// `1` if the list has at least one item, else `0`. The body is ignored;
    /// summed over all lists this counts the used (non-empty) lists, e.g. the
    /// fleet size or the number of open bins.
    Used,
    /// The `k`-th smallest body value (0-indexed) over the iterable, i.e. an
    /// order statistic / quantile. Undefined (infeasible) if fewer than `k + 1`
    /// values are produced. Used for risk quantiles such as value-at-risk.
    SelectKth(usize),
}

/// Comparison for a constraint `reduction <op> rhs`.
#[derive(Clone, Copy)]
pub enum Op {
    Le,
    Ge,
    Eq,
}

/// Index of an [`Expr`] node inside a reduction's [`ExprArena`].
#[derive(Clone, Copy)]
pub struct ExprId(pub u32);

/// One node of a lambda body. Arrays and matrices are shared by `Arc`, so the
/// same constant table reused across many reductions costs one refcount bump,
/// not a deep copy.
#[derive(Clone)]
pub enum Expr {
    Const(i64),
    /// The `k`-th lambda binder: the item for `Items`, source/target for `Edges`.
    Arg(u8),
    /// `array[index]`.
    Array(Arc<Vec<i64>>, ExprId),
    /// `matrix[row][col]`.
    Matrix(Arc<Vec<Vec<i64>>>, ExprId, ExprId),
    Add(ExprId, ExprId),
    Sub(ExprId, ExprId),
    Mul(ExprId, ExprId),
    Min(ExprId, ExprId),
    Max(ExprId, ExprId),
    /// Integer division (`b == 0` yields 0, so it never panics).
    Div(ExprId, ExprId),
    /// Absolute value.
    Abs(ExprId),
    /// Comparisons returning `1` if true, else `0`.
    Lt(ExprId, ExprId),
    Le(ExprId, ExprId),
    Eq(ExprId, ExprId),
    Ne(ExprId, ExprId),
    /// `cond != 0 ? then : otherwise`.
    IfThenElse(ExprId, ExprId, ExprId),
}

/// A flat arena of [`Expr`] nodes; the body is the last/returned [`ExprId`].
#[derive(Clone, Default)]
pub struct ExprArena {
    pub exprs: Vec<Expr>,
}

impl ExprArena {
    fn push(&mut self, e: Expr) -> ExprId {
        let id = ExprId(self.exprs.len() as u32);
        self.exprs.push(e);
        id
    }
    pub fn constant(&mut self, v: i64) -> ExprId {
        self.push(Expr::Const(v))
    }
    pub fn arg(&mut self, k: u8) -> ExprId {
        self.push(Expr::Arg(k))
    }
    pub fn array(&mut self, a: Arc<Vec<i64>>, i: ExprId) -> ExprId {
        self.push(Expr::Array(a, i))
    }
    pub fn matrix(&mut self, m: Arc<Vec<Vec<i64>>>, i: ExprId, j: ExprId) -> ExprId {
        self.push(Expr::Matrix(m, i, j))
    }
    pub fn add(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Add(a, b))
    }
    pub fn sub(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Sub(a, b))
    }
    pub fn mul(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Mul(a, b))
    }
    pub fn min(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Min(a, b))
    }
    pub fn max(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Max(a, b))
    }
    pub fn div(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Div(a, b))
    }
    pub fn abs(&mut self, a: ExprId) -> ExprId {
        self.push(Expr::Abs(a))
    }
    pub fn lt(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Lt(a, b))
    }
    pub fn le(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Le(a, b))
    }
    pub fn eq(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Eq(a, b))
    }
    pub fn ne(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Ne(a, b))
    }
    pub fn if_then_else(&mut self, c: ExprId, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::IfThenElse(c, a, b))
    }
}

/// What a reduction iterates over. Every form names the list it reads.
#[derive(Clone)]
pub enum Iterable {
    /// Each item of the list once; the body sees the item value as `Arg(0)`.
    Items(usize),
    /// Each edge of `[start, items.., end]`; the body sees source as `Arg(0)`
    /// and target as `Arg(1)`. Used for closed-tour distance with a depot.
    Edges { list: usize, start: i32, end: i32 },
    /// Every ordered pair of positions `(i, j)` in the list. The body sees the
    /// item at `i` as `Arg(0)`, the item at `j` as `Arg(1)`, the position `i` as
    /// `Arg(2)` and `j` as `Arg(3)`. Used for quadratic objectives (QAP) and
    /// pairwise conflicts (bin packing with conflicts).
    Pairs(usize),
    /// A left fold along the list, threading an accumulator. At each item the
    /// `step` expression computes the new accumulator from `Arg(0)` (the current
    /// item), `Arg(1)` (the accumulator so far) and `Arg(2)` (the previous item,
    /// or `boundary` at the first step). The reduction's body then emits a
    /// per-step value seeing `Arg(0)` (current item), `Arg(1)` (the *new*
    /// accumulator) and `Arg(2)` (the previous item), aggregated by the
    /// reduction's op. Used for cumulative time and load along a route, e.g.
    /// time-window lateness. The accumulator must not index a table.
    Scan { list: usize, init: i64, boundary: i32, step: ExprId },
    /// Each window of `size` consecutive items. `inner` is summed over the
    /// window items (each seeing the item as `Arg(0)`) to a window total; the
    /// reduction's body then emits a per-window value seeing that total as
    /// `Arg(1)`, aggregated by the reduction's op. Used for sliding-window
    /// counts, e.g. car-sequencing option limits. The window total must not
    /// index a table.
    Windows { list: usize, size: usize, inner: ExprId },
}

impl Iterable {
    /// The list variable this reduction reads.
    pub fn list(&self) -> usize {
        match self {
            Iterable::Items(l) | Iterable::Pairs(l) | Iterable::Scan { list: l, .. } | Iterable::Windows { list: l, .. } => *l,
            Iterable::Edges { list, .. } => *list,
        }
    }
}

/// A reduction over one list: `op(iterable, args => body)`.
#[derive(Clone)]
pub struct Reduction {
    pub op: ReduceOp,
    pub iterable: Iterable,
    pub arena: ExprArena,
    pub body: ExprId,
    pub coeff: i64,
}

/// A constraint `reduction <op> rhs`.
#[derive(Clone)]
pub struct Constraint {
    pub reduction: Reduction,
    pub op: Op,
    pub rhs: i64,
}

/// A constraint relating which list two *items* land in, by item value. Unlike a
/// [`Reduction`], it is global (spans lists), so it is tracked separately from
/// the per-list scores. Used for assembly-line precedence and pickup/delivery
/// same-vehicle.
#[derive(Clone, Copy)]
pub enum GlobalConstraint {
    /// `list_of(before) <= list_of(after)`: `before` is assigned to a list whose
    /// index is no greater than `after`'s (a precedence over ordered stations).
    ListLe { before: i32, after: i32 },
    /// `list_of(a) == list_of(b)`: both items share a list (same vehicle/bin).
    SameList { a: i32, b: i32 },
}

impl GlobalConstraint {
    pub(crate) fn items(&self) -> [i32; 2] {
        match *self {
            GlobalConstraint::ListLe { before, after } => [before, after],
            GlobalConstraint::SameList { a, b } => [a, b],
        }
    }
}

/// A composable `coeff * max(sum(group_0), sum(group_1), ...)` objective
/// component.
#[derive(Clone)]
pub struct MaxTerm {
    pub groups: Vec<Vec<Reduction>>,
    pub coeff: i64,
}

/// One lexicographic objective tier. The tier value is the sum of its reductions
/// plus each `max_terms` component. Earlier tiers dominate later ones (e.g. fleet
/// size before distance). At most [`MAX_TIERS`] tiers.
#[derive(Clone)]
pub struct ObjectiveTier {
    pub minimize: bool,
    pub terms: Vec<Reduction>,
    pub max_terms: Option<Vec<MaxTerm>>,
}

impl ObjectiveTier {
    pub fn reductions(&self) -> impl Iterator<Item = &Reduction> {
        self.terms
            .iter()
            .chain(self.max_terms.iter().flat_map(|terms| terms.iter().flat_map(|term| term.groups.iter().flat_map(|group| group.iter()))))
    }

    pub fn has_max_terms(&self) -> bool {
        self.max_terms.as_ref().is_some_and(|terms| !terms.is_empty())
    }
}

/// One way to run an interval: on `machine` for `duration` time units. A
/// flexible op has several modes; the search picks one.
#[derive(Clone, Copy)]
pub struct Mode {
    pub machine: usize,
    pub duration: i64,
}

/// An interval decision variable: a task whose start is decided in
/// `[0, horizon - duration]`, so `end = start + duration`. With an empty
/// `modes` list the `duration` is fixed and the interval has no machine; with a
/// non-empty `modes` list the search also picks a mode, which sets the duration
/// and the machine (flexible job shop).
#[derive(Clone)]
pub struct IntervalVar {
    pub duration: i64,
    pub horizon: i64,
    pub modes: Vec<Mode>,
}

/// A renewable-resource limit over interval variables. `NoOverlap` is a unary
/// machine over the listed intervals; `MachineNoOverlap` groups moded intervals
/// by their chosen machine and keeps each machine's intervals from overlapping
/// (flexible job shop); `Cumulative` caps the total demand of overlapping
/// intervals at any instant.
#[derive(Clone)]
pub enum Resource {
    NoOverlap(Vec<usize>),
    MachineNoOverlap,
    Cumulative { demands: Vec<(usize, i64)>, capacity: i64 },
}

/// An interval-scheduling subproblem: interval variables plus the general,
/// reusable building blocks over them (precedence, resource limits, and the
/// makespan objective). Job-shop, flexible job-shop, and RCPSP all compose from
/// these, with no per-problem code.
#[derive(Clone)]
pub struct Schedule {
    pub intervals: Vec<IntervalVar>,
    /// `(a, b)`: interval `a` must end before interval `b` starts.
    pub precedences: Vec<(usize, usize)>,
    pub resources: Vec<Resource>,
    /// Minimise the makespan (the latest interval end). Always true in practice.
    pub minimize_makespan: bool,
}

/// A collection model: partition `items` among `lists` ordered lists, optimise
/// the objective tiers lexicographically, subject to the constraints. Each
/// reduction names the list variable it reads through its [`Iterable`].
/// Alternatively a model carries a [`Schedule`] of interval variables (a
/// disjoint solving mode).
#[derive(Clone)]
pub struct CollectionModel {
    pub items: Vec<i32>,
    pub lists: usize,
    pub objectives: Vec<ObjectiveTier>,
    pub constraints: Vec<Constraint>,
    /// Cross-list constraints relating which list items land in.
    pub globals: Vec<GlobalConstraint>,
    /// Interval-scheduling subproblem; when set, the engine solves it instead of
    /// the list model.
    pub schedule: Option<Schedule>,
}

/// A solution: the contents of each list, the objective value of each tier (in
/// the model's order), and feasibility.
#[derive(Clone, Debug)]
pub struct CollectionSolution {
    pub lists: Vec<Vec<i32>>,
    pub objectives: Vec<i64>,
    pub feasible: bool,
    /// Interval start times, for a [`Schedule`] model (empty otherwise).
    pub starts: Vec<i64>,
    /// Chosen machine of each interval (`-1` if it has no modes), for a moded
    /// [`Schedule`] model (empty otherwise).
    pub machines: Vec<i64>,
}
