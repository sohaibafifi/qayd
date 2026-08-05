//! List-reduction IR: a model partitions a universe of item ids among ordered
//! list variables, with objectives and constraints expressed as lambda
//! reductions over a list (an aggregate of a body evaluated at each item, or at
//! each edge of a route). One representation covers routing, sequencing,
//! assignment, and packing by changing only the reductions.

use std::sync::Arc;

/// Deterministic fixed-point value used for controlled real modeling. Every
/// value is represented exactly as `raw / scale`; the scale must be positive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedPoint {
    pub raw: i64,
    pub scale: i64,
}

impl FixedPoint {
    pub fn new(raw: i64, scale: i64) -> Result<Self, String> {
        if scale <= 0 {
            return Err("fixed-point scale must be positive".to_string());
        }
        Ok(Self { raw, scale })
    }

    pub fn from_f64(value: f64, scale: i64) -> Result<Self, String> {
        if !value.is_finite() || scale <= 0 {
            return Err("fixed-point value must be finite and its scale positive".to_string());
        }
        let scaled = value * scale as f64;
        if scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
            return Err("fixed-point value is outside the i64 range".to_string());
        }
        Ok(Self { raw: scaled.round() as i64, scale })
    }

    pub fn to_f64(self) -> f64 {
        self.raw as f64 / self.scale as f64
    }
}

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
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Le,
    Ge,
    Eq,
}

/// Index of an [`Expr`] node inside a reduction's [`ExprArena`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ExprId(pub u32);

/// One node of a lambda body. Arrays and matrices are shared by `Arc`, so the
/// same constant table reused across many reductions costs one refcount bump,
/// not a deep copy.
#[derive(Clone, PartialEq)]
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
    Mod(ExprId, ExprId),
    Pow(ExprId, u32),
    /// Fixed-point multiplication: `round(a * b / scale)`.
    MulScaled(ExprId, ExprId, i64),
    /// Fixed-point division: `round(a * scale / b)`.
    DivScaled(ExprId, ExprId, i64),
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
    /// Continuous piecewise-linear interpolation over strictly increasing
    /// `(x, y)` knots. Values outside the knot range are clamped to an endpoint.
    PiecewiseLinear {
        input: ExprId,
        points: Arc<Vec<(i64, i64)>>,
    },
    /// Call a process-registered deterministic external function.
    External {
        name: Arc<str>,
        args: Arc<Vec<ExprId>>,
    },
}

/// A flat arena of [`Expr`] nodes; the body is the last/returned [`ExprId`].
#[derive(Clone, Default, PartialEq)]
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
    pub fn modulo(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.push(Expr::Mod(a, b))
    }
    pub fn pow(&mut self, base: ExprId, exponent: u32) -> ExprId {
        self.push(Expr::Pow(base, exponent))
    }
    pub fn mul_scaled(&mut self, a: ExprId, b: ExprId, scale: i64) -> ExprId {
        self.push(Expr::MulScaled(a, b, scale))
    }
    pub fn div_scaled(&mut self, a: ExprId, b: ExprId, scale: i64) -> ExprId {
        self.push(Expr::DivScaled(a, b, scale))
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
    pub fn piecewise_linear(&mut self, input: ExprId, points: Arc<Vec<(i64, i64)>>) -> ExprId {
        self.push(Expr::PiecewiseLinear { input, points })
    }
    pub fn external(&mut self, name: impl Into<Arc<str>>, args: Vec<ExprId>) -> ExprId {
        self.push(Expr::External { name: name.into(), args: Arc::new(args) })
    }
}

/// What a reduction iterates over. Every form names the list it reads.
#[derive(Clone)]
pub enum Iterable {
    /// Each item of the list once; the body sees the item value as `Arg(0)`.
    Items(usize),
    /// Each member of an unordered set. Evaluation uses increasing item order,
    /// giving deterministic fixed-point and saturating arithmetic semantics.
    SetItems(usize),
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
    ///
    /// When `end` is `Some(node)`, the fold performs ONE more transition for the
    /// closing edge after the last item -- `step(end, acc_last, last_item)` then a
    /// final emit -- exactly as [`Iterable::Edges`] appends the closing edge
    /// `(last_item, end)`. This bounds a resource over the CLOSED tour (including
    /// the return arc). `None` stops at the last item (the original behaviour). On
    /// an empty list the closing edge is `(boundary, end)`, matching `Edges`.
    Scan { list: usize, init: i64, boundary: i32, step: ExprId, end: Option<i32> },
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
            Iterable::Items(l)
            | Iterable::SetItems(l)
            | Iterable::Pairs(l)
            | Iterable::Scan { list: l, .. }
            | Iterable::Windows { list: l, .. } => *l,
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
#[derive(Clone)]
pub enum GlobalConstraint {
    /// `list_of(before) <= list_of(after)`: `before` is assigned to a list whose
    /// index is no greater than `after`'s (a precedence over ordered stations).
    ListLe { before: i32, after: i32 },
    /// `list_of(a) == list_of(b)`: both items share a list (same vehicle/bin).
    SameList { a: i32, b: i32 },
    /// Two items must be assigned to different lists.
    DifferentList { a: i32, b: i32 },
    /// Every item in the group must share one list.
    AllSameList { items: Arc<Vec<i32>> },
    /// Every item in the group must use a distinct list.
    AllDifferentLists { items: Arc<Vec<i32>> },
    /// The absolute distance between the owner-list indices must be bounded.
    ListDistance { a: i32, b: i32, min: usize, max: usize },
}

impl GlobalConstraint {
    pub(crate) fn items(&self) -> Vec<i32> {
        match self {
            GlobalConstraint::ListLe { before, after } => vec![*before, *after],
            GlobalConstraint::SameList { a, b }
            | GlobalConstraint::DifferentList { a, b }
            | GlobalConstraint::ListDistance { a, b, .. } => vec![*a, *b],
            GlobalConstraint::AllSameList { items } | GlobalConstraint::AllDifferentLists { items } => items.as_ref().clone(),
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
/// size before distance). The number of tiers is dynamic and unbounded by the
/// IR; practical limits are memory and evaluation time only.
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
    /// Whether a fixed-duration interval may be absent. Moded operations remain
    /// mandatory because their modes already use optional member intervals.
    pub optional: bool,
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

/// Certified bound for the primary objective of a feasible collection
/// solution. For minimization, `dual` is a lower bound; for maximization it is
/// an upper bound. `relative_gap` is a ratio, so `0.01` means one percent.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundReport {
    pub dual: i64,
    pub absolute_gap: u64,
    pub relative_gap: f64,
    pub method: String,
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
    /// Interval presence flags for a schedule model (empty for list models).
    pub presences: Vec<bool>,
    /// Chosen machine of each interval (`-1` if it has no modes), for a moded
    /// [`Schedule`] model (empty otherwise).
    pub machines: Vec<i64>,
    /// Certified dual bound and gap for the primary objective, when a supported
    /// relaxation was available.
    pub bound: Option<BoundReport>,
}
