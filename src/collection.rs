//! General collection modeling and local search.
//!
//! A model partitions a universe of item ids among `lists` ordered list
//! variables. Objective and constraints are *reductions over a list*: an
//! aggregate (`sum` / `min` / `max` / `count`) of a lambda body evaluated at
//! each item, or at each edge of the closed tour. The lambda body is a small
//! expression IR with numbered binders (`Arg(0)` is the item or edge source,
//! `Arg(1)` the edge target), so `sum(route, i => demand[i])` and
//! `sum_edges(route, (i, j) => dist[i][j])` are just two reductions over the
//! same machinery. One engine then solves routing, sequencing, assignment, and
//! packing models, since each is a different set of reductions over the lists.
//!
//! Search is constraint-based local search: the score is the pair
//! `(violation, objective)` compared lexicographically, so feasibility is driven
//! to zero first. Moves (relocate, swap, segment reversal) keep the partition by
//! construction. The *sequence* of moves and kicks is a pure function of the
//! seed, but the search runs until the shared `stop` flag is set, so the point
//! at which it stops (and thus the best solution returned) depends on the wall
//! clock budget. Same seed plus same number of iterations gives the same answer;
//! a time budget makes the result best-effort, not bit-reproducible across
//! machines.
//!
//! Empty semantics follow Hexaly: `sum`/`count` of nothing is `0`; `min`/`max`
//! of an empty item set is *undefined* and makes the state infeasible. (`min`
//! and `max` over `Edges` are always defined, since a closed tour has at least
//! the single `start -> end` edge.)
//!
//! Arithmetic in bodies and scores saturates rather than wrapping or panicking,
//! so a pathological table value cannot crash the solver path.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::mix64;

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
    fn items(&self) -> [i32; 2] {
        match *self {
            GlobalConstraint::ListLe { before, after } => [before, after],
            GlobalConstraint::SameList { a, b } => [a, b],
        }
    }
}

/// One lexicographic objective tier: the sum of its reductions, minimised or
/// maximised. Earlier tiers dominate later ones (e.g. fleet size before
/// distance). At most [`MAX_TIERS`] tiers.
#[derive(Clone)]
pub struct ObjectiveTier {
    pub minimize: bool,
    pub terms: Vec<Reduction>,
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

/// Violation weight for an undefined reduction (empty `min`/`max`) or a broken
/// constraint operand, large enough to dominate any real objective.
const INFEASIBLE: i64 = 1_000_000_000;

/// Evaluate one expression node. Assumes the model was validated, so an
/// out-of-range index is unreachable; the `unwrap_or(0)` is a defensive
/// fallback rather than a silent default for user error.
fn eval_expr(arena: &[Expr], id: ExprId, args: &[i64]) -> i64 {
    match &arena[id.0 as usize] {
        Expr::Const(c) => *c,
        Expr::Arg(k) => args.get(*k as usize).copied().unwrap_or(0),
        Expr::Array(a, i) => {
            let idx = eval_expr(arena, *i, args);
            usize::try_from(idx).ok().and_then(|u| a.get(u)).copied().unwrap_or(0)
        }
        Expr::Matrix(m, i, j) => {
            let ri = eval_expr(arena, *i, args);
            let ci = eval_expr(arena, *j, args);
            usize::try_from(ri)
                .ok()
                .and_then(|r| m.get(r))
                .and_then(|row| usize::try_from(ci).ok().and_then(|c| row.get(c)))
                .copied()
                .unwrap_or(0)
        }
        Expr::Add(a, b) => eval_expr(arena, *a, args).saturating_add(eval_expr(arena, *b, args)),
        Expr::Sub(a, b) => eval_expr(arena, *a, args).saturating_sub(eval_expr(arena, *b, args)),
        Expr::Mul(a, b) => eval_expr(arena, *a, args).saturating_mul(eval_expr(arena, *b, args)),
        Expr::Min(a, b) => eval_expr(arena, *a, args).min(eval_expr(arena, *b, args)),
        Expr::Max(a, b) => eval_expr(arena, *a, args).max(eval_expr(arena, *b, args)),
        Expr::Div(a, b) => {
            let d = eval_expr(arena, *b, args);
            if d == 0 { 0 } else { eval_expr(arena, *a, args).checked_div(d).unwrap_or(0) }
        }
        Expr::Abs(a) => eval_expr(arena, *a, args).saturating_abs(),
        Expr::Lt(a, b) => i64::from(eval_expr(arena, *a, args) < eval_expr(arena, *b, args)),
        Expr::Le(a, b) => i64::from(eval_expr(arena, *a, args) <= eval_expr(arena, *b, args)),
        Expr::Eq(a, b) => i64::from(eval_expr(arena, *a, args) == eval_expr(arena, *b, args)),
        Expr::Ne(a, b) => i64::from(eval_expr(arena, *a, args) != eval_expr(arena, *b, args)),
        Expr::IfThenElse(c, a, b) => {
            if eval_expr(arena, *c, args) != 0 {
                eval_expr(arena, *a, args)
            } else {
                eval_expr(arena, *b, args)
            }
        }
    }
}

/// Evaluate a reduction over a list's contents. `None` means undefined (an
/// empty `min`/`max` over items), which the scorer treats as infeasible.
fn eval_reduction(reduction: &Reduction, contents: &[i32]) -> Option<i64> {
    let arena = &reduction.arena.exprs;
    let body = reduction.body;
    let mut sum = 0i64;
    let mut count = 0i64;
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    let mut steps = 0u64;
    // Only an order-statistic op needs to retain every value; the others fold
    // incrementally with no allocation.
    let want_values = matches!(reduction.op, ReduceOp::SelectKth(_));
    let mut values: Vec<i64> = Vec::new();
    let mut fold = |v: i64| {
        sum = sum.saturating_add(v);
        if v != 0 {
            count += 1;
        }
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
        steps += 1;
        if want_values {
            values.push(v);
        }
    };

    match reduction.iterable {
        Iterable::Items(_) => {
            for &item in contents {
                fold(eval_expr(arena, body, &[i64::from(item)]));
            }
        }
        Iterable::Edges { start, end, .. } => {
            // The closed tour [start, items.., end] always has at least one edge
            // (start -> end when the list is empty), so edge min/max is defined.
            let n = contents.len();
            for p in 0..=n {
                let from = if p == 0 { start } else { contents[p - 1] };
                let to = if p == n { end } else { contents[p] };
                fold(eval_expr(arena, body, &[i64::from(from), i64::from(to)]));
            }
        }
        Iterable::Pairs(_) => {
            let n = contents.len();
            for i in 0..n {
                for j in 0..n {
                    let args = [i64::from(contents[i]), i64::from(contents[j]), i as i64, j as i64];
                    fold(eval_expr(arena, body, &args));
                }
            }
        }
        Iterable::Scan { init, boundary, step, .. } => {
            let mut acc = init;
            let mut prev = boundary;
            for &cur in contents {
                let new_acc = eval_expr(arena, step, &[i64::from(cur), acc, i64::from(prev)]);
                fold(eval_expr(arena, body, &[i64::from(cur), new_acc, i64::from(prev)]));
                acc = new_acc;
                prev = cur;
            }
        }
        Iterable::Windows { size, inner, .. } => {
            let n = contents.len();
            if size > 0 && n >= size {
                for start in 0..=(n - size) {
                    let mut total = 0i64;
                    for &cur in &contents[start..start + size] {
                        total = total.saturating_add(eval_expr(arena, inner, &[i64::from(cur)]));
                    }
                    fold(eval_expr(arena, body, &[0, total]));
                }
            }
        }
    }

    match reduction.op {
        ReduceOp::Sum => Some(sum),
        ReduceOp::Count => Some(count),
        ReduceOp::Used => Some(i64::from(steps > 0)),
        ReduceOp::Min => (steps > 0).then_some(lo),
        ReduceOp::Max => (steps > 0).then_some(hi),
        ReduceOp::SelectKth(k) => {
            if k >= values.len() {
                None // not enough values for the k-th order statistic
            } else {
                values.sort_unstable();
                Some(values[k])
            }
        }
    }
}

fn violation_of(value: i64, op: Op, rhs: i64) -> i64 {
    match op {
        Op::Le => value.saturating_sub(rhs).max(0),
        Op::Ge => rhs.saturating_sub(value).max(0),
        Op::Eq => value.saturating_sub(rhs).saturating_abs(),
    }
}

/// Like [`eval_expr`] but reports an out-of-range index or unbound argument
/// instead of falling back to zero. Used only by [`CollectionModel::validate`].
fn eval_expr_checked(arena: &[Expr], id: ExprId, args: &[i64]) -> Result<i64, String> {
    let node = arena.get(id.0 as usize).ok_or_else(|| format!("expr id {} out of arena range", id.0))?;
    Ok(match node {
        Expr::Const(c) => *c,
        Expr::Arg(k) => *args.get(*k as usize).ok_or_else(|| format!("lambda arg {k} is not bound by this iterable"))?,
        Expr::Array(a, i) => {
            let idx = eval_expr_checked(arena, *i, args)?;
            if idx < 0 || idx as usize >= a.len() {
                return Err(format!("array index {idx} out of range 0..{}", a.len()));
            }
            a[idx as usize]
        }
        Expr::Matrix(m, i, j) => {
            let ri = eval_expr_checked(arena, *i, args)?;
            let ci = eval_expr_checked(arena, *j, args)?;
            if ri < 0 || ri as usize >= m.len() {
                return Err(format!("matrix row index {ri} out of range 0..{}", m.len()));
            }
            let row = &m[ri as usize];
            if ci < 0 || ci as usize >= row.len() {
                return Err(format!("matrix column index {ci} out of range 0..{}", row.len()));
            }
            row[ci as usize]
        }
        Expr::Add(a, b) => eval_expr_checked(arena, *a, args)?
            .checked_add(eval_expr_checked(arena, *b, args)?)
            .ok_or_else(|| "integer overflow in a lambda body (+)".to_string())?,
        Expr::Sub(a, b) => eval_expr_checked(arena, *a, args)?
            .checked_sub(eval_expr_checked(arena, *b, args)?)
            .ok_or_else(|| "integer overflow in a lambda body (-)".to_string())?,
        Expr::Mul(a, b) => eval_expr_checked(arena, *a, args)?
            .checked_mul(eval_expr_checked(arena, *b, args)?)
            .ok_or_else(|| "integer overflow in a lambda body (*)".to_string())?,
        Expr::Min(a, b) => eval_expr_checked(arena, *a, args)?.min(eval_expr_checked(arena, *b, args)?),
        Expr::Max(a, b) => eval_expr_checked(arena, *a, args)?.max(eval_expr_checked(arena, *b, args)?),
        Expr::Div(a, b) => {
            let num = eval_expr_checked(arena, *a, args)?;
            let den = eval_expr_checked(arena, *b, args)?;
            if den == 0 { 0 } else { num.checked_div(den).ok_or_else(|| "integer overflow in a lambda body (/)".to_string())? }
        }
        Expr::Abs(a) => eval_expr_checked(arena, *a, args)?.saturating_abs(),
        Expr::Lt(a, b) => i64::from(eval_expr_checked(arena, *a, args)? < eval_expr_checked(arena, *b, args)?),
        Expr::Le(a, b) => i64::from(eval_expr_checked(arena, *a, args)? <= eval_expr_checked(arena, *b, args)?),
        Expr::Eq(a, b) => i64::from(eval_expr_checked(arena, *a, args)? == eval_expr_checked(arena, *b, args)?),
        Expr::Ne(a, b) => i64::from(eval_expr_checked(arena, *a, args)? != eval_expr_checked(arena, *b, args)?),
        Expr::IfThenElse(c, a, b) => {
            // Evaluate both branches so an out-of-range index in either is
            // reported regardless of which the condition would pick.
            let cond = eval_expr_checked(arena, *c, args)?;
            let then = eval_expr_checked(arena, *a, args)?;
            let other = eval_expr_checked(arena, *b, args)?;
            if cond != 0 { then } else { other }
        }
    })
}

/// Bitmask of which `Arg(k)` binders (k < 8) appear in a subtree.
fn args_used(arena: &[Expr], id: ExprId) -> u8 {
    match arena.get(id.0 as usize) {
        None => 0,
        Some(Expr::Const(_)) => 0,
        Some(Expr::Arg(k)) => 1u8.checked_shl(u32::from(*k)).unwrap_or(0),
        Some(Expr::Array(_, i)) => args_used(arena, *i),
        Some(Expr::Matrix(_, i, j)) => args_used(arena, *i) | args_used(arena, *j),
        Some(Expr::Abs(a)) => args_used(arena, *a),
        Some(
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Min(a, b)
            | Expr::Max(a, b)
            | Expr::Div(a, b)
            | Expr::Lt(a, b)
            | Expr::Le(a, b)
            | Expr::Eq(a, b)
            | Expr::Ne(a, b),
        ) => args_used(arena, *a) | args_used(arena, *b),
        Some(Expr::IfThenElse(c, a, b)) => args_used(arena, *c) | args_used(arena, *a) | args_used(arena, *b),
    }
}

/// Call `f` for every assignment of the `varying` arg slots over their domains
/// (other slots stay 0). Bounded recursion: arity is 1 or 2.
fn for_each_combo(domains: &[&[i64]], varying: &[usize], args: &mut [i64], depth: usize, f: &mut dyn FnMut(&[i64]) -> Result<(), String>) -> Result<(), String> {
    if depth == varying.len() {
        return f(args);
    }
    let slot = varying[depth];
    for &val in domains[slot] {
        args[slot] = val;
        for_each_combo(domains, varying, args, depth + 1, f)?;
    }
    Ok(())
}

/// Check that an index expression stays in `0..hi` for every value of the args
/// it uses. Only the used arg slots are enumerated, so a separable index like
/// `Arg(0)` is linear in the node count rather than quadratic.
fn check_index(arena: &[Expr], idx: ExprId, domains: &[&[i64]], hi: usize, what: &str) -> Result<(), String> {
    let mask = args_used(arena, idx);
    let varying: Vec<usize> = (0..domains.len()).filter(|&s| mask & (1u8 << s) != 0).collect();
    let mut args = vec![0i64; domains.len()];
    for_each_combo(domains, &varying, &mut args, 0, &mut |a| {
        let v = eval_expr_checked(arena, idx, a)?;
        if v < 0 || v as usize >= hi {
            return Err(format!("{what} index {v} out of range 0..{hi}"));
        }
        Ok(())
    })
}

/// Reject indexing any array/matrix by `Arg(1)`, used where that binder is an
/// unbounded computed value (a scan accumulator or a window total) with no
/// finite domain to validate against.
fn reject_arg1_index(arena: &[Expr], what: &str) -> Result<(), String> {
    for node in arena {
        let uses_arg1 = match node {
            Expr::Array(_, i) => args_used(arena, *i) & 0b10 != 0,
            Expr::Matrix(_, i, j) => (args_used(arena, *i) | args_used(arena, *j)) & 0b10 != 0,
            _ => false,
        };
        if uses_arg1 {
            return Err(format!("the {what} (Arg(1)) cannot index an array or matrix"));
        }
    }
    Ok(())
}

/// Validate a reduction: it reads a real list, the body id is in range, and
/// every array/matrix access stays in bounds for any value its index can take.
/// Checks each access against only the args its index uses, so the common
/// `dist[i][j]` / `demand[i]` shapes validate in linear time, not quadratic.
fn validate_reduction(reduction: &Reduction, items: &[i32], lists: usize) -> Result<(), String> {
    if reduction.iterable.list() >= lists {
        return Err(format!("reduction reads list {} but only {} exist", reduction.iterable.list(), lists));
    }
    let arena = &reduction.arena.exprs;
    if reduction.body.0 as usize >= arena.len() {
        return Err("reduction body refers to a missing expression".to_string());
    }

    // Node values each binder can take: items for `Arg(0)` under `Items`; items
    // plus the two boundary nodes for `Arg(0)`/`Arg(1)` under `Edges`; for
    // `Pairs`, the two item binders range over items and the two position
    // binders over `0..items.len()` (a list can hold at most every item).
    let items_i: Vec<i64> = items.iter().map(|&x| i64::from(x)).collect();
    let nodes_i: Vec<i64>;
    let positions_i: Vec<i64>;
    let scan_nodes: Vec<i64>;
    let acc_dummy = [0i64];
    let domains: Vec<&[i64]> = match reduction.iterable {
        Iterable::Items(_) => vec![&items_i],
        Iterable::Edges { start, end, .. } => {
            let mut n = items_i.clone();
            n.push(i64::from(start));
            n.push(i64::from(end));
            nodes_i = n;
            vec![&nodes_i, &nodes_i]
        }
        Iterable::Pairs(_) => {
            positions_i = (0..items_i.len() as i64).collect();
            vec![&items_i, &items_i, &positions_i, &positions_i]
        }
        Iterable::Scan { boundary, step, .. } => {
            if step.0 as usize >= arena.len() {
                return Err("scan step refers to a missing expression".to_string());
            }
            // The accumulator (`Arg(1)`) has no finite domain, so it must never
            // index a table; reject that rather than read a silent 0.
            reject_arg1_index(arena, "scan accumulator")?;
            let mut n = items_i.clone();
            n.push(i64::from(boundary));
            scan_nodes = n;
            // Arg(0)=current item and Arg(2)=previous item range over nodes; the
            // accumulator slot is a dummy since it is never an index (checked).
            vec![&items_i, &acc_dummy, &scan_nodes]
        }
        Iterable::Windows { size, inner, .. } => {
            if size == 0 {
                return Err("window size must be positive".to_string());
            }
            if inner.0 as usize >= arena.len() {
                return Err("window inner expression is missing".to_string());
            }
            // The window total (`Arg(1)`) is unbounded; it must not index a table.
            reject_arg1_index(arena, "window total")?;
            // Arg(0)=window item ranges over items; the total slot is a dummy.
            vec![&items_i, &acc_dummy]
        }
    };

    for node in arena {
        match node {
            Expr::Array(arr, idx) => check_index(arena, *idx, &domains, arr.len(), "array")?,
            Expr::Matrix(m, ri, ci) => {
                check_index(arena, *ri, &domains, m.len(), "matrix row")?;
                let rectangular = m.iter().map(Vec::len).min() == m.iter().map(Vec::len).max();
                if rectangular {
                    let cols = m.first().map_or(0, Vec::len);
                    check_index(arena, *ci, &domains, cols, "matrix column")?;
                } else {
                    // Ragged rows: the column bound depends on the row, so the
                    // two indices must be checked jointly.
                    let mask = args_used(arena, *ri) | args_used(arena, *ci);
                    let varying: Vec<usize> = (0..domains.len()).filter(|&s| mask & (1u8 << s) != 0).collect();
                    let mut args = vec![0i64; domains.len()];
                    for_each_combo(&domains, &varying, &mut args, 0, &mut |a| {
                        let r = eval_expr_checked(arena, *ri, a)?;
                        if r < 0 || r as usize >= m.len() {
                            return Err(format!("matrix row index {r} out of range 0..{}", m.len()));
                        }
                        let c = eval_expr_checked(arena, *ci, a)?;
                        let row_len = m[r as usize].len();
                        if c < 0 || c as usize >= row_len {
                            return Err(format!("matrix column index {c} out of range 0..{row_len}"));
                        }
                        Ok(())
                    })?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

impl CollectionModel {
    /// Check the model is well-formed before solving: at least one list, every
    /// reduction reads a real list, and every body only indexes its arrays and
    /// matrices within bounds for any placement of items. Returns a
    /// human-readable error otherwise, so the engine never reads a silent zero.
    pub fn validate(&self) -> Result<(), String> {
        // A schedule model is a disjoint mode: validate the intervals only.
        if let Some(sched) = &self.schedule {
            let n = sched.intervals.len();
            if n == 0 {
                return Err("a schedule needs at least one interval".to_string());
            }
            for iv in &sched.intervals {
                if iv.modes.is_empty() {
                    if iv.duration < 0 || iv.duration > iv.horizon {
                        return Err("interval duration must be in 0..=horizon".to_string());
                    }
                } else {
                    for mode in &iv.modes {
                        if mode.duration < 0 || mode.duration > iv.horizon {
                            return Err("interval mode duration must be in 0..=horizon".to_string());
                        }
                    }
                }
            }
            let in_range = |i: usize| i < n;
            for &(a, b) in &sched.precedences {
                if !in_range(a) || !in_range(b) {
                    return Err("precedence references a missing interval".to_string());
                }
            }
            for res in &sched.resources {
                match res {
                    Resource::NoOverlap(ivs) => {
                        if ivs.iter().any(|&i| !in_range(i)) {
                            return Err("no_overlap references a missing interval".to_string());
                        }
                    }
                    Resource::MachineNoOverlap => {}
                    Resource::Cumulative { demands, capacity } => {
                        if *capacity < 0 {
                            return Err("resource capacity must be non-negative".to_string());
                        }
                        if demands.iter().any(|&(i, d)| !in_range(i) || d < 0) {
                            return Err("cumulative demand references a missing interval or is negative".to_string());
                        }
                    }
                }
            }
            return Ok(());
        }
        if self.lists == 0 {
            return Err("a collection model needs at least one list".to_string());
        }
        if self.items.is_empty() {
            // Without items, an Items body is never evaluated and so never index
            // checked; require a non-empty universe so validation is total.
            return Err("a collection model needs at least one item".to_string());
        }
        // Items must be distinct: global-constraint tracking maps each item value
        // to a single index, so a duplicate value would corrupt that mapping.
        let mut seen = std::collections::HashSet::with_capacity(self.items.len());
        if let Some(dup) = self.items.iter().find(|v| !seen.insert(**v)) {
            return Err(format!("duplicate item {dup}; items must be distinct"));
        }
        if self.objectives.len() > MAX_TIERS {
            return Err(format!("at most {MAX_TIERS} objective tiers are supported"));
        }
        for tier in &self.objectives {
            for r in &tier.terms {
                validate_reduction(r, &self.items, self.lists)?;
            }
        }
        for c in &self.constraints {
            validate_reduction(&c.reduction, &self.items, self.lists)?;
        }
        for g in &self.globals {
            for v in g.items() {
                if !self.items.contains(&v) {
                    return Err(format!("global constraint references item {v}, which is not in the universe"));
                }
            }
        }
        Ok(())
    }
}

/// Precomputed lookup for cross-list global constraints: item value to dense
/// index, and which globals reference each item.
struct Globals {
    cons: Vec<GlobalConstraint>,
    value_to_idx: HashMap<i32, usize>,
    of_idx: Vec<Vec<usize>>,
}

impl Globals {
    fn build(model: &CollectionModel) -> Self {
        let value_to_idx: HashMap<i32, usize> = model.items.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let mut of_idx = vec![Vec::new(); model.items.len()];
        for (g, c) in model.globals.iter().enumerate() {
            for v in c.items() {
                if let Some(&i) = value_to_idx.get(&v) {
                    if !of_idx[i].contains(&g) {
                        of_idx[i].push(g);
                    }
                }
            }
        }
        Self { cons: model.globals.clone(), value_to_idx, of_idx }
    }

    /// List index of `value`, honouring any `(value, list)` overrides.
    fn list_of(&self, item_list: &[usize], value: i32, ov: &[(i32, usize)]) -> usize {
        for &(v, l) in ov {
            if v == value {
                return l;
            }
        }
        self.value_to_idx.get(&value).map_or(0, |&i| item_list[i])
    }

    /// Violation of one global constraint under the current `item_list` plus
    /// overrides. `ListLe` penalises by how many lists out of order; `SameList`
    /// by the list-index distance, so both are smooth for local search.
    fn one(&self, c: &GlobalConstraint, item_list: &[usize], ov: &[(i32, usize)]) -> i64 {
        match c {
            GlobalConstraint::ListLe { before, after } => {
                let lb = self.list_of(item_list, *before, ov) as i64;
                let la = self.list_of(item_list, *after, ov) as i64;
                (lb - la).max(0)
            }
            GlobalConstraint::SameList { a, b } => {
                let la = self.list_of(item_list, *a, ov) as i64;
                let lb = self.list_of(item_list, *b, ov) as i64;
                (la - lb).abs()
            }
        }
    }

    /// Total violation over all global constraints.
    fn total(&self, item_list: &[usize]) -> i64 {
        self.cons.iter().fold(0i64, |acc, c| acc.saturating_add(self.one(c, item_list, &[])))
    }

    /// Change in total global violation if the listed items moved to new lists.
    fn delta(&self, item_list: &[usize], overrides: &[(i32, usize)]) -> i64 {
        let mut affected: Vec<usize> = Vec::new();
        for &(v, _) in overrides {
            if let Some(&i) = self.value_to_idx.get(&v) {
                for &g in &self.of_idx[i] {
                    if !affected.contains(&g) {
                        affected.push(g);
                    }
                }
            }
        }
        let mut d = 0i64;
        for &g in &affected {
            let c = &self.cons[g];
            d = d.saturating_add(self.one(c, item_list, overrides).saturating_sub(self.one(c, item_list, &[])));
        }
        d
    }
}

/// Objective reductions (grouped by tier) and constraints, both grouped by the
/// list they read so a move only rescores the lists it touches. `senses[t]` is
/// true when tier `t` is minimised.
struct PerList {
    objective: Vec<[Vec<Reduction>; MAX_TIERS]>,
    constraints: Vec<Vec<Constraint>>,
    senses: [bool; MAX_TIERS],
    tiers: usize,
    globals: Globals,
    /// Whether every reduction on list `l` supports O(edit) incremental scoring,
    /// so a candidate move can be evaluated without rebuilding/rescanning the
    /// list. Lists with an unsupported reduction (Pairs, Scan, Windows, Min, Max)
    /// fall back to full recomputation.
    list_incremental: Vec<bool>,
}

/// Whether a reduction can be scored incrementally from the old list plus a local
/// edit, in time proportional to the edit rather than the list length.
fn reduction_supported(r: &Reduction) -> bool {
    matches!(
        (r.op, &r.iterable),
        (ReduceOp::Sum, Iterable::Items(_)) | (ReduceOp::Count, Iterable::Items(_)) | (ReduceOp::Used, Iterable::Items(_)) | (ReduceOp::Sum, Iterable::Edges { .. })
    )
}

impl PerList {
    fn build(model: &CollectionModel) -> Self {
        let mut objective: Vec<[Vec<Reduction>; MAX_TIERS]> = (0..model.lists).map(|_| std::array::from_fn(|_| Vec::new())).collect();
        let mut constraints = vec![Vec::new(); model.lists];
        let mut senses = [true; MAX_TIERS];
        for (t, tier) in model.objectives.iter().enumerate() {
            senses[t] = tier.minimize;
            for r in &tier.terms {
                objective[r.iterable.list()][t].push(r.clone());
            }
        }
        for c in &model.constraints {
            constraints[c.reduction.iterable.list()].push(c.clone());
        }
        let list_incremental = (0..model.lists)
            .map(|l| {
                objective[l].iter().all(|tier| tier.iter().all(reduction_supported)) && constraints[l].iter().all(|c| reduction_supported(&c.reduction))
            })
            .collect();
        Self { objective, constraints, senses, tiers: model.objectives.len(), globals: Globals::build(model), list_incremental }
    }
}

/// The comparable score of a state: violation first, then the objective tiers
/// (each already signed so smaller is better), compared lexicographically.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Score {
    violation: i64,
    tiers: [i64; MAX_TIERS],
}

/// Cached per-list contribution: its violation and its raw (unsigned) value in
/// each objective tier.
#[derive(Clone, Copy)]
struct ListScore {
    violation: i64,
    objectives: [i64; MAX_TIERS],
}

fn list_score(per: &PerList, idx: usize, contents: &[i32]) -> ListScore {
    let mut violation = 0i64;
    let mut objectives = [0i64; MAX_TIERS];
    for (t, tier) in per.objective[idx].iter().enumerate() {
        for r in tier {
            match eval_reduction(r, contents) {
                Some(v) => objectives[t] = objectives[t].saturating_add(v),
                None => violation = violation.saturating_add(INFEASIBLE),
            }
        }
    }
    for c in &per.constraints[idx] {
        match eval_reduction(&c.reduction, contents) {
            Some(v) => violation = violation.saturating_add(violation_of(v, c.op, c.rhs)),
            None => violation = violation.saturating_add(INFEASIBLE),
        }
    }
    ListScore { violation, objectives }
}

/// Apply each tier's optimisation direction to raw tier sums (smaller better).
fn signed(per: &PerList, violation: i64, raw: [i64; MAX_TIERS]) -> Score {
    let mut tiers = [0i64; MAX_TIERS];
    for ((slot, &r), &minimize) in tiers.iter_mut().zip(raw.iter()).zip(per.senses.iter()) {
        *slot = if minimize { r } else { r.saturating_neg() };
    }
    Score { violation, tiers }
}

/// Raw (unsigned) totals across all lists: violation and per-tier sums.
fn base_totals(scores: &[ListScore]) -> (i64, [i64; MAX_TIERS]) {
    let mut violation = 0i64;
    let mut raw = [0i64; MAX_TIERS];
    for s in scores {
        violation = violation.saturating_add(s.violation);
        for (r, &o) in raw.iter_mut().zip(s.objectives.iter()) {
            *r = r.saturating_add(o);
        }
    }
    (violation, raw)
}

fn total_score(per: &PerList, scores: &[ListScore]) -> Score {
    let (violation, raw) = base_totals(scores);
    signed(per, violation, raw)
}

/// Full score including the cross-list global violation.
fn full_score(per: &PerList, state: &State) -> Score {
    let mut s = total_score(per, &state.scores);
    s.violation = s.violation.saturating_add(state.global_viol);
    s
}

/// State of the search: list contents, cached per-list scores, and (for global
/// constraints) each item's current list plus the total global violation.
struct State {
    lists: Vec<Vec<i32>>,
    scores: Vec<ListScore>,
    /// Raw value of each constraint reduction per list, so a candidate move can
    /// update the (nonlinear) constraint violation incrementally.
    con_vals: Vec<Vec<Option<i64>>>,
    item_list: Vec<usize>,
    global_viol: i64,
}

/// Raw value of every constraint reduction on a list (`None` = undefined).
fn compute_con_vals(per: &PerList, idx: usize, contents: &[i32]) -> Vec<Option<i64>> {
    per.constraints[idx].iter().map(|c| eval_reduction(&c.reduction, contents)).collect()
}

impl State {
    /// Greedy construction: insert each item (in `order`) into the list whose
    /// resulting per-list score increases least, lexicographically (violation
    /// first, then objective). This yields a feasibility-leaning start — items
    /// avoid lists they would push over a capacity bound, and empty lists with a
    /// `min`/`max` constraint get filled — far better than a blind round robin
    /// on tight instances, while different `order`s give diverse restarts.
    /// Global constraints are ignored here; the search repairs them.
    fn greedy(model: &CollectionModel, per: &PerList, order: &[usize]) -> Self {
        let k = model.lists.max(1);
        let mut lists: Vec<Vec<i32>> = vec![Vec::new(); k];
        let mut scores: Vec<ListScore> = (0..k).map(|idx| list_score(per, idx, &[])).collect();
        let mut item_list = vec![0usize; model.items.len()];
        let mut buf: Vec<i32> = Vec::new();
        for &item_idx in order {
            let item = model.items[item_idx];
            let mut best_l = 0;
            let mut best_key = Score { violation: i64::MAX, tiers: [i64::MAX; MAX_TIERS] };
            let mut best_sc = scores[0];
            for l in 0..k {
                buf.clear();
                buf.extend_from_slice(&lists[l]);
                buf.push(item);
                let sc = list_score(per, l, &buf);
                // Lexicographic increment of placing the item in list l.
                let dv = sc.violation.saturating_sub(scores[l].violation);
                let mut draw = [0i64; MAX_TIERS];
                for ((d, &new), &old) in draw.iter_mut().zip(sc.objectives.iter()).zip(scores[l].objectives.iter()) {
                    *d = new.saturating_sub(old);
                }
                let key = signed(per, dv, draw);
                if key < best_key {
                    best_key = key;
                    best_l = l;
                    best_sc = sc;
                }
            }
            lists[best_l].push(item);
            scores[best_l] = best_sc;
            item_list[item_idx] = best_l;
        }
        let global_viol = per.globals.total(&item_list);
        let con_vals = (0..k).map(|l| compute_con_vals(per, l, &lists[l])).collect();
        Self { lists, scores, con_vals, item_list, global_viol }
    }

    fn rescore(&mut self, per: &PerList, idx: usize) {
        self.scores[idx] = list_score(per, idx, &self.lists[idx]);
        self.con_vals[idx] = compute_con_vals(per, idx, &self.lists[idx]);
    }

    /// Record that `value` now lives in list `l`, for global-constraint tracking.
    fn set_item_list(&mut self, per: &PerList, value: i32, l: usize) {
        if let Some(&i) = per.globals.value_to_idx.get(&value) {
            self.item_list[i] = l;
        }
    }
}

/// Raw (unsigned) value of a tier, undoing the maximisation sign flip.
fn tier_value(per: &PerList, score: &Score, tier: usize) -> i64 {
    if per.senses[tier] { score.tiers[tier] } else { score.tiers[tier].saturating_neg() }
}

/// The raw objective values of every declared tier, in model order.
fn objective_values(per: &PerList, score: &Score) -> Vec<i64> {
    (0..per.tiers).map(|t| tier_value(per, score, t)).collect()
}

/// Solve a collection model with constraint-based local search until `stop`.
/// `report` is called with the objective each time a strictly better *feasible*
/// incumbent is found, for progress output; pass `&mut |_| {}` to ignore it.
pub fn solve_collection(model: &CollectionModel, seed: u64, stop: &AtomicBool, report: &mut dyn FnMut(i64)) -> CollectionSolution {
    // Guard the search path: an invalid model would otherwise panic (bad list
    // index) or read silent zeros (out-of-range table index). Callers like the
    // Python frontend validate first to raise a precise error; this is the
    // backstop for direct Rust callers so the engine never panics or corrupts.
    if let Err(_e) = model.validate() {
        debug_assert!(false, "solve_collection called on an invalid model: {_e}");
        return CollectionSolution { lists: vec![Vec::new(); model.lists.max(1)], objectives: Vec::new(), feasible: false, starts: Vec::new(), machines: Vec::new() };
    }
    if let Some(sched) = &model.schedule {
        return solve_schedule(sched, seed, stop, report);
    }
    let per = PerList::build(model);
    let n = model.items.len();
    let mut order: Vec<usize> = (0..n).collect();
    shuffle(&mut order, seed);
    let mut state = State::greedy(model, &per, &order);

    let (mut best_lists, mut best_score, mut best_feasible) = snapshot(&per, &state);
    if best_feasible && per.tiers > 0 {
        report(tier_value(&per, &best_score, 0));
    }
    // Local optima visited since the incumbent last improved. Descent moves do
    // NOT reset it (otherwise a kick that descent immediately undoes would keep
    // it at zero and the restart below would never fire). After enough fruitless
    // local optima, restart from a fresh random partition (GRASP-style); until
    // then, kick harder the longer the search has been stuck.
    const RESTART_AFTER: u64 = 25;
    let mut since_improve = 0u64;
    let mut iter = 0u64;

    while !stop.load(Ordering::Relaxed) {
        iter += 1;
        match best_improving_move(&per, &state, stop) {
            Some(mv) => apply_move(&per, &mut state, mv),
            None => {
                let (lists, score, feasible) = snapshot(&per, &state);
                if better(feasible, score, best_feasible, best_score) {
                    best_lists = lists;
                    best_score = score;
                    best_feasible = feasible;
                    if feasible && per.tiers > 0 {
                        report(tier_value(&per, &best_score, 0));
                    }
                    since_improve = 0;
                } else {
                    since_improve += 1;
                }
                if since_improve >= RESTART_AFTER {
                    shuffle(&mut order, seed ^ mix64(iter));
                    state = State::greedy(model, &per, &order);
                    since_improve = 0;
                } else {
                    let strength = 1 + (since_improve / 5) as usize;
                    random_kick(&per, &mut state, seed ^ mix64(iter), strength);
                }
            }
        }
    }

    let (lists, score, feasible) = snapshot(&per, &state);
    if better(feasible, score, best_feasible, best_score) {
        best_lists = lists;
        best_score = score;
        best_feasible = feasible;
    }

    // Report the objective values from the same score that drove the search, so
    // they can never disagree with the accepted solution. When infeasible they
    // are best-effort; `feasible` is the signal to trust.
    let objectives = objective_values(&per, &best_score);
    CollectionSolution { lists: best_lists, objectives, feasible: best_feasible, starts: Vec::new(), machines: Vec::new() }
}

/// Score of an interval schedule: total constraint violation (bound, precedence,
/// resource), then the makespan. Smaller is better, violation first.
/// Duration of an interval under the chosen mode (the fixed duration when it has
/// no modes).
fn iv_duration(iv: &IntervalVar, mode: usize) -> i64 {
    if iv.modes.is_empty() {
        iv.duration
    } else {
        iv.modes[mode].duration
    }
}

/// Chosen machine of an interval, or `-1` when it has no modes (no machine).
fn iv_machine(iv: &IntervalVar, mode: usize) -> i64 {
    if iv.modes.is_empty() {
        -1
    } else {
        iv.modes[mode].machine as i64
    }
}

/// Durations and machines implied by the chosen mode of each interval.
fn mode_view(sched: &Schedule, chosen: &[usize]) -> (Vec<i64>, Vec<i64>) {
    let dur = sched.intervals.iter().zip(chosen).map(|(iv, &m)| iv_duration(iv, m)).collect();
    let mach = sched.intervals.iter().zip(chosen).map(|(iv, &m)| iv_machine(iv, m)).collect();
    (dur, mach)
}

/// Overlap (>= 0) of two intervals' time spans.
fn pair_overlap(starts: &[i64], dur: &[i64], i: usize, j: usize) -> i64 {
    ((starts[i] + dur[i]).min(starts[j] + dur[j]) - starts[i].max(starts[j])).max(0)
}

fn schedule_score(sched: &Schedule, dur: &[i64], mach: &[i64], starts: &[i64]) -> (i64, i64) {
    let mut viol = 0i64;
    let mut makespan = 0i64;
    for (i, iv) in sched.intervals.iter().enumerate() {
        let end = starts[i].saturating_add(dur[i]);
        if starts[i] < 0 {
            viol = viol.saturating_add(-starts[i]);
        }
        if end > iv.horizon {
            viol = viol.saturating_add(end - iv.horizon);
        }
        makespan = makespan.max(end);
    }
    for &(a, b) in &sched.precedences {
        viol = viol.saturating_add((starts[a].saturating_add(dur[a]) - starts[b]).max(0));
    }
    for res in &sched.resources {
        match res {
            Resource::NoOverlap(ivs) => {
                for x in 0..ivs.len() {
                    for y in (x + 1)..ivs.len() {
                        viol = viol.saturating_add(pair_overlap(starts, dur, ivs[x], ivs[y]));
                    }
                }
            }
            Resource::MachineNoOverlap => {
                // Group moded intervals by their chosen machine; intervals on the
                // same machine may not overlap.
                let n = sched.intervals.len();
                for i in 0..n {
                    if mach[i] < 0 {
                        continue;
                    }
                    for j in (i + 1)..n {
                        if mach[i] == mach[j] {
                            viol = viol.saturating_add(pair_overlap(starts, dur, i, j));
                        }
                    }
                }
            }
            Resource::Cumulative { demands, capacity } => {
                viol = viol.saturating_add(cumulative_overload(demands, *capacity, dur, starts));
            }
        }
    }
    (viol, makespan)
}

/// Total resource-overload area of a cumulative resource (usage above capacity,
/// integrated over time). Usage changes only at interval boundaries.
fn cumulative_overload(demands: &[(usize, i64)], capacity: i64, dur: &[i64], starts: &[i64]) -> i64 {
    let mut times: Vec<i64> = Vec::with_capacity(demands.len() * 2);
    for &(i, _) in demands {
        times.push(starts[i]);
        times.push(starts[i].saturating_add(dur[i]));
    }
    times.sort_unstable();
    times.dedup();
    let mut total = 0i64;
    for w in times.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        let usage: i64 = demands.iter().filter(|&&(i, _)| starts[i] <= t0 && t0 < starts[i] + dur[i]).map(|&(_, d)| d).sum();
        let over = (usage - capacity).max(0);
        total = total.saturating_add(over.saturating_mul(t1 - t0));
    }
    total
}

/// Earliest start of each interval respecting precedence only (a longest-path
/// forward pass); resources are left to the search to fix.
fn earliest_starts(sched: &Schedule, dur: &[i64]) -> Vec<i64> {
    let mut s = vec![0i64; sched.intervals.len()];
    for _ in 0..sched.intervals.len() {
        let mut changed = false;
        for &(a, b) in &sched.precedences {
            let need = s[a].saturating_add(dur[a]);
            if need > s[b] {
                s[b] = need;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    s
}

/// Candidate start times for shifting interval `i`: its precedence-earliest, the
/// ends of other intervals (for resource packing), and 0; clamped to its window.
fn schedule_candidates(sched: &Schedule, dur: &[i64], starts: &[i64], i: usize) -> Vec<i64> {
    let hi = (sched.intervals[i].horizon - dur[i]).max(0);
    let est = sched
        .precedences
        .iter()
        .filter(|&&(_, b)| b == i)
        .map(|&(a, _)| starts[a] + dur[a])
        .max()
        .unwrap_or(0);
    let mut cands = vec![0, est];
    for (j, &sj) in starts.iter().enumerate() {
        if j != i {
            cands.push(sj + dur[j]);
            cands.push((sj - dur[i]).max(0));
        }
    }
    for c in &mut cands {
        *c = (*c).clamp(0, hi);
    }
    cands.sort_unstable();
    cands.dedup();
    cands
}

/// Solve an interval-scheduling subproblem by local search. The decisions are
/// each interval's start time and, for flexible (moded) intervals, its mode
/// (which sets the duration and machine). Moves shift a start or switch a mode.
fn solve_schedule(sched: &Schedule, seed: u64, stop: &AtomicBool, report: &mut dyn FnMut(i64)) -> CollectionSolution {
    let n = sched.intervals.len();
    let mut chosen = vec![0usize; n];
    let (mut dur, mut mach) = mode_view(sched, &chosen);
    let mut starts = earliest_starts(sched, &dur);
    let mut cur = schedule_score(sched, &dur, &mach, &starts);
    let mut best_starts = starts.clone();
    let mut best_chosen = chosen.clone();
    let mut best = cur;
    if best.0 == 0 {
        report(best.1);
    }
    const RESTART_AFTER: u64 = 25;
    let mut since_improve = 0u64;
    let mut iter = 0u64;

    while !stop.load(Ordering::Relaxed) {
        iter += 1;
        let mut moved = false;
        'scan: for i in 0..n {
            // (a) Shift the start under the current mode.
            for t in schedule_candidates(sched, &dur, &starts, i) {
                if t == starts[i] {
                    continue;
                }
                let old = starts[i];
                starts[i] = t;
                let trial = schedule_score(sched, &dur, &mach, &starts);
                if trial < cur {
                    cur = trial;
                    moved = true;
                    break 'scan;
                }
                starts[i] = old;
            }
            // (b) Switch this interval's mode (machine / duration).
            let modes = sched.intervals[i].modes.len();
            for m in 0..modes {
                if m == chosen[i] {
                    continue;
                }
                let (om, od, oma, os) = (chosen[i], dur[i], mach[i], starts[i]);
                chosen[i] = m;
                dur[i] = iv_duration(&sched.intervals[i], m);
                mach[i] = iv_machine(&sched.intervals[i], m);
                starts[i] = os.clamp(0, (sched.intervals[i].horizon - dur[i]).max(0));
                let trial = schedule_score(sched, &dur, &mach, &starts);
                if trial < cur {
                    cur = trial;
                    moved = true;
                    break 'scan;
                }
                chosen[i] = om;
                dur[i] = od;
                mach[i] = oma;
                starts[i] = os;
            }
            if i % 64 == 0 && stop.load(Ordering::Relaxed) {
                break;
            }
        }
        if moved {
            continue;
        }
        // Local optimum: record, then kick or restart.
        if cur < best {
            best = cur;
            best_starts = starts.clone();
            best_chosen = chosen.clone();
            if cur.0 == 0 {
                report(cur.1);
            }
            since_improve = 0;
        } else {
            since_improve += 1;
        }
        if since_improve >= RESTART_AFTER {
            // Random modes, then precedence-earliest starts with jitter.
            for (i, c) in chosen.iter_mut().enumerate() {
                let modes = sched.intervals[i].modes.len().max(1);
                *c = (mix64(seed ^ mix64(iter ^ 0x9e37).wrapping_add(i as u64)) % modes as u64) as usize;
            }
            (dur, mach) = mode_view(sched, &chosen);
            starts = earliest_starts(sched, &dur);
            for (i, s) in starts.iter_mut().enumerate() {
                let hi = (sched.intervals[i].horizon - dur[i]).max(0);
                if hi > 0 {
                    *s = (mix64(seed ^ mix64(iter).wrapping_add(i as u64)) % (hi as u64 + 1)) as i64;
                }
            }
            since_improve = 0;
        } else if n > 0 {
            let i = (mix64(seed ^ mix64(iter)) % n as u64) as usize;
            let hi = (sched.intervals[i].horizon - dur[i]).max(0);
            if hi > 0 {
                starts[i] = (mix64(seed ^ mix64(iter ^ 0x5151)) % (hi as u64 + 1)) as i64;
            }
        }
        cur = schedule_score(sched, &dur, &mach, &starts);
    }

    if cur < best {
        best = cur;
        best_starts = starts.clone();
        best_chosen = chosen.clone();
    }
    let feasible = best.0 == 0;
    let machines: Vec<i64> = sched.intervals.iter().zip(&best_chosen).map(|(iv, &m)| iv_machine(iv, m)).collect();
    CollectionSolution {
        lists: Vec::new(),
        objectives: if feasible && sched.minimize_makespan { vec![best.1] } else { Vec::new() },
        feasible,
        starts: if feasible { best_starts } else { Vec::new() },
        machines: if feasible { machines } else { Vec::new() },
    }
}

fn snapshot(per: &PerList, state: &State) -> (Vec<Vec<i32>>, Score, bool) {
    let score = full_score(per, state);
    let feasible = score.violation == 0;
    (state.lists.clone(), score, feasible)
}

/// Prefer feasible over infeasible, then better lexicographic score.
fn better(feasible: bool, score: Score, best_feasible: bool, best_score: Score) -> bool {
    match (feasible, best_feasible) {
        (true, false) => true,
        (false, true) => false,
        _ => score < best_score,
    }
}

#[derive(Clone, Copy)]
enum Move {
    Relocate { src: usize, src_pos: usize, dst: usize, dst_pos: usize },
    Swap { a: usize, a_pos: usize, b: usize, b_pos: usize },
    Reverse { list: usize, i: usize, j: usize },
}

/// Score of the state if one list's cached score were replaced by `nl`.
fn delta_one(per: &PerList, base: (i64, [i64; MAX_TIERS]), scores: &[ListScore], l: usize, nl: ListScore) -> Score {
    let v = base.0.saturating_sub(scores[l].violation).saturating_add(nl.violation);
    let mut raw = base.1;
    for ((r, &old), &new) in raw.iter_mut().zip(scores[l].objectives.iter()).zip(nl.objectives.iter()) {
        *r = r.saturating_sub(old).saturating_add(new);
    }
    signed(per, v, raw)
}

/// Score of the state if lists `a` and `b` were replaced by `na`/`nb`.
fn delta_two(per: &PerList, base: (i64, [i64; MAX_TIERS]), scores: &[ListScore], a: usize, na: ListScore, b: usize, nb: ListScore) -> Score {
    let v = base.0.saturating_sub(scores[a].violation).saturating_sub(scores[b].violation).saturating_add(na.violation).saturating_add(nb.violation);
    let mut raw = base.1;
    let (oa, ob) = (&scores[a].objectives, &scores[b].objectives);
    for ((((r, &sa), &sb), &xa), &xb) in raw.iter_mut().zip(oa.iter()).zip(ob.iter()).zip(na.objectives.iter()).zip(nb.objectives.iter()) {
        *r = r.saturating_sub(sa).saturating_sub(sb).saturating_add(xa).saturating_add(xb);
    }
    signed(per, v, raw)
}

/// Copy `list` into `buf` (reusing its capacity) with one edit applied.
fn build_removed(buf: &mut Vec<i32>, list: &[i32], pos: usize) {
    buf.clear();
    buf.extend_from_slice(list);
    buf.remove(pos);
}
fn build_inserted(buf: &mut Vec<i32>, list: &[i32], pos: usize, item: i32) {
    buf.clear();
    buf.extend_from_slice(list);
    buf.insert(pos.min(list.len()), item);
}
fn build_replaced(buf: &mut Vec<i32>, list: &[i32], pos: usize, item: i32) {
    buf.clear();
    buf.extend_from_slice(list);
    buf[pos] = item;
}
fn build_reversed(buf: &mut Vec<i32>, list: &[i32], i: usize, j: usize) {
    buf.clear();
    buf.extend_from_slice(list);
    buf[i..=j].reverse();
}
fn build_moved_within(buf: &mut Vec<i32>, list: &[i32], from: usize, to: usize, item: i32) {
    buf.clear();
    buf.extend_from_slice(list);
    buf.remove(from);
    buf.insert(to.min(buf.len()), item);
}

/// A single local edit to one list. Carries enough to (a) compute the affected
/// reductions' delta in O(edit) and (b) materialise the edited list for the
/// full-recompute fallback. Positions match the `build_*` helpers exactly.
#[derive(Clone, Copy, Debug)]
enum Edit {
    Remove { pos: usize },
    Insert { pos: usize, item: i32 },
    MoveWithin { from: usize, to: usize },
    Replace { pos: usize, item: i32 },
    Reverse { i: usize, j: usize },
}

impl Edit {
    /// Length of the list after the edit.
    fn new_len(&self, len: usize) -> usize {
        match self {
            Edit::Remove { .. } => len - 1,
            Edit::Insert { .. } => len + 1,
            _ => len,
        }
    }

    /// Materialise the edited list into `buf` (the fallback path).
    fn apply(&self, buf: &mut Vec<i32>, list: &[i32]) {
        match *self {
            Edit::Remove { pos } => build_removed(buf, list, pos),
            Edit::Insert { pos, item } => build_inserted(buf, list, pos, item),
            Edit::MoveWithin { from, to } => build_moved_within(buf, list, from, to, list[from]),
            Edit::Replace { pos, item } => build_replaced(buf, list, pos, item),
            Edit::Reverse { i, j } => build_reversed(buf, list, i, j),
        }
    }
}

/// Delta of a per-item (order-independent) value reduction under `edit`, where
/// `value(item)` is the body contribution of one item.
fn items_value_delta(list: &[i32], edit: &Edit, value: impl Fn(i32) -> i64) -> i64 {
    match *edit {
        Edit::Remove { pos } => -value(list[pos]),
        Edit::Insert { item, .. } => value(item),
        Edit::Replace { pos, item } => value(item).saturating_sub(value(list[pos])),
        Edit::MoveWithin { .. } | Edit::Reverse { .. } => 0,
    }
}

/// Delta of a closed-tour edge-sum reduction (`Edges { start, end }`) under
/// `edit`, where `edge(from, to)` is the body contribution of one edge. The tour
/// is `[start, list.., end]`; only edges local to the edit are touched.
fn edges_value_delta(list: &[i32], start: i32, end: i32, edit: &Edit, edge: impl Fn(i32, i32) -> i64) -> i64 {
    let n = list.len();
    // The i-th tour node: 0 -> start, n+1 -> end, else list[i-1].
    let node = |i: usize| -> i32 {
        if i == 0 {
            start
        } else if i == n + 1 {
            end
        } else {
            list[i - 1]
        }
    };
    match *edit {
        Edit::Remove { pos } => {
            let t = pos + 1;
            edge(node(t - 1), node(t + 1)).saturating_sub(edge(node(t - 1), node(t))).saturating_sub(edge(node(t), node(t + 1)))
        }
        Edit::Insert { pos, item } => {
            let (a, b) = (node(pos), node(pos + 1));
            edge(a, item).saturating_add(edge(item, b)).saturating_sub(edge(a, b))
        }
        Edit::Replace { pos, item } => {
            let t = pos + 1;
            let (l, r, old) = (node(t - 1), node(t + 1), node(t));
            edge(l, item).saturating_add(edge(item, r)).saturating_sub(edge(l, old)).saturating_sub(edge(old, r))
        }
        Edit::Reverse { i, j } => {
            let before = node(i);
            let after = node(j + 2);
            let mut old = 0i64;
            for t in i..=j + 1 {
                old = old.saturating_add(edge(node(t), node(t + 1)));
            }
            // New node order across positions i..=j+2: before, list[j..=i], after.
            let mut new = edge(before, list[j]);
            for w in (i..=j).rev().collect::<Vec<_>>().windows(2) {
                new = new.saturating_add(edge(list[w[0]], list[w[1]]));
            }
            new = new.saturating_add(edge(list[i], after));
            new.saturating_sub(old)
        }
        Edit::MoveWithin { from, to } => {
            let item = list[from];
            // Removal delta on the original tour.
            let t = from + 1;
            let rem = edge(node(t - 1), node(t + 1)).saturating_sub(edge(node(t - 1), node(t))).saturating_sub(edge(node(t), node(t + 1)));
            // Insertion delta into the post-removal tour L_rm (length n-1).
            let m = n - 1;
            let to2 = to.min(m);
            let lrm = |i: usize| -> i32 { list[if i < from { i } else { i + 1 }] };
            let node_rm = |i: usize| -> i32 {
                if i == 0 {
                    start
                } else if i == m + 1 {
                    end
                } else {
                    lrm(i - 1)
                }
            };
            let (a, b) = (node_rm(to2), node_rm(to2 + 1));
            let ins = edge(a, item).saturating_add(edge(item, b)).saturating_sub(edge(a, b));
            rem.saturating_add(ins)
        }
    }
}

/// Delta of one supported reduction's raw value under `edit`.
fn reduction_delta(r: &Reduction, list: &[i32], edit: &Edit) -> i64 {
    match (r.op, &r.iterable) {
        (ReduceOp::Sum, Iterable::Items(_)) => items_value_delta(list, edit, |item| eval_expr(&r.arena.exprs, r.body, &[i64::from(item)])),
        (ReduceOp::Count, Iterable::Items(_)) => items_value_delta(list, edit, |item| i64::from(eval_expr(&r.arena.exprs, r.body, &[i64::from(item)]) != 0)),
        (ReduceOp::Used, Iterable::Items(_)) => {
            let old = list.len();
            i64::from(edit.new_len(old) > 0) - i64::from(old > 0)
        }
        (ReduceOp::Sum, Iterable::Edges { start, end, .. }) => {
            edges_value_delta(list, *start, *end, edit, |from, to| eval_expr(&r.arena.exprs, r.body, &[i64::from(from), i64::from(to)]))
        }
        _ => unreachable!("reduction_delta on an unsupported reduction"),
    }
}

/// Trial score of list `idx` after `edit`, computed incrementally from the cached
/// base score and constraint values when the list is fully incremental, else by
/// materialising the edited list and rescoring it in full.
fn trial_list_score(per: &PerList, state: &State, idx: usize, edit: Edit, scratch: &mut Vec<i32>) -> ListScore {
    if per.list_incremental[idx] {
        let list = &state.lists[idx];
        let base = &state.scores[idx];
        let mut objectives = base.objectives;
        for (t, slot) in objectives.iter_mut().enumerate() {
            for r in &per.objective[idx][t] {
                *slot = slot.saturating_add(reduction_delta(r, list, &edit));
            }
        }
        let mut violation = base.violation;
        for (ci, c) in per.constraints[idx].iter().enumerate() {
            let old = state.con_vals[idx][ci].expect("supported constraint reduction is always defined");
            let new = old.saturating_add(reduction_delta(&c.reduction, list, &edit));
            violation = violation.saturating_sub(violation_of(old, c.op, c.rhs)).saturating_add(violation_of(new, c.op, c.rhs));
        }
        ListScore { violation, objectives }
    } else {
        edit.apply(scratch, &state.lists[idx]);
        list_score(per, idx, scratch)
    }
}

/// Test oracle: for `model` with the given list contents, assert the incremental
/// trial score of every single-list edit equals a full recompute of the edited
/// list. Panics on the first mismatch; returns the number of edits checked. The
/// `model` must use only incremental-supported reductions (so the delta paths,
/// not the fallback, are exercised). Exposed for the integration oracle, which
/// keeps the actual test out of `src/`.
#[doc(hidden)]
pub fn audit_incremental(model: &CollectionModel, lists: &[Vec<i32>]) -> usize {
    let per = PerList::build(model);
    let k = lists.len();
    assert!((0..k).all(|i| per.list_incremental[i]), "audit model must be fully incremental");
    let scores: Vec<ListScore> = (0..k).map(|i| list_score(&per, i, &lists[i])).collect();
    let con_vals: Vec<Vec<Option<i64>>> = (0..k).map(|i| compute_con_vals(&per, i, &lists[i])).collect();
    let mut item_list = vec![0usize; model.items.len()];
    for (l, lst) in lists.iter().enumerate() {
        for &v in lst {
            if let Some(&i) = per.globals.value_to_idx.get(&v) {
                item_list[i] = l;
            }
        }
    }
    let global_viol = per.globals.total(&item_list);
    let state = State { lists: lists.to_vec(), scores, con_vals, item_list, global_viol };

    let mut scratch = Vec::new();
    let mut full_buf = Vec::new();
    let mut checked = 0usize;
    let same = |a: &ListScore, b: &ListScore| a.violation == b.violation && a.objectives == b.objectives;
    for (idx, list) in lists.iter().enumerate() {
        let n = list.len();
        let mut edits: Vec<Edit> = Vec::new();
        for pos in 0..n {
            edits.push(Edit::Remove { pos });
        }
        for &item in &model.items {
            for pos in 0..=n {
                edits.push(Edit::Insert { pos, item });
            }
            for pos in 0..n {
                edits.push(Edit::Replace { pos, item });
            }
        }
        for from in 0..n {
            for to in 0..n {
                if from != to {
                    edits.push(Edit::MoveWithin { from, to });
                }
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                edits.push(Edit::Reverse { i, j });
            }
        }
        for edit in edits {
            let inc = trial_list_score(&per, &state, idx, edit, &mut scratch);
            edit.apply(&mut full_buf, list);
            let full = list_score(&per, idx, &full_buf);
            assert!(same(&inc, &full), "incremental != full for list {idx} edit {edit:?}: inc=({},{:?}) full=({},{:?})", inc.violation, inc.objectives, full.violation, full.objectives);
            checked += 1;
        }
    }
    checked
}

/// First-improvement local move: return the first relocate / swap / segment
/// reversal that strictly lowers the score, or `None` at a local optimum (or
/// when `stop` fires). Each candidate is scored in O(1) against the cached base
/// totals and built into a reused scratch buffer, so no allocation happens per
/// candidate. `stop` is polled per candidate (inside the innermost loops) so
/// even an unbalanced state with very long lists honours the time limit.
fn best_improving_move(per: &PerList, state: &State, stop: &AtomicBool) -> Option<Move> {
    let base = base_totals(&state.scores);
    let current = full_score(per, state);
    let gviol = state.global_viol;
    let k = state.lists.len();
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut polled = 0u32;
    let mut stopped = || {
        polled = polled.wrapping_add(1);
        polled.is_multiple_of(1024) && stop.load(Ordering::Relaxed)
    };
    // Add the current global violation plus a move's global delta to a per-list
    // trial score, so cross-list constraints steer the search too.
    let with_global = |s: Score, gdelta: i64| Score { violation: s.violation.saturating_add(gviol).saturating_add(gdelta), tiers: s.tiers };

    for src in 0..k {
        for src_pos in 0..state.lists[src].len() {
            let item = state.lists[src][src_pos];
            for dst in 0..k {
                if dst == src {
                    let len = state.lists[src].len();
                    for dst_pos in 0..len {
                        if dst_pos == src_pos {
                            continue;
                        }
                        if stopped() {
                            return None;
                        }
                        let nl = trial_list_score(per, state, src, Edit::MoveWithin { from: src_pos, to: dst_pos }, &mut a);
                        // Within-list move: no item changes list, so gdelta = 0.
                        if with_global(delta_one(per, base, &state.scores, src, nl), 0) < current {
                            return Some(Move::Relocate { src, src_pos, dst, dst_pos });
                        }
                    }
                } else {
                    let na = trial_list_score(per, state, src, Edit::Remove { pos: src_pos }, &mut a);
                    let gd = per.globals.delta(&state.item_list, &[(item, dst)]);
                    for dst_pos in 0..=state.lists[dst].len() {
                        if stopped() {
                            return None;
                        }
                        let nb = trial_list_score(per, state, dst, Edit::Insert { pos: dst_pos, item }, &mut b);
                        if with_global(delta_two(per, base, &state.scores, src, na, dst, nb), gd) < current {
                            return Some(Move::Relocate { src, src_pos, dst, dst_pos });
                        }
                    }
                }
            }
        }
    }
    for x in 0..k {
        for y in (x + 1)..k {
            for xp in 0..state.lists[x].len() {
                for yp in 0..state.lists[y].len() {
                    if stopped() {
                        return None;
                    }
                    let (vx, vy) = (state.lists[x][xp], state.lists[y][yp]);
                    let na = trial_list_score(per, state, x, Edit::Replace { pos: xp, item: vy }, &mut a);
                    let nb = trial_list_score(per, state, y, Edit::Replace { pos: yp, item: vx }, &mut b);
                    let gd = per.globals.delta(&state.item_list, &[(vx, y), (vy, x)]);
                    if with_global(delta_two(per, base, &state.scores, x, na, y, nb), gd) < current {
                        return Some(Move::Swap { a: x, a_pos: xp, b: y, b_pos: yp });
                    }
                }
            }
        }
    }
    for list in 0..k {
        let len = state.lists[list].len();
        for i in 0..len {
            for j in (i + 1)..len {
                if stopped() {
                    return None;
                }
                let nl = trial_list_score(per, state, list, Edit::Reverse { i, j }, &mut a);
                if with_global(delta_one(per, base, &state.scores, list, nl), 0) < current {
                    return Some(Move::Reverse { list, i, j });
                }
            }
        }
    }
    None
}

fn apply_move(per: &PerList, state: &mut State, mv: Move) {
    match mv {
        Move::Relocate { src, src_pos, dst, dst_pos } => {
            let item = state.lists[src].remove(src_pos);
            let pos = dst_pos.min(state.lists[dst].len());
            state.lists[dst].insert(pos, item);
            state.rescore(per, src);
            state.rescore(per, dst);
            if src != dst {
                state.set_item_list(per, item, dst);
                state.global_viol = per.globals.total(&state.item_list);
            }
        }
        Move::Swap { a, a_pos, b, b_pos } => {
            let tmp = state.lists[a][a_pos];
            state.lists[a][a_pos] = state.lists[b][b_pos];
            state.lists[b][b_pos] = tmp;
            state.rescore(per, a);
            state.rescore(per, b);
            // After the swap, lists[a][a_pos] holds the item that was in b.
            state.set_item_list(per, state.lists[a][a_pos], a);
            state.set_item_list(per, state.lists[b][b_pos], b);
            state.global_viol = per.globals.total(&state.item_list);
        }
        Move::Reverse { list, i, j } => {
            state.lists[list][i..=j].reverse();
            state.rescore(per, list);
            // Reversal keeps every item in the same list, so globals are unchanged.
        }
    }
}

fn shuffle(order: &mut [usize], seed: u64) {
    for i in (1..order.len()).rev() {
        let j = (mix64(seed.wrapping_add(i as u64)) % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}

/// Move `strength` random items, each to a random other list, to escape a local
/// minimum. A larger strength is a bigger perturbation for a deeper basin.
fn random_kick(per: &PerList, state: &mut State, seed: u64, strength: usize) {
    if state.lists.len() < 2 {
        return;
    }
    for step in 0..strength.max(1) {
        let s = seed ^ mix64(step as u64);
        let total: usize = state.lists.iter().map(Vec::len).sum();
        if total == 0 {
            return;
        }
        let mut pick = (mix64(s) % total as u64) as usize;
        let mut src = 0;
        let mut src_pos = 0;
        for (r, l) in state.lists.iter().enumerate() {
            if pick < l.len() {
                src = r;
                src_pos = pick;
                break;
            }
            pick -= l.len();
        }
        let dst = (mix64(s ^ 0x9E37) % state.lists.len() as u64) as usize;
        if dst == src {
            continue;
        }
        let item = state.lists[src].remove(src_pos);
        state.lists[dst].push(item);
        state.rescore(per, src);
        state.rescore(per, dst);
        state.set_item_list(per, item, dst);
    }
    state.global_viol = per.globals.total(&state.item_list);
}
