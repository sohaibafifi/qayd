//! List variable declarations and list-objective evaluation.

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) mod external;
mod ir;
pub(crate) mod scan;
mod validate;
mod verify;

pub use external::{external_function_registered, register_external_function};
pub(crate) use ir::{BoundReport, CollectionModel, CollectionSolution, IntervalVar, Mode, Schedule};
pub use ir::{
    Constraint, Expr, ExprArena, ExprId, FixedPoint, GlobalConstraint, Iterable, MaxTerm, ObjectiveTier, Op, ReduceOp, Reduction, Resource,
};
pub(crate) use verify::verify_collection_solution_interruptible;
#[cfg(test)]
pub(crate) use verify::{audit_record_final_verification_boundary, audit_verification_calls, verify_collection_solution};

/// Reference to a list declaration inside [`Model`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ListVarRef(pub usize);

/// Structured list declaration.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ListDecl {
    /// Immutable item universe for this list.
    pub universe: Vec<i32>,
    /// Ordered sequences and unordered assignment sets share the same physical
    /// collection engine but keep distinct semantic identities.
    pub ordering: ListOrdering,
    /// Hidden remainder lists are explicit model objects, never inferred from
    /// a missing objective reference.
    pub role: ListRole,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ListOrdering {
    Ordered,
    Unordered,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ListRole {
    Decision,
    HiddenRemainder,
    Derived,
}

/// Current list reduction representation during the migration.
pub type ListReduction = Reduction;

/// Current list max-objective component representation during the migration.
pub type ListMaxTerm = MaxTerm;

/// Current list constraint representation during the migration.
pub type ListReductionConstraint = Constraint;

/// Current list reduction operator representation during the migration.
pub type ListReduceOp = ReduceOp;

/// Current list iterable representation during the migration.
pub type ListIterable = Iterable;

/// Structured exact objective term over one list.
#[derive(Clone)]
pub enum ListObjectiveTerm {
    /// Sum of item weights in a list.
    Sum { list: usize, weights: Vec<(i32, i64)>, coeff: i64 },
    /// Count of items whose weight/predicate value is non-zero.
    Count { list: usize, weights: Vec<(i32, i64)>, coeff: i64 },
    /// `1` when the list is non-empty, else `0`.
    Used { list: usize, coeff: i64 },
    /// Any reduction that exact search can score by evaluating a complete list
    /// assignment. This is not linearizable but is correct in enumerate mode.
    Reduction(Reduction),
}

/// Parsed exact-search form of `coeff * max(sum(group_0), ...)`.
#[derive(Clone)]
pub struct ListObjectiveMaxTerm {
    pub groups: Vec<Vec<ListObjectiveTerm>>,
    pub coeff: i64,
}

/// One domain exact lexicographic objective tier.
#[derive(Clone)]
pub struct ListObjectiveTier {
    /// Whether this tier is minimized.
    pub minimize: bool,
    /// Terms summed to form the tier value.
    pub terms: Vec<ListObjectiveTerm>,
    /// Max components added to the tier value.
    pub max_terms: Option<Vec<ListObjectiveMaxTerm>>,
}

/// Parse list objective tiers that domain exact search can evaluate.
#[cfg(test)]
pub(crate) fn list_objective_tiers(tiers: &[ObjectiveTier], items: &[i32]) -> Option<Vec<ListObjectiveTier>> {
    let mut parsed = Vec::with_capacity(tiers.len());
    for tier in tiers {
        let mut terms = Vec::with_capacity(tier.terms.len());
        for reduction in &tier.terms {
            terms.push(list_objective_term(reduction, items)?);
        }
        let max_terms = match &tier.max_terms {
            Some(max_terms) => Some(
                max_terms
                    .iter()
                    .map(|term| {
                        Some(ListObjectiveMaxTerm {
                            groups: term
                                .groups
                                .iter()
                                .map(|group| {
                                    group.iter().map(|reduction| list_objective_term(reduction, items)).collect::<Option<Vec<_>>>()
                                })
                                .collect::<Option<Vec<_>>>()?,
                            coeff: term.coeff,
                        })
                    })
                    .collect::<Option<Vec<_>>>()?,
            ),
            None => None,
        };
        parsed.push(ListObjectiveTier { minimize: tier.minimize, terms, max_terms });
    }
    Some(parsed)
}

/// Evaluate parsed list objective tiers on concrete list contents.
pub(crate) fn evaluate_list_objectives(tiers: &[ListObjectiveTier], lists: &[Vec<i32>]) -> Vec<i64> {
    tiers
        .iter()
        .map(|tier| {
            let mut value = eval_list_objective_terms(&tier.terms, lists, tier.minimize);
            if let Some(max_terms) = &tier.max_terms {
                for term in max_terms {
                    let component =
                        term.groups.iter().map(|group| eval_list_objective_terms(group, lists, tier.minimize)).max().unwrap_or(0);
                    value = value.saturating_add(component.saturating_mul(term.coeff));
                }
            }
            value
        })
        .collect()
}

/// Lexicographic comparison of list objective vectors.
pub(crate) fn list_objectives_better(candidate: &[i64], incumbent: &[i64], tiers: &[ListObjectiveTier]) -> bool {
    for ((candidate, incumbent), tier) in candidate.iter().zip(incumbent).zip(tiers) {
        if candidate == incumbent {
            continue;
        }
        return if tier.minimize { candidate < incumbent } else { candidate > incumbent };
    }
    false
}

#[cfg(test)]
fn list_objective_term(reduction: &Reduction, items: &[i32]) -> Option<ListObjectiveTerm> {
    if let Iterable::Items(list) = &reduction.iterable {
        match reduction.op {
            ReduceOp::Sum => {
                return Some(ListObjectiveTerm::Sum { list: *list, weights: list_item_values(reduction, items)?, coeff: reduction.coeff });
            }
            ReduceOp::Count => {
                return Some(ListObjectiveTerm::Count {
                    list: *list,
                    weights: list_item_values(reduction, items)?,
                    coeff: reduction.coeff,
                });
            }
            ReduceOp::Used => return Some(ListObjectiveTerm::Used { list: *list, coeff: reduction.coeff }),
            ReduceOp::Min | ReduceOp::Max | ReduceOp::SelectKth(_) => {}
        }
    }
    Some(ListObjectiveTerm::Reduction(reduction.clone()))
}

fn eval_list_objective_terms(terms: &[ListObjectiveTerm], lists: &[Vec<i32>], minimize: bool) -> i64 {
    let undefined = if minimize { i64::MAX / 4 } else { i64::MIN / 4 };
    terms.iter().fold(0i64, |acc, term| match eval_list_objective_term(term, lists) {
        Some(value) => acc.saturating_add(value),
        None => acc.saturating_add(undefined),
    })
}

fn eval_list_objective_term(term: &ListObjectiveTerm, lists: &[Vec<i32>]) -> Option<i64> {
    Some(match term {
        ListObjectiveTerm::Sum { list, weights, coeff } => lists
            .get(*list)
            .map_or(0, |items| items.iter().fold(0i64, |sum, item| sum.saturating_add(list_item_weight(weights, *item))))
            .saturating_mul(*coeff),
        ListObjectiveTerm::Count { list, weights, coeff } => lists
            .get(*list)
            .map_or(0, |items| {
                items.iter().filter(|item| list_item_weight(weights, **item) != 0).fold(0i64, |count, _| count.saturating_add(1))
            })
            .saturating_mul(*coeff),
        ListObjectiveTerm::Used { list, coeff } => {
            i64::from(lists.get(*list).is_some_and(|items| !items.is_empty())).saturating_mul(*coeff)
        }
        ListObjectiveTerm::Reduction(reduction) => eval_reduction_on_lists(reduction, lists)?,
    })
}

pub(crate) fn eval_reduction_on_lists(reduction: &Reduction, lists: &[Vec<i32>]) -> Option<i64> {
    eval_reduction_on_lists_impl(reduction, lists, None).ok().flatten()
}

/// Evaluate one reduction on the concrete contents selected by its iterable.
///
/// This is the runtime semantic used by both canonical replay and full LS
/// recomputation. Incremental LS caches may optimize it, but must agree with
/// this function whenever they materialize a complete list.
pub(crate) fn eval_reduction_on_contents(reduction: &Reduction, contents: &[i32]) -> Option<i64> {
    eval_reduction_on_contents_impl(reduction, contents, None).ok().flatten()
}

pub(crate) fn eval_reduction_on_lists_interruptible(
    reduction: &Reduction,
    lists: &[Vec<i32>],
    stop: &AtomicBool,
) -> Result<Option<i64>, ()> {
    eval_reduction_on_lists_impl(reduction, lists, Some(stop))
}

fn eval_reduction_on_lists_impl(reduction: &Reduction, lists: &[Vec<i32>], stop: Option<&AtomicBool>) -> Result<Option<i64>, ()> {
    let Some(contents) = lists.get(reduction.iterable.list()) else {
        return Ok(None);
    };
    eval_reduction_on_contents_impl(reduction, contents, stop)
}

fn eval_reduction_on_contents_impl(reduction: &Reduction, contents: &[i32], stop: Option<&AtomicBool>) -> Result<Option<i64>, ()> {
    let check_stop = || {
        if stop.is_some_and(|stop| stop.load(Ordering::Acquire)) {
            Err(())
        } else {
            Ok(())
        }
    };
    check_stop()?;
    let arena = &reduction.arena.exprs;
    let body = reduction.body;
    let mut sum = 0i64;
    let mut count = 0i64;
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    let mut steps = 0u64;
    let select_k = match reduction.op {
        ReduceOp::SelectKth(k) => Some(k),
        _ => None,
    };
    let mut smallest = BinaryHeap::new();
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
        if let Some(k) = select_k {
            smallest.push(v);
            if smallest.len() > k.saturating_add(1) {
                smallest.pop();
            }
        }
    };

    let eval = |id, args: &[i64]| match stop {
        Some(stop) => eval_objective_expr_interruptible(arena, id, args, stop),
        None => Ok(eval_objective_expr(arena, id, args)),
    };

    match reduction.iterable {
        Iterable::Items(_) => {
            for &item in contents {
                check_stop()?;
                fold(eval(body, &[i64::from(item)])?);
            }
        }
        Iterable::SetItems(_) => {
            // Every SetItems reduction is permutation invariant. Iterating the
            // canonical assignment directly avoids a non-interruptible clone
            // and sort at the final verification boundary.
            for &item in contents {
                check_stop()?;
                fold(eval(body, &[i64::from(item)])?);
            }
        }
        Iterable::Edges { start, end, .. } => {
            let n = contents.len();
            for p in 0..=n {
                check_stop()?;
                let from = if p == 0 { start } else { contents[p - 1] };
                let to = if p == n { end } else { contents[p] };
                fold(eval(body, &[i64::from(from), i64::from(to)])?);
            }
        }
        Iterable::Pairs(_) => {
            let n = contents.len();
            for i in 0..n {
                for j in 0..n {
                    check_stop()?;
                    let args = [i64::from(contents[i]), i64::from(contents[j]), i as i64, j as i64];
                    fold(eval(body, &args)?);
                }
            }
        }
        Iterable::Scan { init, boundary, step, end, .. } => {
            let mut acc = init;
            let mut prev = boundary;
            for &cur in contents {
                check_stop()?;
                let new_acc = eval(step, &[i64::from(cur), acc, i64::from(prev)])?;
                fold(eval(body, &[i64::from(cur), new_acc, i64::from(prev)])?);
                acc = new_acc;
                prev = cur;
            }
            // Closing edge to `end`: one more transition, mirroring `Edges`.
            if let Some(end) = end {
                check_stop()?;
                let new_acc = eval(step, &[i64::from(end), acc, i64::from(prev)])?;
                fold(eval(body, &[i64::from(end), new_acc, i64::from(prev)])?);
            }
        }
        Iterable::Windows { size, inner, .. } => {
            let n = contents.len();
            if size > 0 && n >= size {
                for start in 0..=(n - size) {
                    check_stop()?;
                    let mut total = 0i64;
                    for &cur in &contents[start..start + size] {
                        check_stop()?;
                        total = total.saturating_add(eval(inner, &[i64::from(cur)])?);
                    }
                    fold(eval(body, &[0, total])?);
                }
            }
        }
    }

    check_stop()?;
    let value = match reduction.op {
        ReduceOp::Sum => Some(sum),
        ReduceOp::Count => Some(count),
        ReduceOp::Used => Some(i64::from(steps > 0)),
        ReduceOp::Min => (steps > 0).then_some(lo),
        ReduceOp::Max => (steps > 0).then_some(hi),
        ReduceOp::SelectKth(k) => (usize::try_from(steps).unwrap_or(usize::MAX) > k)
            .then(|| *smallest.peek().expect("a populated order-statistic heap has a root")),
    };
    check_stop()?;
    Ok(value.map(|value| value.saturating_mul(reduction.coeff)))
}

pub(crate) fn eval_expr_checked(arena: &[Expr], id: ExprId, args: &[i64]) -> Option<i64> {
    eval_expr_impl(arena, id, args, None).ok()
}

/// Total runtime expression semantics used by search scoring. Invalid array
/// accesses, external failures, and other undefined subexpressions contribute
/// zero, matching reduction replay.
pub(crate) fn eval_expr_runtime(arena: &[Expr], id: ExprId, args: &[i64]) -> i64 {
    eval_expr_checked(arena, id, args).unwrap_or(0)
}

fn eval_expr_checked_interruptible(arena: &[Expr], id: ExprId, args: &[i64], stop: &AtomicBool) -> Result<Option<i64>, ()> {
    match eval_expr_impl(arena, id, args, Some(stop)) {
        Ok(value) => Ok(Some(value)),
        Err(EvalError::Undefined) => Ok(None),
        Err(EvalError::Interrupted) => Err(()),
    }
}

#[derive(Clone, Copy)]
enum EvalError {
    Undefined,
    Interrupted,
}

fn eval_expr_impl(arena: &[Expr], id: ExprId, args: &[i64], stop: Option<&AtomicBool>) -> Result<i64, EvalError> {
    if stop.is_some_and(|stop| stop.load(Ordering::Acquire)) {
        return Err(EvalError::Interrupted);
    }
    let rec = |child| eval_expr_impl(arena, child, args, stop);
    Ok(match arena.get(id.0 as usize).ok_or(EvalError::Undefined)? {
        Expr::Const(c) => *c,
        Expr::Arg(k) => *args.get(*k as usize).ok_or(EvalError::Undefined)?,
        Expr::Array(a, i) => *a.get(usize::try_from(rec(*i)?).map_err(|_| EvalError::Undefined)?).ok_or(EvalError::Undefined)?,
        Expr::Matrix(m, i, j) => {
            let row = usize::try_from(rec(*i)?).map_err(|_| EvalError::Undefined)?;
            let column = usize::try_from(rec(*j)?).map_err(|_| EvalError::Undefined)?;
            *m.get(row).and_then(|values| values.get(column)).ok_or(EvalError::Undefined)?
        }
        Expr::Add(a, b) => rec(*a)?.saturating_add(rec(*b)?),
        Expr::Sub(a, b) => rec(*a)?.saturating_sub(rec(*b)?),
        Expr::Mul(a, b) => rec(*a)?.saturating_mul(rec(*b)?),
        Expr::Mod(a, b) => rec(*a)?.checked_rem(rec(*b)?).unwrap_or(0),
        Expr::Pow(base, exponent) => saturating_pow(rec(*base)?, *exponent),
        Expr::MulScaled(a, b, scale) => scaled_ratio(rec(*a)?, rec(*b)?, *scale, false),
        Expr::DivScaled(a, b, scale) => scaled_ratio(rec(*a)?, rec(*b)?, *scale, true),
        Expr::Min(a, b) => rec(*a)?.min(rec(*b)?),
        Expr::Max(a, b) => rec(*a)?.max(rec(*b)?),
        Expr::Div(a, b) => rec(*a)?.checked_div(rec(*b)?).unwrap_or(0),
        Expr::Abs(a) => rec(*a)?.saturating_abs(),
        Expr::Lt(a, b) => i64::from(rec(*a)? < rec(*b)?),
        Expr::Le(a, b) => i64::from(rec(*a)? <= rec(*b)?),
        Expr::Eq(a, b) => i64::from(rec(*a)? == rec(*b)?),
        Expr::Ne(a, b) => i64::from(rec(*a)? != rec(*b)?),
        Expr::IfThenElse(c, a, b) => {
            if rec(*c)? != 0 {
                rec(*a)?
            } else {
                rec(*b)?
            }
        }
        Expr::PiecewiseLinear { input, points } => piecewise(rec(*input)?, points),
        Expr::External { name, args: external_args } => {
            let values = external_args.iter().map(|id| rec(*id)).collect::<Result<Vec<_>, _>>()?;
            match stop {
                Some(stop) => {
                    external::call_external_function_interruptible(name, &values, stop).map_err(|()| EvalError::Interrupted)?.unwrap_or(0)
                }
                None => external::call_external_function(name, &values).unwrap_or(0),
            }
        }
    })
}

fn eval_objective_expr(arena: &[Expr], id: ExprId, args: &[i64]) -> i64 {
    eval_expr_checked(arena, id, args).unwrap_or(0)
}

fn eval_objective_expr_interruptible(arena: &[Expr], id: ExprId, args: &[i64], stop: &AtomicBool) -> Result<i64, ()> {
    Ok(eval_expr_checked_interruptible(arena, id, args, stop)?.unwrap_or(0))
}

fn saturating_pow(mut base: i64, mut exponent: u32) -> i64 {
    let mut result = 1i64;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result.saturating_mul(base);
        }
        exponent >>= 1;
        if exponent > 0 {
            base = base.saturating_mul(base);
        }
    }
    result
}

fn round_ratio(numerator: i128, denominator: i128) -> i128 {
    if denominator == 0 {
        return 0;
    }
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    if remainder.abs().saturating_mul(2) >= denominator.abs() {
        quotient + if numerator.signum() == denominator.signum() { 1 } else { -1 }
    } else {
        quotient
    }
}

fn scaled_ratio(a: i64, b: i64, scale: i64, divide: bool) -> i64 {
    if scale <= 0 || (divide && b == 0) {
        return 0;
    }
    let (numerator, denominator) =
        if divide { (i128::from(a) * i128::from(scale), i128::from(b)) } else { (i128::from(a) * i128::from(b), i128::from(scale)) };
    round_ratio(numerator, denominator).clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

fn piecewise(input: i64, points: &[(i64, i64)]) -> i64 {
    let Some(&(first_x, first_y)) = points.first() else { return 0 };
    if input <= first_x {
        return first_y;
    }
    for window in points.windows(2) {
        let [(x0, y0), (x1, y1)] = window else { unreachable!() };
        if input <= *x1 {
            let dx = i128::from(input) - i128::from(*x0);
            let dy = i128::from(*y1) - i128::from(*y0);
            let width = i128::from(*x1) - i128::from(*x0);
            return (i128::from(*y0) + round_ratio(dx * dy, width)).clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
        }
    }
    points.last().map_or(first_y, |&(_, y)| y)
}

#[cfg(test)]
fn list_item_values(reduction: &Reduction, items: &[i32]) -> Option<Vec<(i32, i64)>> {
    let mut values = Vec::with_capacity(items.len());
    for &item in items {
        values.push((item, eval_expr_checked(&reduction.arena.exprs, reduction.body, &[i64::from(item)])?));
    }
    Some(values)
}

fn list_item_weight(weights: &[(i32, i64)], item: i32) -> i64 {
    debug_assert!(weights.iter().any(|(known, _)| *known == item));
    weights.iter().find_map(|(known, weight)| (*known == item).then_some(*weight)).unwrap_or(0)
}
