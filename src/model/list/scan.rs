//! Sound integer view of a `Sum` scan reduction for the routing lowering.
//!
//! A scan threads an accumulator along a route (`acc[to] = step(cur, acc[from],
//! prev)`) and emits a per-stop value (`emit(cur, acc[to], prev)`). The exact
//! routing engine reproduces this in the successor world as a resource extension:
//! per-customer accumulator variables pinned by the chosen predecessor arc, with
//! a per-route cumulative sum bounded against the constraint's rhs.
//!
//! This module is the single predicate shared by the classifier gate
//! ([`crate::model::classify`]) and the engine parse ([`crate::engines::routing`])
//! so they never disagree on which scans lower. Anything it rejects falls back to
//! local search.

use crate::expr::{self, Expr as KExpr};
use crate::ids::VarId;

use super::ir::{Constraint, Expr, ExprArena, ExprId, Iterable, Op, ReduceOp};

/// Half of the i32 range: store variables are i32-backed, so every reachable
/// accumulator / cumulative-sum value must fit here with headroom for the
/// propagator's internal `acc[from] + K` add before it clamps.
const I32_HALF_LO: i64 = (i32::MIN as i64) / 2;
const I32_HALF_HI: i64 = (i32::MAX as i64) / 2;

/// Sentinel variable id standing in for the accumulator when deriving bounds.
/// Table indices fold to constants (they are accumulator-free), so this is the
/// only variable leaf the bounds pass ever sees.
const ACC: VarId = VarId(u32::MAX);

/// A homogeneous per-route `Sum` scan, ready to lower onto the successor world.
/// Derived domains are a sound superset of every reachable value (the i32 store
/// width is the ceiling; anything wider falls back to local search).
#[derive(Clone, PartialEq)]
pub(crate) struct ScanRoutingSpec {
    pub arena: ExprArena,
    pub step: ExprId,
    pub emit: ExprId,
    pub init: i64,
    pub boundary: i32,
    pub op: Op,
    pub rhs: i64,
    pub coeff: i64,
    /// Sound `[lo, hi]` for a customer accumulator variable.
    pub acc_dom: (i64, i64),
    /// Sound `[lo, hi]` for a single stop's emitted value.
    pub emit_dom: (i64, i64),
    /// Sound `[lo, hi]` for a route's cumulative `coeff * emit` sum.
    pub total_dom: (i64, i64),
}

/// The list this scan constraint reads, if it is a `Sum` scan; `None` otherwise.
pub(crate) fn scan_constraint_list(constraint: &Constraint) -> Option<usize> {
    match (&constraint.reduction.op, &constraint.reduction.iterable) {
        (ReduceOp::Sum, Iterable::Scan { list, .. }) => Some(*list),
        _ => None,
    }
}

/// Parse a scan constraint into a lowerable signature, or `None` (fall back to
/// local search). Rejects non-`Sum` ops, an accumulator inside a divisor or a
/// table index, and any derived interval that leaves the i32-safe band.
pub(crate) fn scan_routing_signature(constraint: &Constraint, items: &[i32]) -> Option<ScanRoutingSpec> {
    if !matches!(constraint.reduction.op, ReduceOp::Sum) {
        return None;
    }
    // Accept a reified `Sum` scan directly, or a per-item `Sum` over `Items`
    // (`cp.sum(x, |i| ...)`), which is exactly the identity-accumulator scan: the
    // accumulator stays at 0 and the body reads only the current item (`Arg(0)`).
    // Normalising it here -- the one parser both classify and the engine funnel
    // through -- keeps them in lockstep and reuses the whole scan lowering, so a
    // per-item reward gets the same dominance bound and warm start as a scan reward.
    let (owned_arena, step, init, boundary) = match constraint.reduction.iterable {
        Iterable::Scan { init, boundary, step, .. } => (constraint.reduction.arena.clone(), step, init, boundary),
        Iterable::Items(_) => {
            // A well-formed `Items` body binds only `Arg(0)`; reject any body that
            // reads a scan-only binder, since reinterpreting it as the accumulator
            // (`Arg(1)`) or predecessor (`Arg(2)`) would silently change its value.
            if reads_non_item_arg(&constraint.reduction.arena.exprs, constraint.reduction.body) {
                return None;
            }
            let mut arena = constraint.reduction.arena.clone();
            let step = arena.arg(1); // identity accumulator: new_acc = acc (stays init = 0)
            (arena, step, 0, 0)
        }
        _ => return None,
    };
    let arena = &owned_arena.exprs;
    let emit = constraint.reduction.body;
    if step.0 as usize >= arena.len() || emit.0 as usize >= arena.len() {
        return None;
    }

    let n = items.len();
    // Node-constant set: boundary plus every item value. Using this superset for
    // both cur and prev over-approximates but stays sound.
    let mut cv: Vec<i64> = Vec::with_capacity(n + 1);
    cv.push(i64::from(boundary));
    cv.extend(items.iter().map(|&x| i64::from(x)));

    // Detection walk: every node must lower. Reject
    //  - an accumulator (Arg(1)) inside a table index (no finite domain to fold);
    //  - any binder beyond the three scan args;
    //  - a divisor that references the accumulator, or that folds to 0 for some
    //    (cur, prev): LS reads a zero divisor as the value 0, but the kernel `Div`
    //    makes the stop infeasible, so the two engines would disagree;
    //  - a table/const value outside the i32-safe band: the kernel `Expr::eval`
    //    uses plain i64 arithmetic on intermediates (LS saturates), so an
    //    out-of-band leaf could overflow before the bounded result.
    for node in arena {
        let reject = match node {
            Expr::Const(c) => !within_i32_half((*c, *c)),
            Expr::Array(a, i) => uses_acc(arena, *i) || a.iter().any(|&v| !within_i32_half((v, v))),
            Expr::Matrix(m, i, j) => {
                uses_acc(arena, *i) || uses_acc(arena, *j) || m.iter().flatten().any(|&v| !within_i32_half((v, v)))
            }
            Expr::Div(_, b) => uses_acc(arena, *b) || cv.iter().any(|&cur| cv.iter().any(|&prev| eval_index(arena, *b, cur, prev) == 0)),
            Expr::Arg(k) => *k >= 3,
            _ => false,
        };
        if reject {
            return None;
        }
    }

    // Accumulator fixpoint: reach_k is a superset of the position-k accumulator
    // set; acc_dom unions reach_1..reach_n. A route holds at most n customers, so
    // exactly n iterations cover every position (the recurrence keeps growing, so
    // this is not a to-fixpoint loop).
    let step_lowered: Vec<KExpr> =
        cv.iter().flat_map(|&cur| cv.iter().map(move |&prev| (cur, prev))).map(|(cur, prev)| lower_scan(arena, step, cur, prev, &expr::var(ACC))).collect();
    let mut reach = (init, init);
    let mut acc_dom: Option<(i64, i64)> = None;
    for _ in 0..n {
        let mut next = (i64::MAX, i64::MIN);
        for e in &step_lowered {
            let (l, h) = e.bounds(&|v| if v == ACC { reach } else { (0, 0) });
            next = (next.0.min(l), next.1.max(h));
        }
        reach = next;
        acc_dom = Some(acc_dom.map_or(reach, |(lo, hi)| (lo.min(reach.0), hi.max(reach.1))));
        if !within_i32_half(reach) {
            return None;
        }
    }
    let acc_dom = acc_dom.unwrap_or((init, init));

    // Emit bounds: one pass with acc := acc_dom (a superset of every position's
    // new accumulator), cur/prev over CV.
    let mut emit_dom = (i64::MAX, i64::MIN);
    for &cur in &cv {
        for &prev in &cv {
            let e = lower_scan(arena, emit, cur, prev, &expr::var(ACC));
            let (l, h) = e.bounds(&|v| if v == ACC { acc_dom } else { (0, 0) });
            emit_dom = (emit_dom.0.min(l), emit_dom.1.max(h));
        }
    }
    if emit_dom.0 > emit_dom.1 {
        emit_dom = (0, 0); // no stops evaluated (empty item universe)
    }

    // A route's cumulative sum holds 1..=n terms of `coeff * emit`, so it spans
    // `[0, n * coeff*emit]` (0 covers the empty route).
    let coeff = constraint.reduction.coeff;
    let ce_lo = (coeff as i128 * emit_dom.0 as i128).min(coeff as i128 * emit_dom.1 as i128);
    let ce_hi = (coeff as i128 * emit_dom.0 as i128).max(coeff as i128 * emit_dom.1 as i128);
    let n128 = n as i128;
    let total_dom = (clamp(ce_lo.min(n128 * ce_lo).min(0)), clamp(ce_hi.max(n128 * ce_hi).max(0)));

    if !within_i32_half(acc_dom) || !within_i32_half(emit_dom) || !within_i32_half(total_dom) {
        return None;
    }

    Some(ScanRoutingSpec {
        arena: owned_arena,
        step,
        emit,
        init,
        boundary,
        op: constraint.op,
        rhs: constraint.rhs,
        coeff: constraint.reduction.coeff,
        acc_dom,
        emit_dom,
        total_dom,
    })
}

/// Lower an arena expression to a kernel [`KExpr`] with `cur`/`prev` as concrete
/// node constants and `acc` as a supplied kernel expression (a variable or a
/// constant). Table indices are accumulator-free (guaranteed by the detection
/// walk) so they fold to constants against `cur`/`prev`.
pub(crate) fn lower_scan(arena: &[Expr], id: ExprId, cur: i64, prev: i64, acc: &KExpr) -> KExpr {
    let rec = |child: ExprId| lower_scan(arena, child, cur, prev, acc);
    match &arena[id.0 as usize] {
        Expr::Const(c) => expr::int(*c),
        Expr::Arg(0) => expr::int(cur),
        Expr::Arg(1) => acc.clone(),
        Expr::Arg(2) => expr::int(prev),
        Expr::Arg(_) => expr::int(0),
        Expr::Array(a, i) => {
            let idx = eval_index(arena, *i, cur, prev);
            expr::int(usize::try_from(idx).ok().and_then(|u| a.get(u)).copied().unwrap_or(0))
        }
        Expr::Matrix(m, i, j) => {
            let r = eval_index(arena, *i, cur, prev);
            let c = eval_index(arena, *j, cur, prev);
            let v = usize::try_from(r)
                .ok()
                .and_then(|r| m.get(r))
                .and_then(|row| usize::try_from(c).ok().and_then(|c| row.get(c)))
                .copied()
                .unwrap_or(0);
            expr::int(v)
        }
        Expr::Add(a, b) => expr::add(vec![rec(*a), rec(*b)]),
        Expr::Sub(a, b) => expr::sub(rec(*a), rec(*b)),
        Expr::Mul(a, b) => expr::mul(vec![rec(*a), rec(*b)]),
        Expr::Min(a, b) => expr::min_of(vec![rec(*a), rec(*b)]),
        Expr::Max(a, b) => expr::max_of(vec![rec(*a), rec(*b)]),
        Expr::Div(a, b) => expr::div(rec(*a), rec(*b)),
        Expr::Abs(a) => expr::abs(rec(*a)),
        Expr::Lt(a, b) => expr::lt(rec(*a), rec(*b)),
        Expr::Le(a, b) => expr::le(rec(*a), rec(*b)),
        Expr::Eq(a, b) => expr::eq(rec(*a), rec(*b)),
        Expr::Ne(a, b) => expr::ne(rec(*a), rec(*b)),
        Expr::IfThenElse(c, a, b) => expr::ite(rec(*c), rec(*a), rec(*b)),
    }
}

/// Evaluate an accumulator-free index subtree to a constant, mirroring the LS
/// evaluator (saturating arithmetic, out-of-range table reads as 0).
fn eval_index(arena: &[Expr], id: ExprId, cur: i64, prev: i64) -> i64 {
    let rec = |child: ExprId| eval_index(arena, child, cur, prev);
    match &arena[id.0 as usize] {
        Expr::Const(c) => *c,
        Expr::Arg(0) => cur,
        Expr::Arg(2) => prev,
        Expr::Arg(_) => 0,
        Expr::Array(a, i) => {
            let idx = rec(*i);
            usize::try_from(idx).ok().and_then(|u| a.get(u)).copied().unwrap_or(0)
        }
        Expr::Matrix(m, i, j) => {
            let r = rec(*i);
            let c = rec(*j);
            usize::try_from(r)
                .ok()
                .and_then(|r| m.get(r))
                .and_then(|row| usize::try_from(c).ok().and_then(|c| row.get(c)))
                .copied()
                .unwrap_or(0)
        }
        Expr::Add(a, b) => rec(*a).saturating_add(rec(*b)),
        Expr::Sub(a, b) => rec(*a).saturating_sub(rec(*b)),
        Expr::Mul(a, b) => rec(*a).saturating_mul(rec(*b)),
        Expr::Min(a, b) => rec(*a).min(rec(*b)),
        Expr::Max(a, b) => rec(*a).max(rec(*b)),
        Expr::Div(a, b) => {
            let d = rec(*b);
            if d == 0 {
                0
            } else {
                rec(*a).checked_div(d).unwrap_or(0)
            }
        }
        Expr::Abs(a) => rec(*a).saturating_abs(),
        Expr::Lt(a, b) => i64::from(rec(*a) < rec(*b)),
        Expr::Le(a, b) => i64::from(rec(*a) <= rec(*b)),
        Expr::Eq(a, b) => i64::from(rec(*a) == rec(*b)),
        Expr::Ne(a, b) => i64::from(rec(*a) != rec(*b)),
        Expr::IfThenElse(c, a, b) => {
            if rec(*c) != 0 {
                rec(*a)
            } else {
                rec(*b)
            }
        }
    }
}

/// Whether an expression subtree references any binder beyond `Arg(0)` (the
/// current item). Used to confirm a per-item (`Items`) body reads only the
/// current item before it is reinterpreted as an identity-accumulator scan emit.
fn reads_non_item_arg(arena: &[Expr], id: ExprId) -> bool {
    match &arena[id.0 as usize] {
        Expr::Const(_) => false,
        Expr::Arg(k) => *k >= 1,
        Expr::Array(_, i) | Expr::Abs(i) => reads_non_item_arg(arena, *i),
        Expr::Matrix(_, i, j)
        | Expr::Add(i, j)
        | Expr::Sub(i, j)
        | Expr::Mul(i, j)
        | Expr::Min(i, j)
        | Expr::Max(i, j)
        | Expr::Div(i, j)
        | Expr::Lt(i, j)
        | Expr::Le(i, j)
        | Expr::Eq(i, j)
        | Expr::Ne(i, j) => reads_non_item_arg(arena, *i) || reads_non_item_arg(arena, *j),
        Expr::IfThenElse(c, a, b) => reads_non_item_arg(arena, *c) || reads_non_item_arg(arena, *a) || reads_non_item_arg(arena, *b),
    }
}

/// Whether an expression subtree references the accumulator binder `Arg(1)`.
fn uses_acc(arena: &[Expr], id: ExprId) -> bool {
    match &arena[id.0 as usize] {
        Expr::Const(_) => false,
        Expr::Arg(k) => *k == 1,
        Expr::Array(_, i) | Expr::Abs(i) => uses_acc(arena, *i),
        Expr::Matrix(_, i, j)
        | Expr::Add(i, j)
        | Expr::Sub(i, j)
        | Expr::Mul(i, j)
        | Expr::Min(i, j)
        | Expr::Max(i, j)
        | Expr::Div(i, j)
        | Expr::Lt(i, j)
        | Expr::Le(i, j)
        | Expr::Eq(i, j)
        | Expr::Ne(i, j) => uses_acc(arena, *i) || uses_acc(arena, *j),
        Expr::IfThenElse(c, a, b) => uses_acc(arena, *c) || uses_acc(arena, *a) || uses_acc(arena, *b),
    }
}

fn within_i32_half((lo, hi): (i64, i64)) -> bool {
    lo <= hi && lo >= I32_HALF_LO && hi <= I32_HALF_HI
}

fn clamp(x: i128) -> i64 {
    x.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}
