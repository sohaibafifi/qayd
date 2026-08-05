use super::external::call_external_function;
use super::ir::{CollectionModel, Expr, ExprId, GlobalConstraint, Iterable, Reduction, Resource};
use std::sync::atomic::{AtomicBool, Ordering};

const VALIDATION_STOPPED: &str = "qayd validation stopped by solve budget";

fn check_stop(stop: Option<&AtomicBool>) -> Result<(), String> {
    if stop.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        Err(VALIDATION_STOPPED.to_string())
    } else {
        Ok(())
    }
}

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
        Expr::Mod(a, b) => eval_expr_checked(arena, *a, args)?.checked_rem(eval_expr_checked(arena, *b, args)?).unwrap_or(0),
        Expr::Pow(base, exponent) => eval_expr_checked(arena, *base, args)?
            .checked_pow(*exponent)
            .ok_or_else(|| "integer overflow in a lambda body (pow)".to_string())?,
        Expr::MulScaled(a, b, scale) => {
            checked_scaled(eval_expr_checked(arena, *a, args)?, eval_expr_checked(arena, *b, args)?, *scale, false)?
        }
        Expr::DivScaled(a, b, scale) => {
            checked_scaled(eval_expr_checked(arena, *a, args)?, eval_expr_checked(arena, *b, args)?, *scale, true)?
        }
        Expr::Min(a, b) => eval_expr_checked(arena, *a, args)?.min(eval_expr_checked(arena, *b, args)?),
        Expr::Max(a, b) => eval_expr_checked(arena, *a, args)?.max(eval_expr_checked(arena, *b, args)?),
        Expr::Div(a, b) => {
            let num = eval_expr_checked(arena, *a, args)?;
            let den = eval_expr_checked(arena, *b, args)?;
            if den == 0 {
                0
            } else {
                num.checked_div(den).ok_or_else(|| "integer overflow in a lambda body (/)".to_string())?
            }
        }
        Expr::Abs(a) => eval_expr_checked(arena, *a, args)?.saturating_abs(),
        Expr::Lt(a, b) => i64::from(eval_expr_checked(arena, *a, args)? < eval_expr_checked(arena, *b, args)?),
        Expr::Le(a, b) => i64::from(eval_expr_checked(arena, *a, args)? <= eval_expr_checked(arena, *b, args)?),
        Expr::Eq(a, b) => i64::from(eval_expr_checked(arena, *a, args)? == eval_expr_checked(arena, *b, args)?),
        Expr::Ne(a, b) => i64::from(eval_expr_checked(arena, *a, args)? != eval_expr_checked(arena, *b, args)?),
        Expr::IfThenElse(c, a, b) => {
            // Lazy, matching solve semantics: evaluate only the selected branch,
            // so a guarded out-of-range index in the branch the condition never
            // picks is accepted here exactly as it is at solve time.
            if eval_expr_checked(arena, *c, args)? != 0 {
                eval_expr_checked(arena, *a, args)?
            } else {
                eval_expr_checked(arena, *b, args)?
            }
        }
        Expr::PiecewiseLinear { input, points } => checked_piecewise(eval_expr_checked(arena, *input, args)?, points)?,
        Expr::External { name, args: external_args } => {
            let values = external_args.iter().map(|id| eval_expr_checked(arena, *id, args)).collect::<Result<Vec<_>, _>>()?;
            call_external_function(name, &values)?
        }
    })
}

fn checked_round_ratio(numerator: i128, denominator: i128) -> i128 {
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

fn checked_scaled(a: i64, b: i64, scale: i64, divide: bool) -> Result<i64, String> {
    if scale <= 0 {
        return Err("fixed-point scale must be positive".to_string());
    }
    if divide && b == 0 {
        return Ok(0);
    }
    let (numerator, denominator) =
        if divide { (i128::from(a) * i128::from(scale), i128::from(b)) } else { (i128::from(a) * i128::from(b), i128::from(scale)) };
    i64::try_from(checked_round_ratio(numerator, denominator)).map_err(|_| "fixed-point result overflows i64".to_string())
}

fn checked_piecewise(input: i64, points: &[(i64, i64)]) -> Result<i64, String> {
    if points.is_empty() || points.windows(2).any(|window| window[0].0 >= window[1].0) {
        return Err("piecewise-linear knots must be non-empty with strictly increasing x coordinates".to_string());
    }
    if input <= points[0].0 {
        return Ok(points[0].1);
    }
    for window in points.windows(2) {
        let (x0, y0) = window[0];
        let (x1, y1) = window[1];
        if input <= x1 {
            let numerator = (i128::from(input) - i128::from(x0)) * (i128::from(y1) - i128::from(y0));
            let value = i128::from(y0) + checked_round_ratio(numerator, i128::from(x1) - i128::from(x0));
            return i64::try_from(value).map_err(|_| "piecewise-linear result overflows i64".to_string());
        }
    }
    Ok(points.last().map_or(0, |&(_, y)| y))
}

/// Bitmask of which `Arg(k)` binders (k < 8) appear in a subtree.
fn args_used(arena: &[Expr], id: ExprId) -> u8 {
    match arena.get(id.0 as usize) {
        None => 0,
        Some(Expr::Const(_)) => 0,
        Some(Expr::Arg(k)) => 1u8.checked_shl(u32::from(*k)).unwrap_or(0),
        Some(Expr::Array(_, i)) => args_used(arena, *i),
        Some(Expr::Matrix(_, i, j)) => args_used(arena, *i) | args_used(arena, *j),
        Some(Expr::Abs(a) | Expr::Pow(a, _) | Expr::PiecewiseLinear { input: a, .. }) => args_used(arena, *a),
        Some(
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Mod(a, b)
            | Expr::MulScaled(a, b, _)
            | Expr::DivScaled(a, b, _)
            | Expr::Min(a, b)
            | Expr::Max(a, b)
            | Expr::Div(a, b)
            | Expr::Lt(a, b)
            | Expr::Le(a, b)
            | Expr::Eq(a, b)
            | Expr::Ne(a, b),
        ) => args_used(arena, *a) | args_used(arena, *b),
        Some(Expr::IfThenElse(c, a, b)) => args_used(arena, *c) | args_used(arena, *a) | args_used(arena, *b),
        Some(Expr::External { args, .. }) => args.iter().fold(0, |mask, &id| mask | args_used(arena, id)),
    }
}

/// Call `f` for every assignment of the `varying` arg slots over their domains
/// (other slots stay 0). Bounded recursion: arity is 1 or 2.
fn for_each_combo(
    domains: &[&[i64]],
    varying: &[usize],
    args: &mut [i64],
    depth: usize,
    stop: Option<&AtomicBool>,
    f: &mut dyn FnMut(&[i64]) -> Result<(), String>,
) -> Result<(), String> {
    check_stop(stop)?;
    if depth == varying.len() {
        return f(args);
    }
    let slot = varying[depth];
    for &val in domains[slot] {
        check_stop(stop)?;
        args[slot] = val;
        for_each_combo(domains, varying, args, depth + 1, stop, f)?;
    }
    Ok(())
}

/// Check that an index expression stays in `0..hi` for every value of the args
/// it uses. Only the used arg slots are enumerated, so a separable index like
/// `Arg(0)` is linear in the node count rather than quadratic.
fn check_index(arena: &[Expr], idx: ExprId, domains: &[&[i64]], hi: usize, what: &str, stop: Option<&AtomicBool>) -> Result<(), String> {
    let mask = args_used(arena, idx);
    let varying: Vec<usize> = (0..domains.len()).filter(|&s| mask & (1u8 << s) != 0).collect();
    let mut args = vec![0i64; domains.len()];
    for_each_combo(domains, &varying, &mut args, 0, stop, &mut |a| {
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
fn reject_arg1_index(arena: &[Expr], what: &str, stop: Option<&AtomicBool>) -> Result<(), String> {
    for node in arena {
        check_stop(stop)?;
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
fn validate_reduction(reduction: &Reduction, items: &[i32], lists: usize, stop: Option<&AtomicBool>) -> Result<(), String> {
    check_stop(stop)?;
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
    let scan_cur: Vec<i64>;
    let acc_dummy = [0i64];
    let domains: Vec<&[i64]> = match reduction.iterable {
        Iterable::Items(_) | Iterable::SetItems(_) => vec![&items_i],
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
        Iterable::Scan { boundary, step, end, .. } => {
            if step.0 as usize >= arena.len() {
                return Err("scan step refers to a missing expression".to_string());
            }
            // The accumulator (`Arg(1)`) has no finite domain, so it must never
            // index a table; reject that rather than read a silent 0.
            reject_arg1_index(arena, "scan accumulator", stop)?;
            let mut prev_nodes = items_i.clone();
            prev_nodes.push(i64::from(boundary));
            scan_nodes = prev_nodes;
            // Arg(0)=current ranges over items, plus the closing node `end` when the
            // scan folds the return edge; Arg(2)=previous ranges over items and the
            // boundary; the accumulator slot is a dummy (never an index, checked).
            let mut cur = items_i.clone();
            if let Some(e) = end {
                cur.push(i64::from(e));
            }
            scan_cur = cur;
            vec![&scan_cur, &acc_dummy, &scan_nodes]
        }
        Iterable::Windows { size, inner, .. } => {
            if size == 0 {
                return Err("window size must be positive".to_string());
            }
            if inner.0 as usize >= arena.len() {
                return Err("window inner expression is missing".to_string());
            }
            // The window total (`Arg(1)`) is unbounded; it must not index a table.
            reject_arg1_index(arena, "window total", stop)?;
            // Arg(0)=window item ranges over items; the total slot is a dummy.
            vec![&items_i, &acc_dummy]
        }
    };

    for node in arena {
        check_stop(stop)?;
        match node {
            Expr::MulScaled(_, _, scale) | Expr::DivScaled(_, _, scale) if *scale <= 0 => {
                return Err("fixed-point scale must be positive".to_string());
            }
            Expr::PiecewiseLinear { points, .. } if points.is_empty() || points.windows(2).any(|window| window[0].0 >= window[1].0) => {
                return Err("piecewise-linear knots must be non-empty with strictly increasing x coordinates".to_string());
            }
            Expr::External { name, .. } if !super::external_function_registered(name) => {
                return Err(format!("external function '{name}' is not registered"));
            }
            _ => {}
        }
    }

    // A conditional body is evaluated lazily at solve (only the selected branch
    // runs), so a guarded out-of-range index in an unselected branch is never
    // reached. Validate it the same way -- lazily, over every argument combination
    // -- instead of checking each index in isolation, which cannot see the guard.
    //
    // This is sound only when `body` is the single evaluated root and every binder
    // has a finite domain. `Scan`/`Windows` also evaluate `step`/`inner` and carry
    // an unbounded accumulator binder (`Arg(1)`), whose dummy domain would let lazy
    // evaluation miss an out-of-range access; they keep the conservative static
    // check below.
    let has_conditional = arena.iter().any(|node| matches!(node, Expr::IfThenElse(..)));
    let single_finite_root =
        matches!(reduction.iterable, Iterable::Items(_) | Iterable::SetItems(_) | Iterable::Edges { .. } | Iterable::Pairs(_));
    if has_conditional && single_finite_root {
        let mask = args_used(arena, reduction.body);
        let varying: Vec<usize> = (0..domains.len()).filter(|&s| mask & (1u8 << s) != 0).collect();
        let mut args = vec![0i64; domains.len()];
        for_each_combo(&domains, &varying, &mut args, 0, stop, &mut |a| {
            eval_expr_checked(arena, reduction.body, a)?;
            Ok(())
        })?;
        return Ok(());
    }

    for node in arena {
        check_stop(stop)?;
        match node {
            Expr::Array(arr, idx) => check_index(arena, *idx, &domains, arr.len(), "array", stop)?,
            Expr::Matrix(m, ri, ci) => {
                check_index(arena, *ri, &domains, m.len(), "matrix row", stop)?;
                let rectangular = m.iter().map(Vec::len).min() == m.iter().map(Vec::len).max();
                if rectangular {
                    let cols = m.first().map_or(0, Vec::len);
                    check_index(arena, *ci, &domains, cols, "matrix column", stop)?;
                } else {
                    // Ragged rows: the column bound depends on the row, so the
                    // two indices must be checked jointly.
                    let mask = args_used(arena, *ri) | args_used(arena, *ci);
                    let varying: Vec<usize> = (0..domains.len()).filter(|&s| mask & (1u8 << s) != 0).collect();
                    let mut args = vec![0i64; domains.len()];
                    for_each_combo(&domains, &varying, &mut args, 0, stop, &mut |a| {
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
            Expr::MulScaled(_, _, scale) | Expr::DivScaled(_, _, scale) if *scale <= 0 => {
                return Err("fixed-point scale must be positive".to_string());
            }
            Expr::PiecewiseLinear { points, .. } if points.is_empty() || points.windows(2).any(|window| window[0].0 >= window[1].0) => {
                return Err("piecewise-linear knots must be non-empty with strictly increasing x coordinates".to_string());
            }
            Expr::External { name, .. } if !super::external_function_registered(name) => {
                return Err(format!("external function '{name}' is not registered"));
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
        self.validate_impl(None)
    }

    /// Budget-aware validation. `Ok(false)` means the stop flag fired while
    /// validation was still in progress; malformed models remain ordinary
    /// errors. This lets frontends count validation against the same wall-clock
    /// budget as classification and search.
    pub fn validate_interruptible(&self, stop: &AtomicBool) -> Result<bool, String> {
        match self.validate_impl(Some(stop)) {
            Ok(()) => Ok(true),
            Err(error) if error == VALIDATION_STOPPED => Ok(false),
            Err(error) => Err(error),
        }
    }

    fn validate_impl(&self, stop: Option<&AtomicBool>) -> Result<(), String> {
        check_stop(stop)?;
        // A schedule model is a disjoint mode: validate the intervals only.
        if let Some(sched) = &self.schedule {
            let n = sched.intervals.len();
            if n == 0 {
                return Err("a schedule needs at least one interval".to_string());
            }
            for iv in &sched.intervals {
                check_stop(stop)?;
                if iv.modes.is_empty() {
                    if iv.duration < 0 || iv.duration > iv.horizon {
                        return Err("interval duration must be in 0..=horizon".to_string());
                    }
                } else {
                    if iv.optional {
                        return Err("a moded operation cannot also be optional; make its mode intervals optional instead".to_string());
                    }
                    for mode in &iv.modes {
                        if mode.duration < 0 || mode.duration > iv.horizon {
                            return Err("interval mode duration must be in 0..=horizon".to_string());
                        }
                    }
                }
            }
            let in_range = |i: usize| i < n;
            for &(a, b) in &sched.precedences {
                check_stop(stop)?;
                if !in_range(a) || !in_range(b) {
                    return Err("precedence references a missing interval".to_string());
                }
            }
            for res in &sched.resources {
                check_stop(stop)?;
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
        for tier in &self.objectives {
            for r in tier.reductions() {
                validate_reduction(r, &self.items, self.lists, stop)?;
            }
        }
        for c in &self.constraints {
            validate_reduction(&c.reduction, &self.items, self.lists, stop)?;
        }
        for g in &self.globals {
            check_stop(stop)?;
            for v in g.items() {
                if !self.items.contains(&v) {
                    return Err(format!("global constraint references item {v}, which is not in the universe"));
                }
            }
            match g {
                GlobalConstraint::AllSameList { items } | GlobalConstraint::AllDifferentLists { items } => {
                    if items.len() < 2 {
                        return Err("an all-same/all-different list group needs at least two items".to_string());
                    }
                    let mut seen = std::collections::HashSet::with_capacity(items.len());
                    if items.iter().any(|item| !seen.insert(*item)) {
                        return Err("an all-same/all-different list group cannot contain duplicate items".to_string());
                    }
                }
                GlobalConstraint::ListDistance { min, max, .. } if min > max => {
                    return Err("list-distance minimum cannot exceed its maximum".to_string());
                }
                _ => {}
            }
        }
        Ok(())
    }
}
