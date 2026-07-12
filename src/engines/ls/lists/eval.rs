use crate::model::list::{Expr, ExprId, Iterable, Op, ReduceOp, Reduction};

/// Violation weight for an undefined reduction (empty `min`/`max`) or a broken
/// constraint operand, large enough to dominate any real objective.
pub(super) const INFEASIBLE: i64 = 1_000_000_000;

pub(super) fn eval_expr(arena: &[Expr], id: ExprId, args: &[i64]) -> i64 {
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
            if d == 0 {
                0
            } else {
                eval_expr(arena, *a, args).checked_div(d).unwrap_or(0)
            }
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
pub(super) fn eval_reduction(reduction: &Reduction, contents: &[i32]) -> Option<i64> {
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
        Iterable::Scan { init, boundary, step, end, .. } => {
            let mut acc = init;
            let mut prev = boundary;
            for &cur in contents {
                let new_acc = eval_expr(arena, step, &[i64::from(cur), acc, i64::from(prev)]);
                fold(eval_expr(arena, body, &[i64::from(cur), new_acc, i64::from(prev)]));
                acc = new_acc;
                prev = cur;
            }
            // Closing edge to `end`: one more transition `(cur=end, prev=last item
            // or boundary)`, mirroring how `Edges` appends `(last_item, end)`.
            if let Some(end) = end {
                let new_acc = eval_expr(arena, step, &[i64::from(end), acc, i64::from(prev)]);
                fold(eval_expr(arena, body, &[i64::from(end), new_acc, i64::from(prev)]));
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

    let value = match reduction.op {
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
    };
    value.map(|value| value.saturating_mul(reduction.coeff))
}

pub(super) fn violation_of(value: i64, op: Op, rhs: i64) -> i64 {
    match op {
        Op::Le => value.saturating_sub(rhs).max(0),
        Op::Ge => rhs.saturating_sub(value).max(0),
        Op::Eq => value.saturating_sub(rhs).saturating_abs(),
    }
}
