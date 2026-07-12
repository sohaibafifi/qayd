use super::ir::{CollectionModel, Expr, ExprId, Iterable, Reduction, Resource, MAX_TIERS};

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
fn for_each_combo(
    domains: &[&[i64]],
    varying: &[usize],
    args: &mut [i64],
    depth: usize,
    f: &mut dyn FnMut(&[i64]) -> Result<(), String>,
) -> Result<(), String> {
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
    let scan_cur: Vec<i64>;
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
        Iterable::Scan { boundary, step, end, .. } => {
            if step.0 as usize >= arena.len() {
                return Err("scan step refers to a missing expression".to_string());
            }
            // The accumulator (`Arg(1)`) has no finite domain, so it must never
            // index a table; reject that rather than read a silent 0.
            reject_arg1_index(arena, "scan accumulator")?;
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
            reject_arg1_index(arena, "window total")?;
            // Arg(0)=window item ranges over items; the total slot is a dummy.
            vec![&items_i, &acc_dummy]
        }
    };

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
    let single_finite_root = matches!(reduction.iterable, Iterable::Items(_) | Iterable::Edges { .. } | Iterable::Pairs(_));
    if has_conditional && single_finite_root {
        let mask = args_used(arena, reduction.body);
        let varying: Vec<usize> = (0..domains.len()).filter(|&s| mask & (1u8 << s) != 0).collect();
        let mut args = vec![0i64; domains.len()];
        for_each_combo(&domains, &varying, &mut args, 0, &mut |a| {
            eval_expr_checked(arena, reduction.body, a)?;
            Ok(())
        })?;
        return Ok(());
    }

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
            for r in tier.reductions() {
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
