//! List variable declarations and list-objective evaluation.

mod ir;
pub(crate) mod scan;
mod validate;

pub use ir::*;

/// Reference to a list declaration inside [`Model`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ListVarRef(pub usize);

/// Structured list declaration.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ListDecl {
    /// Immutable item universe for this list.
    pub universe: Vec<i32>,
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
pub fn list_objective_tiers(tiers: &[ObjectiveTier], items: &[i32]) -> Option<Vec<ListObjectiveTier>> {
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
pub fn evaluate_list_objectives(tiers: &[ListObjectiveTier], lists: &[Vec<i32>]) -> Vec<i64> {
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
pub fn list_objectives_better(candidate: &[i64], incumbent: &[i64], tiers: &[ListObjectiveTier]) -> bool {
    for ((candidate, incumbent), tier) in candidate.iter().zip(incumbent).zip(tiers) {
        if candidate == incumbent {
            continue;
        }
        return if tier.minimize { candidate < incumbent } else { candidate > incumbent };
    }
    false
}

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
    let contents = lists.get(reduction.iterable.list())?;
    let arena = &reduction.arena.exprs;
    let body = reduction.body;
    let mut sum = 0i64;
    let mut count = 0i64;
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    let mut steps = 0u64;
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
                fold(eval_objective_expr(arena, body, &[i64::from(item)]));
            }
        }
        Iterable::Edges { start, end, .. } => {
            let n = contents.len();
            for p in 0..=n {
                let from = if p == 0 { start } else { contents[p - 1] };
                let to = if p == n { end } else { contents[p] };
                fold(eval_objective_expr(arena, body, &[i64::from(from), i64::from(to)]));
            }
        }
        Iterable::Pairs(_) => {
            let n = contents.len();
            for i in 0..n {
                for j in 0..n {
                    let args = [i64::from(contents[i]), i64::from(contents[j]), i as i64, j as i64];
                    fold(eval_objective_expr(arena, body, &args));
                }
            }
        }
        Iterable::Scan { init, boundary, step, .. } => {
            let mut acc = init;
            let mut prev = boundary;
            for &cur in contents {
                let new_acc = eval_objective_expr(arena, step, &[i64::from(cur), acc, i64::from(prev)]);
                fold(eval_objective_expr(arena, body, &[i64::from(cur), new_acc, i64::from(prev)]));
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
                        total = total.saturating_add(eval_objective_expr(arena, inner, &[i64::from(cur)]));
                    }
                    fold(eval_objective_expr(arena, body, &[0, total]));
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
                None
            } else {
                values.sort_unstable();
                Some(values[k])
            }
        }
    };
    value.map(|value| value.saturating_mul(reduction.coeff))
}

fn eval_objective_expr(arena: &[Expr], id: ExprId, args: &[i64]) -> i64 {
    match &arena[id.0 as usize] {
        Expr::Const(c) => *c,
        Expr::Arg(k) => args.get(*k as usize).copied().unwrap_or(0),
        Expr::Array(a, i) => {
            let idx = eval_objective_expr(arena, *i, args);
            usize::try_from(idx).ok().and_then(|u| a.get(u)).copied().unwrap_or(0)
        }
        Expr::Matrix(m, i, j) => {
            let ri = eval_objective_expr(arena, *i, args);
            let ci = eval_objective_expr(arena, *j, args);
            usize::try_from(ri)
                .ok()
                .and_then(|r| m.get(r))
                .and_then(|row| usize::try_from(ci).ok().and_then(|c| row.get(c)))
                .copied()
                .unwrap_or(0)
        }
        Expr::Add(a, b) => eval_objective_expr(arena, *a, args).saturating_add(eval_objective_expr(arena, *b, args)),
        Expr::Sub(a, b) => eval_objective_expr(arena, *a, args).saturating_sub(eval_objective_expr(arena, *b, args)),
        Expr::Mul(a, b) => eval_objective_expr(arena, *a, args).saturating_mul(eval_objective_expr(arena, *b, args)),
        Expr::Min(a, b) => eval_objective_expr(arena, *a, args).min(eval_objective_expr(arena, *b, args)),
        Expr::Max(a, b) => eval_objective_expr(arena, *a, args).max(eval_objective_expr(arena, *b, args)),
        Expr::Div(a, b) => {
            let d = eval_objective_expr(arena, *b, args);
            if d == 0 {
                0
            } else {
                eval_objective_expr(arena, *a, args).checked_div(d).unwrap_or(0)
            }
        }
        Expr::Abs(a) => eval_objective_expr(arena, *a, args).saturating_abs(),
        Expr::Lt(a, b) => i64::from(eval_objective_expr(arena, *a, args) < eval_objective_expr(arena, *b, args)),
        Expr::Le(a, b) => i64::from(eval_objective_expr(arena, *a, args) <= eval_objective_expr(arena, *b, args)),
        Expr::Eq(a, b) => i64::from(eval_objective_expr(arena, *a, args) == eval_objective_expr(arena, *b, args)),
        Expr::Ne(a, b) => i64::from(eval_objective_expr(arena, *a, args) != eval_objective_expr(arena, *b, args)),
        Expr::IfThenElse(c, a, b) => {
            if eval_objective_expr(arena, *c, args) != 0 {
                eval_objective_expr(arena, *a, args)
            } else {
                eval_objective_expr(arena, *b, args)
            }
        }
    }
}

fn list_item_values(reduction: &Reduction, items: &[i32]) -> Option<Vec<(i32, i64)>> {
    let mut values = Vec::with_capacity(items.len());
    for &item in items {
        values.push((item, super::classify::eval_collection_expr_one(&reduction.arena.exprs, reduction.body, i64::from(item))?));
    }
    Some(values)
}

fn list_item_weight(weights: &[(i32, i64)], item: i32) -> i64 {
    debug_assert!(weights.iter().any(|(known, _)| *known == item));
    weights.iter().find_map(|(known, weight)| (*known == item).then_some(*weight)).unwrap_or(0)
}
