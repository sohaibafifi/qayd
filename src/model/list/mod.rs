//! List variable declarations and list-objective evaluation.

mod ir;
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
    Sum { list: usize, weights: Vec<(i32, i64)> },
    /// Count of items whose weight/predicate value is non-zero.
    Count { list: usize, weights: Vec<(i32, i64)> },
    /// `1` when the list is non-empty, else `0`.
    Used { list: usize },
}

/// One domain exact lexicographic objective tier.
#[derive(Clone)]
pub struct ListObjectiveTier {
    /// Whether this tier is minimized.
    pub minimize: bool,
    /// Terms summed to form the tier value.
    pub terms: Vec<ListObjectiveTerm>,
}

/// Parse list objective tiers that domain exact search can evaluate.
pub fn list_objective_tiers(tiers: &[ObjectiveTier], items: &[i32]) -> Option<Vec<ListObjectiveTier>> {
    let mut parsed = Vec::with_capacity(tiers.len());
    for tier in tiers {
        let mut terms = Vec::with_capacity(tier.terms.len());
        for reduction in &tier.terms {
            terms.push(list_objective_term(reduction, items)?);
        }
        parsed.push(ListObjectiveTier { minimize: tier.minimize, terms });
    }
    Some(parsed)
}

/// Evaluate parsed list objective tiers on concrete list contents.
pub fn evaluate_list_objectives(tiers: &[ListObjectiveTier], lists: &[Vec<i32>]) -> Vec<i64> {
    tiers
        .iter()
        .map(|tier| {
            tier.terms.iter().fold(0i64, |acc, term| {
                acc.saturating_add(match term {
                    ListObjectiveTerm::Sum { list, weights } => lists
                        .get(*list)
                        .map_or(0, |items| items.iter().fold(0i64, |sum, item| sum.saturating_add(list_item_weight(weights, *item)))),
                    ListObjectiveTerm::Count { list, weights } => lists.get(*list).map_or(0, |items| {
                        items.iter().filter(|item| list_item_weight(weights, **item) != 0).fold(0i64, |count, _| count.saturating_add(1))
                    }),
                    ListObjectiveTerm::Used { list } => i64::from(lists.get(*list).is_some_and(|items| !items.is_empty())),
                })
            })
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
    let Iterable::Items(list) = &reduction.iterable else {
        return None;
    };
    match reduction.op {
        ReduceOp::Sum => Some(ListObjectiveTerm::Sum { list: *list, weights: list_item_values(reduction, items)? }),
        ReduceOp::Count => Some(ListObjectiveTerm::Count { list: *list, weights: list_item_values(reduction, items)? }),
        ReduceOp::Used => Some(ListObjectiveTerm::Used { list: *list }),
        ReduceOp::Min | ReduceOp::Max | ReduceOp::SelectKth(_) => None,
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
