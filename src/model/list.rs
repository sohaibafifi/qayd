//! List variable declarations and structured list-objective evaluation.

use crate::collection;

/// Reference to a structured list declaration inside [`Model`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ListVarRef(pub usize);

/// Structured list declaration.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ListDecl {
    /// Immutable item universe for this list.
    pub universe: Vec<i32>,
}

/// Current list reduction representation during the migration.
pub type ListReduction = collection::Reduction;

/// Current list constraint representation during the migration.
pub type ListReductionConstraint = collection::Constraint;

/// Current list reduction operator representation during the migration.
pub type ListReduceOp = collection::ReduceOp;

/// Current list iterable representation during the migration.
pub type ListIterable = collection::Iterable;

/// Structured exact objective term over one list.
#[derive(Clone)]
pub enum StructuredListObjectiveTerm {
    /// Sum of item weights in a list.
    Sum { list: usize, weights: Vec<(i32, i64)> },
    /// Count of items whose weight/predicate value is non-zero.
    Count { list: usize, weights: Vec<(i32, i64)> },
    /// `1` when the list is non-empty, else `0`.
    Used { list: usize },
}

/// One structured exact lexicographic objective tier.
#[derive(Clone)]
pub struct StructuredListObjectiveTier {
    /// Whether this tier is minimized.
    pub minimize: bool,
    /// Terms summed to form the tier value.
    pub terms: Vec<StructuredListObjectiveTerm>,
}

/// Parse list objective tiers that structured exact search can evaluate.
pub fn structured_list_objective_tiers(tiers: &[collection::ObjectiveTier], items: &[i32]) -> Option<Vec<StructuredListObjectiveTier>> {
    let mut parsed = Vec::with_capacity(tiers.len());
    for tier in tiers {
        let mut terms = Vec::with_capacity(tier.terms.len());
        for reduction in &tier.terms {
            terms.push(structured_list_objective_term(reduction, items)?);
        }
        parsed.push(StructuredListObjectiveTier { minimize: tier.minimize, terms });
    }
    Some(parsed)
}

/// Evaluate parsed structured objective tiers on concrete list contents.
pub fn evaluate_structured_list_objectives(tiers: &[StructuredListObjectiveTier], lists: &[Vec<i32>]) -> Vec<i64> {
    tiers
        .iter()
        .map(|tier| {
            tier.terms.iter().fold(0i64, |acc, term| {
                acc.saturating_add(match term {
                    StructuredListObjectiveTerm::Sum { list, weights } => lists
                        .get(*list)
                        .map_or(0, |items| items.iter().fold(0i64, |sum, item| sum.saturating_add(structured_weight(weights, *item)))),
                    StructuredListObjectiveTerm::Count { list, weights } => lists.get(*list).map_or(0, |items| {
                        items.iter().filter(|item| structured_weight(weights, **item) != 0).fold(0i64, |count, _| count.saturating_add(1))
                    }),
                    StructuredListObjectiveTerm::Used { list } => i64::from(lists.get(*list).is_some_and(|items| !items.is_empty())),
                })
            })
        })
        .collect()
}

/// Lexicographic comparison of structured objective vectors.
pub fn structured_list_objectives_better(candidate: &[i64], incumbent: &[i64], tiers: &[StructuredListObjectiveTier]) -> bool {
    for ((candidate, incumbent), tier) in candidate.iter().zip(incumbent).zip(tiers) {
        if candidate == incumbent {
            continue;
        }
        return if tier.minimize { candidate < incumbent } else { candidate > incumbent };
    }
    false
}

fn structured_list_objective_term(reduction: &collection::Reduction, items: &[i32]) -> Option<StructuredListObjectiveTerm> {
    let collection::Iterable::Items(list) = &reduction.iterable else {
        return None;
    };
    match reduction.op {
        collection::ReduceOp::Sum => {
            Some(StructuredListObjectiveTerm::Sum { list: *list, weights: structured_item_values(reduction, items)? })
        }
        collection::ReduceOp::Count => {
            Some(StructuredListObjectiveTerm::Count { list: *list, weights: structured_item_values(reduction, items)? })
        }
        collection::ReduceOp::Used => Some(StructuredListObjectiveTerm::Used { list: *list }),
        collection::ReduceOp::Min | collection::ReduceOp::Max | collection::ReduceOp::SelectKth(_) => None,
    }
}

fn structured_item_values(reduction: &collection::Reduction, items: &[i32]) -> Option<Vec<(i32, i64)>> {
    let mut values = Vec::with_capacity(items.len());
    for &item in items {
        values.push((item, super::classify::eval_collection_expr_one(&reduction.arena.exprs, reduction.body, i64::from(item))?));
    }
    Some(values)
}

fn structured_weight(weights: &[(i32, i64)], item: i32) -> i64 {
    debug_assert!(weights.iter().any(|(known, _)| *known == item));
    weights.iter().find_map(|(known, weight)| (*known == item).then_some(*weight)).unwrap_or(0)
}
