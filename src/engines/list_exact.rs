//! Structured list exact-search backend.
//!
//! Lowers a list-reduction model to list domains, posts the supported
//! constraints, and selects one of two exact backends. Returns `Ok(None)` when
//! the model shape is not supported at all.
//!
//! # The two backends
//!
//! Both are built from the *same* solver (`build_solver`); selection only
//! changes how it is searched.
//!
//! - **member-var** (`run_member_var_search`) - branches the `item in list`
//!   membership variables under the learning (CDCL) engine. A single linearizable
//!   objective tier is integer-backed and branch-and-bounded (`optimize_linear_tier`),
//!   so it prunes on an objective bound instead of enumerating every solution.
//! - **chrono** (`run_domain_search`) - the non-learning chronological domain
//!   DFS, scoring each complete assignment externally. The fallback for everything
//!   the member-var backend does not (yet) own.
//!
//! # Integer-backing and explanations
//!
//! A list fact is routable to the member-var backend only once it is both an
//! integer variable (so CDCL can branch/learn on it) *and* explained by a
//! propagator (so a learned clause is sound). Done so far, each guarded by a
//! CDCL-vs-chronological oracle proving no over-pruning:
//!
//! | fact / constraint   | backing                          | explained by            |
//! |---------------------|----------------------------------|-------------------------|
//! | `item in list`      | one bool var per (list, item)    | `partition`, `same_list`|
//! | list length         | one length var per list          | `ListCardinality`       |
//! | `Length` bound      | tightens the length var          | `ListLength`            |
//! | weighted item sum   | linear over member vars          | `ListItemSum`           |
//! | list-index order    | membership vars                  | `ItemPrecedence`        |
//! | `used = len >= 1`   | one bool var per list            | `ListUsed`              |
//! | objective (1 tier)  | `Objective::Linear` over the above | branch-and-bound        |
//!
//! # Routing predicate (`member_var_exact_supported`)
//!
//! Explained is necessary but **not sufficient** to route: routing is a
//! performance decision. Currently routed to member-var: partition + `Length` +
//! `same_list` (+ a single linearizable objective tier). Everything else stays on
//! chrono:
//!
//! - **`item_sum` → chrono.** Explained and oracle-checked, and the objective is
//!   integer-backed, but on `ItemSum`-heavy COPs (bin packing) the member-var
//!   *search* still trails chrono: the objective bound alone does not compensate
//!   for the lack of a packing-aware branching heuristic (bin_packing went
//!   SATISFIABLE → UNKNOWN when routed). Soundness is fine; search quality is not.
//! - **`item_precedence` → chrono.** Explained and oracle-checked, but routing it
//!   is an unvalidated perf decision, so it is deliberately not routed yet.
//! - **edge / routing → chrono.** Not integer-backed at all (see below).
//!
//! # What is missing to route routing / edge-sum
//!
//! These need the list *order* domain, which does not exist yet: per-list item
//! positions and successor arcs as integer variables, explained order/arc
//! propagators over them, edge-cost lowering onto that order, and almost certainly
//! a dedicated branching. That is the list-ordering domain, a separate tranche -
//! not "one more constraint" over membership.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::constraints::list as list_constraints;
use crate::ids::{ListId, VarId};
use crate::model::list::{self, CollectionModel, ListObjectiveMaxTerm, ListObjectiveTerm};
use crate::model::{evaluate_list_objectives, list_objectives_better, ListObjectiveTier};
use crate::search::{self, Objective as SearchObjective, SearchControl, SolveStats};
use crate::store::Solver;

use super::{CollectionCompileContext, CompileFailure};

/// Solver status returned by the list exact engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Optimal,
    Satisfiable,
    Unsatisfiable,
    Unknown,
}

impl Status {
    /// Stable status text for frontends.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Optimal => "OPTIMAL",
            Self::Satisfiable => "SATISFIABLE",
            Self::Unsatisfiable => "UNSATISFIABLE",
            Self::Unknown => "UNKNOWN",
        }
    }
}

/// Structured list exact-search result.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub status: Status,
    /// Best objective value per lexicographic tier (empty for a pure CSP).
    pub objectives: Vec<i64>,
    /// Required item contents of each list in the best solution.
    pub solution: Option<Vec<Vec<i32>>>,
    pub stats: SolveStats,
}

fn interrupted_outcome(stats: SolveStats) -> Outcome {
    Outcome { status: Status::Unknown, objectives: Vec::new(), solution: None, stats }
}

fn stop_requested(stop: &AtomicBool) -> bool {
    stop.load(Ordering::Acquire)
}

/// List model already parsed into the exact engine's physical constraint and
/// objective representation. A successful compilation cannot be declined by
/// execution later.
#[derive(Clone)]
pub(crate) struct CompiledListExact {
    objective_tiers: Vec<ListObjectiveTier>,
    constraints: Vec<ListConstraint>,
}

#[derive(Clone)]
enum ListConstraint {
    Length {
        list: usize,
        min: usize,
        max: usize,
    },
    ItemSum {
        list: usize,
        weights: Vec<(i32, i64)>,
        min: i64,
        max: i64,
    },
    /// A sequence reduction (scan) checked by evaluating it on the completed
    /// ordered lists during enumeration, rather than posted into the solver.
    Reduction {
        reduction: list::Reduction,
        op: list::Op,
        rhs: i64,
    },
    Impossible,
}

fn list_constraint(constraint: &list::Constraint, items: &[i32], stop: Option<&AtomicBool>) -> Option<ListConstraint> {
    if stop.is_some_and(|stop| stop.load(std::sync::atomic::Ordering::Acquire)) {
        return None;
    }
    let list::Iterable::Items(list) = &constraint.reduction.iterable else {
        // Order-sensitive and set reductions have no linearizable posting here;
        // they are checked on completed assignments.
        if matches!(constraint.reduction.iterable, list::Iterable::Scan { .. } | list::Iterable::SetItems(_)) {
            return Some(ListConstraint::Reduction { reduction: constraint.reduction.clone(), op: constraint.op, rhs: constraint.rhs });
        }
        return None;
    };
    let list = *list;

    if matches!(constraint.reduction.op, list::ReduceOp::Used) {
        if constraint.reduction.coeff != 1 {
            return None;
        }
        return used_constraint_as_length(list, constraint.op, constraint.rhs, items.len());
    }

    if matches!(constraint.reduction.op, list::ReduceOp::Count) {
        if constraint.reduction.coeff != 1 {
            return None;
        }
        let body = constraint.reduction.arena.exprs.get(constraint.reduction.body.0 as usize)?;
        let list::Expr::Const(value) = body else {
            return None;
        };
        if *value == 0 {
            return None;
        }

        let n = items.len() as i64;
        let rhs = constraint.rhs;
        let (min, max) = match constraint.op {
            list::Op::Le => (0, rhs),
            list::Op::Ge => (rhs, n),
            list::Op::Eq => (rhs, rhs),
        };
        if max < 0 || min > n {
            return Some(ListConstraint::Impossible);
        }

        let min = min.max(0) as usize;
        let max = max.min(n) as usize;
        return if min > max { Some(ListConstraint::Impossible) } else { Some(ListConstraint::Length { list, min, max }) };
    }

    if matches!(constraint.reduction.op, list::ReduceOp::Sum) {
        let mut weights = Vec::with_capacity(items.len());
        for &item in items {
            if stop.is_some_and(|stop| stop.load(std::sync::atomic::Ordering::Acquire)) {
                return None;
            }
            weights.push((
                item,
                eval_collection_expr_one(&constraint.reduction.arena.exprs, constraint.reduction.body, i64::from(item))?
                    .saturating_mul(constraint.reduction.coeff),
            ));
        }
        let (min, max) = match constraint.op {
            list::Op::Le => (i64::MIN / 4, constraint.rhs),
            list::Op::Ge => (constraint.rhs, i64::MAX / 4),
            list::Op::Eq => (constraint.rhs, constraint.rhs),
        };
        return Some(ListConstraint::ItemSum { list, weights, min, max });
    }

    None
}

fn used_constraint_as_length(list: usize, op: list::Op, rhs: i64, item_count: usize) -> Option<ListConstraint> {
    let (min, max) = match op {
        list::Op::Le if rhs < 0 => return Some(ListConstraint::Impossible),
        list::Op::Le if rhs == 0 => (0, 0),
        list::Op::Le => (0, item_count),
        list::Op::Ge if rhs <= 0 => (0, item_count),
        list::Op::Ge if rhs == 1 => (1, item_count),
        list::Op::Ge => return Some(ListConstraint::Impossible),
        list::Op::Eq if rhs == 0 => (0, 0),
        list::Op::Eq if rhs == 1 => (1, item_count),
        list::Op::Eq => return Some(ListConstraint::Impossible),
    };
    Some(ListConstraint::Length { list, min, max })
}

fn eval_collection_expr_one(arena: &[list::Expr], id: list::ExprId, arg0: i64) -> Option<i64> {
    list::eval_expr_checked(arena, id, &[arg0])
}

/// Solve a supported list model. `objective_tiers` is parsed by the
/// caller (it also needs the objective sense); `on_improve` receives each
/// improving tier vector. Returns `Ok(None)` when the model is not supported.
pub fn solve(
    model: &CollectionModel,
    objective_tiers: &[ListObjectiveTier],
    stop: &AtomicBool,
    mut on_improve: impl FnMut(&[i64]),
) -> Result<Option<Outcome>, String> {
    if stop_requested(stop) {
        return Ok(Some(interrupted_outcome(SolveStats::default())));
    }
    let Some(parsed) = parse_constraints_interruptible(model, stop) else {
        return Ok(stop_requested(stop).then(|| interrupted_outcome(SolveStats::default())));
    };
    Ok(Some(solve_parsed(model, objective_tiers, &parsed, stop, &mut on_improve)))
}

/// Compile the model and its objective tiers in one capability operation.
pub(crate) fn compile(context: &CollectionCompileContext<'_>, stop: &AtomicBool) -> Result<CompiledListExact, CompileFailure> {
    #[cfg(test)]
    COMPILE_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));
    let model = context.physical();
    if stop_requested(stop) {
        return Err(CompileFailure::Interrupted { phase: "before list exact capability analysis" });
    }
    debug_assert!(context.semantic().intervals().is_empty());
    let should_stop = || stop_requested(stop);
    let Some(objective_tiers) = list_objective_tiers_interruptible(&model.objectives, &model.items, &should_stop) else {
        return if should_stop() {
            Err(CompileFailure::Interrupted { phase: "during list exact objective lowering" })
        } else {
            Err(CompileFailure::Unsupported { code: "list-objective", detail: "list objective is outside the exact list backend" })
        };
    };
    if should_stop() {
        return Err(CompileFailure::Interrupted { phase: "after list exact objective lowering" });
    }
    let Some(constraints) = parse_constraints_interruptible(model, stop) else {
        return if should_stop() {
            Err(CompileFailure::Interrupted { phase: "during list exact constraint lowering" })
        } else {
            Err(CompileFailure::Unsupported { code: "list-constraint", detail: "list constraint is outside the exact list backend" })
        };
    };
    if should_stop() {
        return Err(CompileFailure::Interrupted { phase: "after list exact constraint lowering" });
    }
    if should_stop() {
        return Err(CompileFailure::Interrupted { phase: "during list exact physical model construction" });
    }
    Ok(CompiledListExact { objective_tiers, constraints })
}

#[cfg(test)]
pub(crate) fn lower(context: &CollectionCompileContext<'_>, stop: &AtomicBool) -> Option<CompiledListExact> {
    compile(context, stop).ok()
}

#[cfg(test)]
std::thread_local! {
    static COMPILE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_compile_calls_for_test() {
    COMPILE_CALLS.with(|calls| calls.set(0));
}

#[cfg(test)]
pub(crate) fn compile_calls_for_test() -> usize {
    COMPILE_CALLS.with(std::cell::Cell::get)
}

fn list_objective_tiers_interruptible(
    tiers: &[list::ObjectiveTier],
    items: &[i32],
    should_stop: &dyn Fn() -> bool,
) -> Option<Vec<ListObjectiveTier>> {
    if should_stop() {
        return None;
    }
    let mut parsed = Vec::with_capacity(tiers.len());
    for tier in tiers {
        if should_stop() {
            return None;
        }
        let mut terms = Vec::with_capacity(tier.terms.len());
        for reduction in &tier.terms {
            terms.push(list_objective_term_interruptible(reduction, items, should_stop)?);
        }

        let max_terms = if let Some(terms) = &tier.max_terms {
            let mut parsed_terms = Vec::with_capacity(terms.len());
            for term in terms {
                if should_stop() {
                    return None;
                }
                let mut groups = Vec::with_capacity(term.groups.len());
                for group in &term.groups {
                    let mut parsed_group = Vec::with_capacity(group.len());
                    for reduction in group {
                        parsed_group.push(list_objective_term_interruptible(reduction, items, should_stop)?);
                    }
                    groups.push(parsed_group);
                }
                parsed_terms.push(ListObjectiveMaxTerm { groups, coeff: term.coeff });
            }
            Some(parsed_terms)
        } else {
            None
        };
        parsed.push(ListObjectiveTier { minimize: tier.minimize, terms, max_terms });
    }
    (!should_stop()).then_some(parsed)
}

fn list_objective_term_interruptible(
    reduction: &list::Reduction,
    items: &[i32],
    should_stop: &dyn Fn() -> bool,
) -> Option<ListObjectiveTerm> {
    if should_stop() {
        return None;
    }
    if let list::Iterable::Items(list) = &reduction.iterable {
        match reduction.op {
            list::ReduceOp::Sum | list::ReduceOp::Count => {
                let mut weights = Vec::with_capacity(items.len());
                for &item in items {
                    if should_stop() {
                        return None;
                    }
                    let value = list::eval_expr_checked(&reduction.arena.exprs, reduction.body, &[i64::from(item)])?;
                    weights.push((item, value));
                }
                return match reduction.op {
                    list::ReduceOp::Sum => Some(ListObjectiveTerm::Sum { list: *list, weights, coeff: reduction.coeff }),
                    list::ReduceOp::Count => Some(ListObjectiveTerm::Count { list: *list, weights, coeff: reduction.coeff }),
                    _ => unreachable!("sum/count branch was checked above"),
                };
            }
            list::ReduceOp::Used => return Some(ListObjectiveTerm::Used { list: *list, coeff: reduction.coeff }),
            list::ReduceOp::Min | list::ReduceOp::Max | list::ReduceOp::SelectKth(_) => {}
        }
    }
    if should_stop() {
        return None;
    }
    let term = ListObjectiveTerm::Reduction(reduction.clone());
    (!should_stop()).then_some(term)
}

pub(crate) fn solve_compiled(
    compiled: &CompiledListExact,
    model: &CollectionModel,
    stop: &AtomicBool,
    mut on_improve: impl FnMut(&[i64]),
) -> Outcome {
    solve_parsed(model, &compiled.objective_tiers, &compiled.constraints, stop, &mut on_improve)
}

#[cfg(test)]
pub(crate) fn solve_compiled_with_build_stop_for_test(
    compiled: &CompiledListExact,
    model: &CollectionModel,
    stop: &AtomicBool,
    should_stop: &dyn Fn() -> bool,
) -> Outcome {
    solve_parsed_until(model, &compiled.objective_tiers, &compiled.constraints, stop, should_stop, &mut |_| {})
}

fn solve_parsed(
    model: &CollectionModel,
    objective_tiers: &[ListObjectiveTier],
    parsed: &[ListConstraint],
    stop: &AtomicBool,
    on_improve: &mut impl FnMut(&[i64]),
) -> Outcome {
    let should_stop = || stop_requested(stop);
    solve_parsed_until(model, objective_tiers, parsed, stop, &should_stop, on_improve)
}

fn solve_parsed_until(
    model: &CollectionModel,
    objective_tiers: &[ListObjectiveTier],
    parsed: &[ListConstraint],
    stop: &AtomicBool,
    should_stop: &dyn Fn() -> bool,
    on_improve: &mut impl FnMut(&[i64]),
) -> Outcome {
    if should_stop() {
        return interrupted_outcome(SolveStats::default());
    }
    let mut impossible = false;
    for constraint in parsed {
        if should_stop() {
            return interrupted_outcome(SolveStats::default());
        }
        impossible |= matches!(constraint, ListConstraint::Impossible);
    }
    if impossible {
        return if should_stop() {
            interrupted_outcome(SolveStats::default())
        } else {
            Outcome { status: Status::Unsatisfiable, objectives: Vec::new(), solution: None, stats: SolveStats::default() }
        };
    }

    // A `Reduction` constraint (scan) is not posted into the solver, only checked
    // on completed ordered lists, so it needs the enumeration path. The member-var
    // and chrono paths would accept assignments that violate it.
    let Some(has_unposted_reduction) = any_until(parsed, should_stop, |constraint| matches!(constraint, ListConstraint::Reduction { .. }))
    else {
        return interrupted_outcome(SolveStats::default());
    };
    let Some(has_unposted_global) = any_until(&model.globals, should_stop, |global| {
        !matches!(global, list::GlobalConstraint::ListLe { .. } | list::GlobalConstraint::SameList { .. })
    }) else {
        return interrupted_outcome(SolveStats::default());
    };
    let Some(order_dependent_objective) = objective_tiers_order_dependent_until(objective_tiers, should_stop) else {
        return interrupted_outcome(SolveStats::default());
    };
    if has_unposted_reduction || has_unposted_global || order_dependent_objective {
        if should_stop() {
            return interrupted_outcome(SolveStats::default());
        }
        return run_ordered_search(model, parsed, objective_tiers, stop, on_improve);
    }

    let Some(member_var_supported) = member_var_exact_supported_until(parsed, &model.globals, objective_tiers, should_stop) else {
        return interrupted_outcome(SolveStats::default());
    };
    let Some((solver, lists)) = build_solver(model, parsed, should_stop) else {
        return interrupted_outcome(SolveStats::default());
    };
    if should_stop() {
        return interrupted_outcome(SolveStats::default());
    }
    if member_var_supported {
        run_member_var_search(solver, &lists, &model.items, objective_tiers, stop, should_stop, on_improve)
    } else {
        run_domain_search(solver, objective_tiers, stop, on_improve)
    }
}

/// Whether the member-variable CDCL backend should solve this model exactly.
/// `ItemSum` is integer-backed and explained, and the objective is now integer-
/// backed too (branch-and-bound rather than enumerate), but on `ItemSum`-heavy
/// COPs such as bin packing the member-var *search* still trails chronological
/// search: the objective bound alone is not enough without a packing-aware
/// branching heuristic, so those stay on chrono to avoid regressing them. Item
/// precedence (`ListLe`) is now explained and oracle-checked, but routing it is a
/// separate (perf) decision still pending, so it stays on chrono for now. The
/// routed set: partition + `Length` + `same_list`, with objectives that do not
/// depend on list order.
fn member_var_exact_supported_until(
    parsed: &[ListConstraint],
    globals: &[list::GlobalConstraint],
    objective_tiers: &[ListObjectiveTier],
    should_stop: &dyn Fn() -> bool,
) -> Option<bool> {
    for constraint in parsed {
        if should_stop() {
            return None;
        }
        if !matches!(constraint, ListConstraint::Length { .. }) {
            return Some(false);
        }
    }
    for global in globals {
        if should_stop() {
            return None;
        }
        if !matches!(global, list::GlobalConstraint::SameList { .. }) {
            return Some(false);
        }
    }
    for tier in objective_tiers {
        match member_var_objective_supported_until(tier, should_stop) {
            Some(true) => {}
            result => return result,
        }
    }
    Some(!should_stop())
}

fn member_var_objective_supported_until(tier: &ListObjectiveTier, should_stop: &dyn Fn() -> bool) -> Option<bool> {
    for term in &tier.terms {
        if should_stop() {
            return None;
        }
        if !member_var_objective_term_supported(term) {
            return Some(false);
        }
    }
    if let Some(max_terms) = &tier.max_terms {
        for max_term in max_terms {
            for group in &max_term.groups {
                for term in group {
                    if should_stop() {
                        return None;
                    }
                    if !member_var_objective_term_supported(term) {
                        return Some(false);
                    }
                }
            }
        }
    }
    Some(!should_stop())
}

fn member_var_objective_term_supported(term: &ListObjectiveTerm) -> bool {
    match term {
        ListObjectiveTerm::Sum { .. } | ListObjectiveTerm::Count { .. } | ListObjectiveTerm::Used { .. } => true,
        ListObjectiveTerm::Reduction(reduction) => matches!(reduction.iterable, list::Iterable::Items(_)),
    }
}

fn objective_tiers_order_dependent_until(tiers: &[ListObjectiveTier], should_stop: &dyn Fn() -> bool) -> Option<bool> {
    for tier in tiers {
        match member_var_objective_supported_until(tier, should_stop) {
            Some(true) => {}
            Some(false) => return Some(true),
            None => return None,
        }
    }
    Some(false)
}

fn any_until<T>(values: &[T], should_stop: &dyn Fn() -> bool, predicate: impl Fn(&T) -> bool) -> Option<bool> {
    for value in values {
        if should_stop() {
            return None;
        }
        if predicate(value) {
            return Some(true);
        }
    }
    (!should_stop()).then_some(false)
}

fn run_ordered_search(
    model: &CollectionModel,
    parsed: &[ListConstraint],
    objective_tiers: &[ListObjectiveTier],
    stop: &AtomicBool,
    on_improve: &mut impl FnMut(&[i64]),
) -> Outcome {
    let has_objective = !objective_tiers.is_empty();
    let mut lists = vec![Vec::new(); model.lists];
    let mut solution = None;
    let mut best_objectives = Vec::new();
    let mut stats = SolveStats::default();

    let complete = enumerate_ordered_partitions(
        0,
        &model.items,
        &mut lists,
        parsed,
        &model.globals,
        objective_tiers,
        has_objective,
        &mut solution,
        &mut best_objectives,
        &mut stats,
        stop,
        on_improve,
    ) && !stop_requested(stop);

    if stop_requested(stop) {
        return interrupted_outcome(stats);
    }

    Outcome { status: outcome_status(solution.is_some(), has_objective, complete), objectives: best_objectives, solution, stats }
}

#[allow(clippy::too_many_arguments)]
fn enumerate_ordered_partitions(
    item_idx: usize,
    items: &[i32],
    lists: &mut [Vec<i32>],
    parsed: &[ListConstraint],
    globals: &[list::GlobalConstraint],
    objective_tiers: &[ListObjectiveTier],
    has_objective: bool,
    solution: &mut Option<Vec<Vec<i32>>>,
    best_objectives: &mut Vec<i64>,
    stats: &mut SolveStats,
    stop: &AtomicBool,
    on_improve: &mut impl FnMut(&[i64]),
) -> bool {
    if stop_requested(stop) {
        return false;
    }
    if item_idx == items.len() {
        stats.solutions += 1;
        if !ordered_lists_feasible(parsed, globals, lists) {
            stats.failures += 1;
            return true;
        }
        if !has_objective {
            let candidate = lists.to_vec();
            if stop_requested(stop) {
                return false;
            }
            *solution = Some(candidate);
            return false;
        }
        let candidate = evaluate_list_objectives(objective_tiers, lists);
        if stop_requested(stop) {
            return false;
        }
        if solution.is_none() || list_objectives_better(&candidate, best_objectives, objective_tiers) {
            on_improve(&candidate);
            if stop_requested(stop) {
                return false;
            }
            *solution = Some(lists.to_vec());
            *best_objectives = candidate;
        }
        return true;
    }

    let item = items[item_idx];
    for list_idx in 0..lists.len() {
        if list_exceeds_max_len(parsed, list_idx, lists[list_idx].len() + 1) {
            continue;
        }
        for pos in 0..=lists[list_idx].len() {
            stats.nodes += 1;
            lists[list_idx].insert(pos, item);
            let complete = enumerate_ordered_partitions(
                item_idx + 1,
                items,
                lists,
                parsed,
                globals,
                objective_tiers,
                has_objective,
                solution,
                best_objectives,
                stats,
                stop,
                on_improve,
            );
            lists[list_idx].remove(pos);
            if !complete {
                return false;
            }
        }
    }
    true
}

fn list_exceeds_max_len(parsed: &[ListConstraint], list: usize, len: usize) -> bool {
    parsed.iter().any(|constraint| matches!(constraint, ListConstraint::Length { list: l, max, .. } if *l == list && len > *max))
}

fn ordered_lists_feasible(parsed: &[ListConstraint], globals: &[list::GlobalConstraint], lists: &[Vec<i32>]) -> bool {
    parsed.iter().all(|constraint| match constraint {
        ListConstraint::Length { list, min, max } => lists.get(*list).is_some_and(|items| (*min..=*max).contains(&items.len())),
        ListConstraint::ItemSum { list, weights, min, max } => lists.get(*list).is_some_and(|items| {
            let total = items.iter().fold(0i64, |sum, item| sum.saturating_add(objective_weight(weights, *item)));
            (*min..=*max).contains(&total)
        }),
        ListConstraint::Reduction { reduction, op, rhs } => match list::eval_reduction_on_lists(reduction, lists) {
            Some(value) => match op {
                list::Op::Le => value <= *rhs,
                list::Op::Ge => value >= *rhs,
                list::Op::Eq => value == *rhs,
            },
            None => false,
        },
        ListConstraint::Impossible => false,
    }) && globals.iter().all(|global| match global {
        list::GlobalConstraint::ListLe { before, after } => {
            let Some(before_list) = ordered_item_owner(lists, *before) else { return false };
            let Some(after_list) = ordered_item_owner(lists, *after) else { return false };
            before_list <= after_list
        }
        list::GlobalConstraint::SameList { a, b } => {
            let Some(a_list) = ordered_item_owner(lists, *a) else { return false };
            let Some(b_list) = ordered_item_owner(lists, *b) else { return false };
            a_list == b_list
        }
        list::GlobalConstraint::DifferentList { a, b } => {
            ordered_item_owner(lists, *a).zip(ordered_item_owner(lists, *b)).is_some_and(|(a, b)| a != b)
        }
        list::GlobalConstraint::AllSameList { items } => {
            let mut owners = items.iter().map(|item| ordered_item_owner(lists, *item));
            let Some(Some(first)) = owners.next() else { return false };
            owners.all(|owner| owner == Some(first))
        }
        list::GlobalConstraint::AllDifferentLists { items } => {
            let mut owners = std::collections::HashSet::with_capacity(items.len());
            items.iter().all(|item| ordered_item_owner(lists, *item).is_some_and(|owner| owners.insert(owner)))
        }
        list::GlobalConstraint::ListDistance { a, b, min, max } => {
            ordered_item_owner(lists, *a).zip(ordered_item_owner(lists, *b)).is_some_and(|(a, b)| (*min..=*max).contains(&a.abs_diff(b)))
        }
    })
}

fn objective_weight(weights: &[(i32, i64)], item: i32) -> i64 {
    weights.iter().find_map(|(known, weight)| (*known == item).then_some(*weight)).unwrap_or(0)
}

fn ordered_item_owner(lists: &[Vec<i32>], item: i32) -> Option<usize> {
    lists.iter().position(|list| list.contains(&item))
}

fn run_member_var_search(
    mut solver: Solver,
    lists: &[ListId],
    items: &[i32],
    objective_tiers: &[ListObjectiveTier],
    stop: &AtomicBool,
    should_stop: &dyn Fn() -> bool,
    on_improve: &mut impl FnMut(&[i64]),
) -> Outcome {
    if should_stop() {
        return interrupted_outcome(SolveStats::default());
    }
    // A single linearizable tier is integer-backed and solved by branch-and-bound
    // (a real objective bound), instead of enumerating every feasible solution and
    // scoring it externally. Multiple tiers (lex) keep the enumerate path.
    if objective_tiers.len() == 1 {
        if let Some((coeffs, obj_vars)) = lower_linear_tier(&mut solver, lists, &objective_tiers[0], should_stop) {
            if should_stop() {
                return interrupted_outcome(SolveStats::default());
            }
            return optimize_linear_tier(
                solver,
                lists,
                items,
                objective_tiers[0].minimize,
                coeffs,
                obj_vars,
                stop,
                should_stop,
                on_improve,
            );
        }
        if should_stop() {
            return interrupted_outcome(SolveStats::default());
        }
    }

    let has_objective = !objective_tiers.is_empty();
    let Some(vars) = member_vars(&solver, lists, items, should_stop) else {
        return interrupted_outcome(SolveStats::default());
    };
    let mut halted_early = false;
    let mut solution = None;
    let mut best_objectives = Vec::new();
    let stats = search::solve_interruptible(
        &mut solver,
        &vars,
        |solver| {
            let Some(candidate_solution) = collect_list_solution_until(solver, lists, items, should_stop) else {
                return SearchControl::Stop;
            };
            if !has_objective {
                solution = Some(candidate_solution);
                halted_early = true;
                return SearchControl::Stop;
            }
            let candidate = evaluate_list_objectives(objective_tiers, &candidate_solution);
            if should_stop() {
                return SearchControl::Stop;
            }
            if solution.is_none() || list_objectives_better(&candidate, &best_objectives, objective_tiers) {
                on_improve(&candidate);
                if should_stop() {
                    return SearchControl::Stop;
                }
                solution = Some(candidate_solution);
                best_objectives = candidate;
            }
            SearchControl::Continue
        },
        stop,
    );
    if should_stop() || stop_requested(stop) {
        return interrupted_outcome(stats);
    }
    let complete = !halted_early;

    Outcome { status: outcome_status(solution.is_some(), has_objective, complete), objectives: best_objectives, solution, stats }
}

/// Lower a single objective tier to a linear form `sum(coeff_i * var_i)` over
/// member variables (`Sum` terms) and freshly posted `used = length >= 1`
/// indicators (`Used` terms). Returns `None` for anything not yet linearizable
/// (currently `Count`), so the caller falls back to the enumerate path.
fn lower_linear_tier(
    solver: &mut Solver,
    lists: &[ListId],
    tier: &ListObjectiveTier,
    should_stop: &dyn Fn() -> bool,
) -> Option<(Vec<i64>, Vec<VarId>)> {
    if should_stop() {
        return None;
    }
    if tier.max_terms.as_ref().is_some_and(|terms| !terms.is_empty()) {
        return None;
    }
    let mut coeffs = Vec::new();
    let mut vars = Vec::new();
    for term in &tier.terms {
        if should_stop() {
            return None;
        }
        match term {
            ListObjectiveTerm::Sum { list, weights, coeff } => {
                for &(item, weight) in weights {
                    if should_stop() {
                        return None;
                    }
                    let weight = weight.saturating_mul(*coeff);
                    if weight == 0 {
                        continue;
                    }
                    coeffs.push(weight);
                    vars.push(solver.store.list_member_var(lists[*list], item)?);
                }
            }
            ListObjectiveTerm::Used { list, coeff } => {
                if *coeff == 0 {
                    continue;
                }
                if should_stop() {
                    return None;
                }
                let used = solver.store.new_var_range(0, 1);
                list_constraints::list_used_until(solver, lists[*list], used, should_stop)?;
                coeffs.push(*coeff);
                vars.push(used);
            }
            ListObjectiveTerm::Count { .. } | ListObjectiveTerm::Reduction(_) => return None,
        }
    }
    (!should_stop()).then_some((coeffs, vars))
}

/// Branch-and-bound over the member variables with an integer-backed linear
/// objective, so the search prunes on the objective bound instead of enumerating
/// every feasible solution.
#[allow(clippy::too_many_arguments)]
fn optimize_linear_tier(
    mut solver: Solver,
    lists: &[ListId],
    items: &[i32],
    minimize: bool,
    coeffs: Vec<i64>,
    obj_vars: Vec<VarId>,
    stop: &AtomicBool,
    should_stop: &dyn Fn() -> bool,
    on_improve: &mut impl FnMut(&[i64]),
) -> Outcome {
    let Some(mut search_vars) = member_vars(&solver, lists, items, should_stop) else {
        return interrupted_outcome(SolveStats::default());
    };
    let n_members = search_vars.len();
    for &var in &obj_vars {
        if should_stop() {
            return interrupted_outcome(SolveStats::default());
        }
        if !search_vars.contains(&var) {
            search_vars.push(var);
        }
    }
    let (best, stats, complete) = search::optimize_seeded(
        &mut solver,
        &search_vars,
        SearchObjective::Linear { coeffs: &coeffs, vars: &obj_vars },
        minimize,
        stop,
        0,
        None,
        None,
        &[],
        None,
        Vec::new(),
        Vec::new(),
        |value, _| on_improve(&[value]),
    );
    if should_stop() || stop_requested(stop) {
        return interrupted_outcome(stats);
    }
    match best {
        Some((assignment, value)) => {
            let Some(solution) = list_solution_from_members(&assignment[..n_members], lists.len(), items, should_stop) else {
                return interrupted_outcome(stats);
            };
            Outcome {
                status: if complete { Status::Optimal } else { Status::Satisfiable },
                objectives: vec![value],
                solution: Some(solution),
                stats,
            }
        }
        None => Outcome {
            status: if complete { Status::Unsatisfiable } else { Status::Unknown },
            objectives: Vec::new(),
            solution: None,
            stats,
        },
    }
}

/// Reconstruct each list's contents from a flat member-variable assignment, in
/// `member_vars` order (list-major, item-minor).
fn list_solution_from_members(
    member_values: &[i32],
    n_lists: usize,
    items: &[i32],
    should_stop: &dyn Fn() -> bool,
) -> Option<Vec<Vec<i32>>> {
    let n = items.len();
    let mut lists = Vec::with_capacity(n_lists);
    for li in 0..n_lists {
        let mut contents = Vec::new();
        for (idx, &item) in items.iter().enumerate() {
            if should_stop() {
                return None;
            }
            if member_values[li * n + idx] == 1 {
                contents.push(item);
            }
        }
        lists.push(contents);
    }
    (!should_stop()).then_some(lists)
}

fn run_domain_search(
    mut solver: Solver,
    objective_tiers: &[ListObjectiveTier],
    stop: &AtomicBool,
    on_improve: &mut impl FnMut(&[i64]),
) -> Outcome {
    let has_objective = !objective_tiers.is_empty();
    let mut solution = None;
    let mut best_objectives = Vec::new();
    let (stats, complete) = search::solve_domains_interruptible(
        &mut solver,
        |_, domain| {
            if stop_requested(stop) {
                return SearchControl::Stop;
            }
            if !has_objective {
                let candidate = domain.lists.clone();
                if stop_requested(stop) {
                    return SearchControl::Stop;
                }
                solution = Some(candidate);
                return SearchControl::Stop;
            }
            let candidate = evaluate_list_objectives(objective_tiers, &domain.lists);
            if stop_requested(stop) {
                return SearchControl::Stop;
            }
            if solution.is_none() || list_objectives_better(&candidate, &best_objectives, objective_tiers) {
                on_improve(&candidate);
                if stop_requested(stop) {
                    return SearchControl::Stop;
                }
                solution = Some(domain.lists.clone());
                best_objectives = candidate;
            }
            SearchControl::Continue
        },
        stop,
    );
    if stop_requested(stop) {
        return interrupted_outcome(stats);
    }
    Outcome { status: outcome_status(solution.is_some(), has_objective, complete), objectives: best_objectives, solution, stats }
}

fn outcome_status(has_solution: bool, has_objective: bool, complete: bool) -> Status {
    if has_solution {
        if has_objective && complete {
            Status::Optimal
        } else {
            Status::Satisfiable
        }
    } else if complete {
        Status::Unsatisfiable
    } else {
        Status::Unknown
    }
}

fn parse_constraints(model: &CollectionModel) -> Option<Vec<ListConstraint>> {
    let stop = AtomicBool::new(false);
    parse_constraints_interruptible(model, &stop)
}

fn parse_constraints_interruptible(model: &CollectionModel, stop: &AtomicBool) -> Option<Vec<ListConstraint>> {
    let mut parsed = Vec::with_capacity(model.constraints.len());
    for constraint in &model.constraints {
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return None;
        }
        parsed.push(list_constraint(constraint, &model.items, Some(stop))?);
    }
    Some(parsed)
}

fn build_solver(model: &CollectionModel, parsed: &[ListConstraint], should_stop: &dyn Fn() -> bool) -> Option<(Solver, Vec<ListId>)> {
    if should_stop() {
        return None;
    }
    let mut solver = Solver::new();
    let mut lists = Vec::with_capacity(model.lists);
    for _ in 0..model.lists {
        let universe = clone_slice_until(&model.items, should_stop)?;
        lists.push(solver.store.new_list_interruptible(universe, should_stop)?);
    }
    // Couple each list's memberships to its length variable.
    for &list in &lists {
        let items = clone_slice_until(&model.items, should_stop)?;
        list_constraints::list_cardinality_until(&mut solver, list, items, should_stop)?;
    }
    let partition_lists = clone_slice_until(&lists, should_stop)?;
    let partition_items = clone_slice_until(&model.items, should_stop)?;
    list_constraints::partition_until(&mut solver, partition_lists, partition_items, should_stop)?;
    for constraint in parsed {
        if should_stop() {
            return None;
        }
        match constraint {
            ListConstraint::Length { list, min, max } => {
                list_constraints::list_len_until(&mut solver, lists[*list], *min, *max, should_stop)?;
            }
            ListConstraint::ItemSum { list, weights, min, max } => {
                let weights = clone_slice_until(weights, should_stop)?;
                list_constraints::list_item_sum_until(&mut solver, lists[*list], weights, *min, *max, should_stop)?;
            }
            // Enumeration-checked in `ordered_lists_feasible`; nothing to post.
            ListConstraint::Reduction { .. } | ListConstraint::Impossible => {}
        }
    }
    for global in &model.globals {
        if should_stop() {
            return None;
        }
        match global {
            list::GlobalConstraint::ListLe { before, after } => {
                let global_lists = clone_slice_until(&lists, should_stop)?;
                list_constraints::item_precedence_until(&mut solver, global_lists, *before, *after, should_stop)?;
            }
            list::GlobalConstraint::SameList { a, b } => {
                let global_lists = clone_slice_until(&lists, should_stop)?;
                list_constraints::same_list_until(&mut solver, global_lists, *a, *b, should_stop)?;
            }
            list::GlobalConstraint::DifferentList { .. }
            | list::GlobalConstraint::AllSameList { .. }
            | list::GlobalConstraint::AllDifferentLists { .. }
            | list::GlobalConstraint::ListDistance { .. } => {}
        }
    }

    (!should_stop()).then_some((solver, lists))
}

fn clone_slice_until<T: Copy>(values: &[T], should_stop: &dyn Fn() -> bool) -> Option<Vec<T>> {
    if should_stop() {
        return None;
    }
    let mut cloned = Vec::with_capacity(values.len());
    for &value in values {
        if should_stop() {
            return None;
        }
        cloned.push(value);
    }
    (!should_stop()).then_some(cloned)
}

fn member_vars(solver: &Solver, lists: &[ListId], items: &[i32], should_stop: &dyn Fn() -> bool) -> Option<Vec<crate::ids::VarId>> {
    if should_stop() {
        return None;
    }
    let mut vars = Vec::with_capacity(lists.len().saturating_mul(items.len()));
    for &list in lists {
        for &item in items {
            if should_stop() {
                return None;
            }
            vars.push(solver.store.list_member_var(list, item).expect("new list contains every model item"));
        }
    }
    (!should_stop()).then_some(vars)
}

fn collect_list_solution_until(solver: &Solver, lists: &[ListId], items: &[i32], should_stop: &dyn Fn() -> bool) -> Option<Vec<Vec<i32>>> {
    let mut solution = Vec::with_capacity(lists.len());
    for &list in lists {
        let mut contents = Vec::new();
        for &item in items {
            if should_stop() {
                return None;
            }
            if solver.store.list_required(list, item) {
                contents.push(item);
            }
        }
        solution.push(contents);
    }
    (!should_stop()).then_some(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::list::{GlobalConstraint, ListObjectiveTerm};

    fn chrono_solve(model: &CollectionModel, objective_tiers: &[ListObjectiveTier]) -> Outcome {
        let parsed = parse_constraints(model).expect("supported test model");
        let (mut solver, _) = build_solver(model, &parsed, &|| false).expect("non-interruptible test build completes");
        let stop = AtomicBool::new(false);
        let has_objective = !objective_tiers.is_empty();
        let mut solution = None;
        let mut best_objectives = Vec::new();
        let (stats, complete) = search::solve_domains_interruptible(
            &mut solver,
            |_, domain| {
                if !has_objective {
                    solution = Some(domain.lists.clone());
                    return SearchControl::Stop;
                }
                let candidate = evaluate_list_objectives(objective_tiers, &domain.lists);
                if solution.is_none() || list_objectives_better(&candidate, &best_objectives, objective_tiers) {
                    solution = Some(domain.lists.clone());
                    best_objectives = candidate;
                }
                SearchControl::Continue
            },
            &stop,
        );
        let status = if solution.is_some() {
            if has_objective && complete {
                Status::Optimal
            } else {
                Status::Satisfiable
            }
        } else if complete {
            Status::Unsatisfiable
        } else {
            Status::Unknown
        };
        Outcome { status, objectives: best_objectives, solution, stats }
    }

    #[test]
    fn member_var_backend_matches_chrono_for_partition_same_list() {
        let model = CollectionModel {
            items: vec![10, 20, 30],
            lists: 3,
            objectives: Vec::new(),
            constraints: Vec::new(),
            globals: vec![GlobalConstraint::SameList { a: 10, b: 20 }],
            schedule: None,
        };
        let objective_tiers = vec![ListObjectiveTier {
            minimize: true,
            terms: vec![
                ListObjectiveTerm::Sum { list: 0, weights: vec![(10, 0), (20, 0), (30, 3)], coeff: 1 },
                ListObjectiveTerm::Sum { list: 1, weights: vec![(10, 10), (20, 10), (30, 0)], coeff: 1 },
                ListObjectiveTerm::Sum { list: 2, weights: vec![(10, 20), (20, 20), (30, 1)], coeff: 1 },
            ],
            max_terms: None,
        }];
        let stop = AtomicBool::new(false);
        let member = solve(&model, &objective_tiers, &stop, |_| {}).expect("solve").expect("supported");
        let chrono = chrono_solve(&model, &objective_tiers);

        assert_eq!(member.status, Status::Optimal);
        assert_eq!(member.status, chrono.status);
        assert_eq!(member.objectives, chrono.objectives);
        assert_eq!(member.solution, chrono.solution);
        assert_eq!(member.objectives, vec![0]);
        let solution = member.solution.expect("solution");
        assert!(solution.iter().any(|list| list.contains(&10) && list.contains(&20)), "same_list keeps 10 and 20 together");
    }

    #[test]
    fn member_var_backend_matches_chrono_for_list_len() {
        // list 0 must hold exactly two of three items (a Count == 2 reduction). The
        // expanded predicate routes this to the member-var CDCL backend; it must
        // match chronological domain search, including the optimum.
        use crate::model::list::{Constraint, ExprArena, Iterable, Op, ReduceOp, Reduction};
        let mut arena = ExprArena::default();
        let body = arena.constant(1);
        let length_eq_two = Constraint {
            reduction: Reduction { op: ReduceOp::Count, iterable: Iterable::Items(0), arena, body, coeff: 1 },
            op: Op::Eq,
            rhs: 2,
        };
        let model = CollectionModel {
            items: vec![10, 20, 30],
            lists: 2,
            objectives: Vec::new(),
            constraints: vec![length_eq_two],
            globals: Vec::new(),
            schedule: None,
        };
        // Unique optimum: list 1 holds exactly one item; the cheapest is item 10.
        let objective_tiers = vec![ListObjectiveTier {
            minimize: true,
            terms: vec![ListObjectiveTerm::Sum { list: 1, weights: vec![(10, 1), (20, 2), (30, 3)], coeff: 1 }],
            max_terms: None,
        }];
        let stop = AtomicBool::new(false);
        let member = solve(&model, &objective_tiers, &stop, |_| {}).expect("solve").expect("supported");
        let chrono = chrono_solve(&model, &objective_tiers);

        assert_eq!(member.status, Status::Optimal);
        assert_eq!(member.status, chrono.status);
        assert_eq!(member.objectives, chrono.objectives);
        assert_eq!(member.solution, chrono.solution);
        assert_eq!(member.objectives, vec![1]);
        let solution = member.solution.expect("solution");
        assert_eq!(solution[0].len(), 2, "list 0 holds exactly two items");
    }

    #[test]
    fn member_var_backend_matches_chrono_for_used_objective() {
        // Minimize the number of non-empty lists (the bin-packing `used` objective)
        // with no capacity limit: the optimum packs everything into one list. This
        // integer-backs `used` via ListUsed and branch-and-bounds, and must reach
        // the same optimum as chronological enumerate-and-score.
        let model = CollectionModel {
            items: vec![10, 20, 30],
            lists: 3,
            objectives: Vec::new(),
            constraints: Vec::new(),
            globals: Vec::new(),
            schedule: None,
        };
        let objective_tiers = vec![ListObjectiveTier {
            minimize: true,
            terms: vec![
                ListObjectiveTerm::Used { list: 0, coeff: 1 },
                ListObjectiveTerm::Used { list: 1, coeff: 1 },
                ListObjectiveTerm::Used { list: 2, coeff: 1 },
            ],
            max_terms: None,
        }];
        let stop = AtomicBool::new(false);
        let member = solve(&model, &objective_tiers, &stop, |_| {}).expect("solve").expect("supported");
        let chrono = chrono_solve(&model, &objective_tiers);

        assert_eq!(member.status, Status::Optimal);
        assert_eq!(member.status, chrono.status);
        assert_eq!(member.objectives, chrono.objectives);
        assert_eq!(member.objectives, vec![1], "all items in a single list means one used list");
    }

    #[test]
    fn order_dependent_objective_uses_domain_search() {
        use std::sync::Arc;

        use crate::model::list::{ExprArena, Iterable, ObjectiveTier, ReduceOp, Reduction};
        use crate::model::list_objective_tiers;

        let mut dist = vec![vec![50; 4]; 4];
        dist[0][2] = 1;
        dist[2][1] = 1;
        dist[1][3] = 1;
        dist[3][0] = 1;

        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let j = arena.arg(1);
        let body = arena.matrix(Arc::new(dist), i, j);
        let edge_cost = Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list: 0, start: 0, end: 0 }, arena, body, coeff: 1 };
        let model = CollectionModel {
            items: vec![1, 2, 3],
            lists: 1,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge_cost], max_terms: None }],
            constraints: Vec::new(),
            globals: Vec::new(),
            schedule: None,
        };
        let objective_tiers = list_objective_tiers(&model.objectives, &model.items).expect("edge objective parses");
        let stop = AtomicBool::new(false);

        let outcome = solve(&model, &objective_tiers, &stop, |_| {}).expect("solve").expect("supported");

        assert_eq!(outcome.status, Status::Optimal);
        assert_eq!(outcome.objectives, vec![4]);
        assert_eq!(outcome.solution, Some(vec![vec![2, 1, 3]]));
    }
}
