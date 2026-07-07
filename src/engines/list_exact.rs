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

use std::sync::atomic::AtomicBool;

use crate::constraints::list as list_constraints;
use crate::ids::{ListId, VarId};
use crate::model::list::{self, CollectionModel, ListObjectiveTerm};
use crate::model::{evaluate_list_objectives, list_objectives_better, ListObjectiveTier};
use crate::search::{self, Objective as SearchObjective, SearchControl, SolveStats};
use crate::store::Solver;

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

#[derive(Clone)]
enum ListConstraint {
    Length { list: usize, min: usize, max: usize },
    ItemSum { list: usize, weights: Vec<(i32, i64)>, min: i64, max: i64 },
    Impossible,
}

fn list_constraint(constraint: &list::Constraint, items: &[i32]) -> Option<ListConstraint> {
    let list::Iterable::Items(list) = &constraint.reduction.iterable else {
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
    let node = arena.get(id.0 as usize)?;
    Some(match node {
        list::Expr::Const(value) => *value,
        list::Expr::Arg(0) => arg0,
        list::Expr::Arg(_) => return None,
        list::Expr::Array(values, index) => {
            let index = eval_collection_expr_one(arena, *index, arg0)?;
            *values.get(usize::try_from(index).ok()?)?
        }
        list::Expr::Matrix(_, _, _) => return None,
        list::Expr::Add(a, b) => eval_collection_expr_one(arena, *a, arg0)?.checked_add(eval_collection_expr_one(arena, *b, arg0)?)?,
        list::Expr::Sub(a, b) => eval_collection_expr_one(arena, *a, arg0)?.checked_sub(eval_collection_expr_one(arena, *b, arg0)?)?,
        list::Expr::Mul(a, b) => eval_collection_expr_one(arena, *a, arg0)?.checked_mul(eval_collection_expr_one(arena, *b, arg0)?)?,
        list::Expr::Min(a, b) => eval_collection_expr_one(arena, *a, arg0)?.min(eval_collection_expr_one(arena, *b, arg0)?),
        list::Expr::Max(a, b) => eval_collection_expr_one(arena, *a, arg0)?.max(eval_collection_expr_one(arena, *b, arg0)?),
        list::Expr::Div(a, b) => {
            let numerator = eval_collection_expr_one(arena, *a, arg0)?;
            let denominator = eval_collection_expr_one(arena, *b, arg0)?;
            if denominator == 0 {
                0
            } else {
                numerator.checked_div(denominator)?
            }
        }
        list::Expr::Abs(a) => eval_collection_expr_one(arena, *a, arg0)?.saturating_abs(),
        list::Expr::Lt(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? < eval_collection_expr_one(arena, *b, arg0)?),
        list::Expr::Le(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? <= eval_collection_expr_one(arena, *b, arg0)?),
        list::Expr::Eq(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? == eval_collection_expr_one(arena, *b, arg0)?),
        list::Expr::Ne(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? != eval_collection_expr_one(arena, *b, arg0)?),
        list::Expr::IfThenElse(c, a, b) => {
            if eval_collection_expr_one(arena, *c, arg0)? != 0 {
                eval_collection_expr_one(arena, *a, arg0)?
            } else {
                eval_collection_expr_one(arena, *b, arg0)?
            }
        }
    })
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
    let Some(parsed) = parse_constraints(model) else {
        return Ok(None);
    };
    if parsed.iter().any(|c| matches!(c, ListConstraint::Impossible)) {
        return Ok(Some(Outcome { status: Status::Unsatisfiable, objectives: Vec::new(), solution: None, stats: SolveStats::default() }));
    }

    let (solver, lists) = build_solver(model, &parsed);
    let outcome = if member_var_exact_supported(&parsed, &model.globals) {
        run_member_var_search(solver, &lists, &model.items, objective_tiers, stop, &mut on_improve)
    } else {
        run_domain_search(solver, objective_tiers, stop, &mut on_improve)
    };
    Ok(Some(outcome))
}

/// Whether the member-variable CDCL backend should solve this model exactly.
/// `ItemSum` is integer-backed and explained, and the objective is now integer-
/// backed too (branch-and-bound rather than enumerate), but on `ItemSum`-heavy
/// COPs such as bin packing the member-var *search* still trails chronological
/// search: the objective bound alone is not enough without a packing-aware
/// branching heuristic, so those stay on chrono to avoid regressing them. Item
/// precedence (`ListLe`) is now explained and oracle-checked, but routing it is a
/// separate (perf) decision still pending, so it stays on chrono for now. The
/// routed set: partition + `Length` + `same_list`.
fn member_var_exact_supported(parsed: &[ListConstraint], globals: &[list::GlobalConstraint]) -> bool {
    parsed.iter().all(|c| matches!(c, ListConstraint::Length { .. }))
        && globals.iter().all(|global| matches!(global, list::GlobalConstraint::SameList { .. }))
}

fn run_member_var_search(
    mut solver: Solver,
    lists: &[ListId],
    items: &[i32],
    objective_tiers: &[ListObjectiveTier],
    stop: &AtomicBool,
    on_improve: &mut impl FnMut(&[i64]),
) -> Outcome {
    // A single linearizable tier is integer-backed and solved by branch-and-bound
    // (a real objective bound), instead of enumerating every feasible solution and
    // scoring it externally. Multiple tiers (lex) keep the enumerate path.
    if objective_tiers.len() == 1 {
        if let Some((coeffs, obj_vars)) = lower_linear_tier(&mut solver, lists, &objective_tiers[0]) {
            return optimize_linear_tier(solver, lists, items, objective_tiers[0].minimize, coeffs, obj_vars, stop, on_improve);
        }
    }

    let has_objective = !objective_tiers.is_empty();
    let vars = member_vars(&solver, lists, items);
    let mut halted_early = false;
    let mut solution = None;
    let mut best_objectives = Vec::new();
    let stats = search::solve_interruptible(
        &mut solver,
        &vars,
        |solver| {
            let candidate_solution = collect_list_solution(solver, lists, items);
            if !has_objective {
                solution = Some(candidate_solution);
                halted_early = true;
                return SearchControl::Stop;
            }
            let candidate = evaluate_list_objectives(objective_tiers, &candidate_solution);
            if solution.is_none() || list_objectives_better(&candidate, &best_objectives, objective_tiers) {
                on_improve(&candidate);
                solution = Some(candidate_solution);
                best_objectives = candidate;
            }
            SearchControl::Continue
        },
        stop,
    );
    let complete = !halted_early && !stop.load(std::sync::atomic::Ordering::Relaxed);

    Outcome { status: outcome_status(solution.is_some(), has_objective, complete), objectives: best_objectives, solution, stats }
}

/// Lower a single objective tier to a linear form `sum(coeff_i * var_i)` over
/// member variables (`Sum` terms) and freshly posted `used = length >= 1`
/// indicators (`Used` terms). Returns `None` for anything not yet linearizable
/// (currently `Count`), so the caller falls back to the enumerate path.
fn lower_linear_tier(solver: &mut Solver, lists: &[ListId], tier: &ListObjectiveTier) -> Option<(Vec<i64>, Vec<VarId>)> {
    if tier.max_terms.as_ref().is_some_and(|terms| !terms.is_empty()) {
        return None;
    }
    let mut coeffs = Vec::new();
    let mut vars = Vec::new();
    for term in &tier.terms {
        match term {
            ListObjectiveTerm::Sum { list, weights, coeff } => {
                for &(item, weight) in weights {
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
                let used = solver.store.new_var_range(0, 1);
                list_constraints::list_used(solver, lists[*list], used);
                coeffs.push(*coeff);
                vars.push(used);
            }
            ListObjectiveTerm::Count { .. } | ListObjectiveTerm::Reduction(_) => return None,
        }
    }
    Some((coeffs, vars))
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
    on_improve: &mut impl FnMut(&[i64]),
) -> Outcome {
    let mut search_vars = member_vars(&solver, lists, items);
    let n_members = search_vars.len();
    for &var in &obj_vars {
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
    match best {
        Some((assignment, value)) => Outcome {
            status: if complete { Status::Optimal } else { Status::Satisfiable },
            objectives: vec![value],
            solution: Some(list_solution_from_members(&assignment[..n_members], lists.len(), items)),
            stats,
        },
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
fn list_solution_from_members(member_values: &[i32], n_lists: usize, items: &[i32]) -> Vec<Vec<i32>> {
    let n = items.len();
    (0..n_lists)
        .map(|li| items.iter().enumerate().filter(|&(idx, _)| member_values[li * n + idx] == 1).map(|(_, &item)| item).collect())
        .collect()
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
            if !has_objective {
                solution = Some(domain.lists.clone());
                return SearchControl::Stop;
            }
            let candidate = evaluate_list_objectives(objective_tiers, &domain.lists);
            if solution.is_none() || list_objectives_better(&candidate, &best_objectives, objective_tiers) {
                on_improve(&candidate);
                solution = Some(domain.lists.clone());
                best_objectives = candidate;
            }
            SearchControl::Continue
        },
        stop,
    );
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
    let mut parsed = Vec::with_capacity(model.constraints.len());
    for constraint in &model.constraints {
        parsed.push(list_constraint(constraint, &model.items)?);
    }
    Some(parsed)
}

fn build_solver(model: &CollectionModel, parsed: &[ListConstraint]) -> (Solver, Vec<ListId>) {
    let mut solver = Solver::new();
    let lists: Vec<ListId> = (0..model.lists).map(|_| solver.store.new_list(model.items.clone())).collect();
    // Couple each list's memberships to its length variable.
    for &list in &lists {
        list_constraints::list_cardinality(&mut solver, list);
    }
    list_constraints::partition(&mut solver, &lists, &model.items);
    for constraint in parsed {
        match constraint {
            ListConstraint::Length { list, min, max } => {
                list_constraints::list_len(&mut solver, lists[*list], *min, *max);
            }
            ListConstraint::ItemSum { list, weights, min, max } => {
                list_constraints::list_item_sum(&mut solver, lists[*list], weights.clone(), *min, *max);
            }
            ListConstraint::Impossible => {}
        }
    }
    for global in &model.globals {
        match *global {
            list::GlobalConstraint::ListLe { before, after } => {
                list_constraints::item_precedence(&mut solver, &lists, before, after);
            }
            list::GlobalConstraint::SameList { a, b } => {
                list_constraints::same_list(&mut solver, &lists, a, b);
            }
        }
    }

    (solver, lists)
}

fn member_vars(solver: &Solver, lists: &[ListId], items: &[i32]) -> Vec<crate::ids::VarId> {
    let mut vars = Vec::with_capacity(lists.len().saturating_mul(items.len()));
    for &list in lists {
        for &item in items {
            vars.push(solver.store.list_member_var(list, item).expect("new list contains every model item"));
        }
    }
    vars
}

fn collect_list_solution(solver: &Solver, lists: &[ListId], items: &[i32]) -> Vec<Vec<i32>> {
    lists.iter().map(|&list| items.iter().copied().filter(|&item| solver.store.list_required(list, item)).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::list::{GlobalConstraint, ListObjectiveTerm};

    fn chrono_solve(model: &CollectionModel, objective_tiers: &[ListObjectiveTier]) -> Outcome {
        let parsed = parse_constraints(model).expect("supported test model");
        let (mut solver, _) = build_solver(model, &parsed);
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
}
