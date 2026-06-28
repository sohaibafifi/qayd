//! Structured list exact-search backend.
//!
//! Lowers a list-reduction model to list domains, posts the supported
//! constraints, and runs the domain DFS with lexicographic objective
//! branch-and-bound. Returns `Ok(None)` when the model shape is not supported.

use std::sync::atomic::AtomicBool;

use crate::constraints::list as list_constraints;
use crate::ids::ListId;
use crate::model::list::{self, CollectionModel};
use crate::model::{evaluate_list_objectives, list_objectives_better, ListObjectiveTier};
use crate::search::{self, SearchControl, SolveStats};
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
        return used_constraint_as_length(list, constraint.op, constraint.rhs, items.len());
    }

    if matches!(constraint.reduction.op, list::ReduceOp::Count) {
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
        return if min > max {
            Some(ListConstraint::Impossible)
        } else {
            Some(ListConstraint::Length { list, min, max })
        };
    }

    if matches!(constraint.reduction.op, list::ReduceOp::Sum) {
        let mut weights = Vec::with_capacity(items.len());
        for &item in items {
            weights.push((item, eval_collection_expr_one(&constraint.reduction.arena.exprs, constraint.reduction.body, i64::from(item))?));
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
        list::Expr::Add(a, b) => {
            eval_collection_expr_one(arena, *a, arg0)?.checked_add(eval_collection_expr_one(arena, *b, arg0)?)?
        }
        list::Expr::Sub(a, b) => {
            eval_collection_expr_one(arena, *a, arg0)?.checked_sub(eval_collection_expr_one(arena, *b, arg0)?)?
        }
        list::Expr::Mul(a, b) => {
            eval_collection_expr_one(arena, *a, arg0)?.checked_mul(eval_collection_expr_one(arena, *b, arg0)?)?
        }
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
    let has_objective = !objective_tiers.is_empty();

    let mut parsed = Vec::with_capacity(model.constraints.len());
    for constraint in &model.constraints {
        let Some(c) = list_constraint(constraint, &model.items) else {
            return Ok(None);
        };
        parsed.push(c);
    }
    if parsed.iter().any(|c| matches!(c, ListConstraint::Impossible)) {
        return Ok(Some(Outcome { status: Status::Unsatisfiable, objectives: Vec::new(), solution: None, stats: SolveStats::default() }));
    }

    let mut solver = Solver::new();
    let lists: Vec<ListId> = (0..model.lists).map(|_| solver.store.new_list(model.items.clone())).collect();
    list_constraints::partition(&mut solver, &lists, &model.items);
    for constraint in parsed {
        match constraint {
            ListConstraint::Length { list, min, max } => {
                list_constraints::list_len(&mut solver, lists[list], min, max);
            }
            ListConstraint::ItemSum { list, weights, min, max } => {
                list_constraints::list_item_sum(&mut solver, lists[list], weights, min, max);
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

    let mut solution = None;
    let mut best_objectives = Vec::new();
    let (stats, complete) = search::solve_domains_interruptible(
        &mut solver,
        |_, domain| {
            if !has_objective {
                solution = Some(domain.clone());
                return SearchControl::Stop;
            }
            let candidate = evaluate_list_objectives(objective_tiers, &domain.lists);
            if solution.is_none() || list_objectives_better(&candidate, &best_objectives, objective_tiers) {
                on_improve(&candidate);
                solution = Some(domain.clone());
                best_objectives = candidate;
            }
            SearchControl::Continue
        },
        stop,
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
    Ok(Some(Outcome { status, objectives: best_objectives, solution: solution.map(|domain| domain.lists), stats }))
}
