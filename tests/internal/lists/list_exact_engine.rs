use std::sync::atomic::AtomicBool;

use crate::engines::list_exact::{build_solver, parse_constraints, solve, Outcome, Status};
use crate::model::list::{CollectionModel, GlobalConstraint, ListObjectiveTerm};
use crate::model::{evaluate_list_objectives, list_objectives_better, ListObjectiveTier};
use crate::search::{self, SearchControl};

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
