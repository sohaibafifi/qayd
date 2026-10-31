use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use qayd::engines::{list_exact, schedule};
use qayd::model::list::{
    self, CollectionModel, Constraint, ExprArena, GlobalConstraint, IntervalVar, Iterable, ObjectiveTier, Op, ReduceOp, Reduction, Schedule,
};

fn model_with_objectives(items: Vec<i32>, objectives: Vec<ObjectiveTier>) -> CollectionModel {
    CollectionModel { items, lists: 1, objectives, constraints: Vec::new(), globals: Vec::new(), schedule: None }
}

fn constant_sum(value: i64) -> Reduction {
    let mut arena = ExprArena::default();
    let body = arena.constant(value);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(0), arena, body, coeff: 1 }
}

#[test]
fn lexicographic_objective_has_no_fixed_tier_ceiling() {
    let objectives = (0..12)
        .map(|value| ObjectiveTier { minimize: value % 2 == 0, terms: vec![constant_sum(value)], max_terms: None })
        .collect::<Vec<_>>();
    let model = model_with_objectives(vec![7], objectives);
    model.validate().unwrap();
    let tiers = list::list_objective_tiers(&model.objectives, &model.items).unwrap();
    let outcome = list_exact::solve(&model, &tiers, &AtomicBool::new(false), |_| {}).unwrap().unwrap();
    assert_eq!(outcome.objectives, (0..12).collect::<Vec<_>>());
}

#[test]
fn unordered_set_reduction_uses_canonical_member_order() {
    let mut arena = ExprArena::default();
    let item = arena.arg(0);
    let values = Arc::new(vec![i64::MAX, i64::MAX, i64::MIN]);
    let body = arena.array(values, item);
    let reduction = Reduction { op: ReduceOp::Sum, iterable: Iterable::SetItems(0), arena, body, coeff: 1 };
    let model = model_with_objectives(vec![2, 0, 1], vec![ObjectiveTier { minimize: true, terms: vec![reduction], max_terms: None }]);
    model.validate().unwrap();
    let tiers = list::list_objective_tiers(&model.objectives, &model.items).unwrap();
    let outcome = list_exact::solve(&model, &tiers, &AtomicBool::new(false), |_| {}).unwrap().unwrap();
    assert_eq!(outcome.objectives, vec![-1]);
}

#[test]
fn fixed_nonlinear_piecewise_and_external_expressions_compose() {
    let name = "phase7_double_external_v1";
    if !list::external_function_registered(name) {
        list::register_external_function(name, |args| Ok(args[0].saturating_mul(2))).unwrap();
    }
    let mut arena = ExprArena::default();
    let item = arena.arg(0);
    let cube = arena.pow(item, 3);
    let left = arena.constant(1_500);
    let right = arena.constant(2_000);
    let product = arena.mul_scaled(left, right, 1_000);
    let curve = arena.piecewise_linear(item, Arc::new(vec![(0, 0), (10, 100)]));
    let external = arena.external(name, vec![item]);
    let modulo_base = arena.constant(11);
    let modulo_divisor = arena.constant(4);
    let modulo = arena.modulo(modulo_base, modulo_divisor);
    let first = arena.add(cube, product);
    let second = arena.add(curve, external);
    let third = arena.add(first, second);
    let body = arena.add(third, modulo);
    let reduction = Reduction { op: ReduceOp::Sum, iterable: Iterable::SetItems(0), arena, body, coeff: 1 };
    let model = model_with_objectives(vec![2], vec![ObjectiveTier { minimize: true, terms: vec![reduction], max_terms: None }]);
    model.validate().unwrap();
    let tiers = list::list_objective_tiers(&model.objectives, &model.items).unwrap();
    let outcome = list_exact::solve(&model, &tiers, &AtomicBool::new(false), |_| {}).unwrap().unwrap();
    assert_eq!(outcome.objectives, vec![3_035]);
}

#[test]
fn rich_inter_list_constraints_are_enforced_exactly() {
    let globals = vec![
        GlobalConstraint::AllSameList { items: Arc::new(vec![1, 2]) },
        GlobalConstraint::DifferentList { a: 2, b: 3 },
        GlobalConstraint::AllDifferentLists { items: Arc::new(vec![1, 3, 4]) },
        GlobalConstraint::ListDistance { a: 1, b: 3, min: 1, max: 2 },
        GlobalConstraint::ListLe { before: 1, after: 4 },
    ];
    let model =
        CollectionModel { items: vec![1, 2, 3, 4], lists: 4, objectives: Vec::new(), constraints: Vec::new(), globals, schedule: None };
    model.validate().unwrap();
    let outcome = list_exact::solve(&model, &[], &AtomicBool::new(false), |_| {}).unwrap().unwrap();
    let lists = outcome.solution.unwrap();
    let owner = |item| lists.iter().position(|list| list.contains(&item)).unwrap();
    assert_eq!(owner(1), owner(2));
    assert_ne!(owner(2), owner(3));
    assert_eq!(3, [owner(1), owner(3), owner(4)].into_iter().collect::<std::collections::HashSet<_>>().len());
    assert!((1..=2).contains(&owner(1).abs_diff(owner(3))));
    assert!(owner(1) <= owner(4));
}

#[test]
fn optional_schedule_interval_has_first_class_presence() {
    let schedule_model = Schedule {
        intervals: vec![
            IntervalVar { duration: 3, horizon: 10, modes: Vec::new(), optional: false },
            IntervalVar { duration: 5, horizon: 10, modes: Vec::new(), optional: true },
        ],
        precedences: Vec::new(),
        resources: Vec::new(),
        minimize_makespan: true,
    };
    let model = CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::<Constraint>::new(),
        globals: Vec::new(),
        schedule: Some(schedule_model.clone()),
    };
    model.validate().unwrap();
    let outcome = schedule::solve(&schedule_model, &AtomicBool::new(false), schedule::Options::default(), |_| {}).unwrap().unwrap();
    assert_eq!(outcome.objective, Some(3));
    assert_eq!(outcome.presences, vec![true, false]);
}

#[test]
fn set_constraint_is_checked_with_the_same_semantics_as_its_objective() {
    let mut arena = ExprArena::default();
    let body = arena.arg(0);
    let reduction = Reduction { op: ReduceOp::Sum, iterable: Iterable::SetItems(0), arena, body, coeff: 1 };
    let model = CollectionModel {
        items: vec![1, 2, 3],
        lists: 1,
        objectives: Vec::new(),
        constraints: vec![Constraint { reduction, op: Op::Eq, rhs: 6 }],
        globals: Vec::new(),
        schedule: None,
    };
    model.validate().unwrap();
    let outcome = list_exact::solve(&model, &[], &AtomicBool::new(false), |_| {}).unwrap().unwrap();
    assert!(outcome.solution.is_some());
}
