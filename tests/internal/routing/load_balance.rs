use std::sync::atomic::AtomicBool;

use crate::engines::ls::load_balance::{LoadBalanceLimits, LoadBalancePlan};
use crate::model::{Constraint, IntExpr, IntGlobalConstraint, IntVarRef, Model, Objective, Relation};

fn square(variable: IntVarRef) -> IntExpr {
    IntExpr::Mul(vec![variable.into(), variable.into()])
}

fn small_model() -> (Model, Vec<IntVarRef>, Vec<IntVarRef>, Vec<IntVarRef>) {
    let mut model = Model::new();
    let items = (0..5).map(|_| model.int_range(0, 2)).collect::<Vec<_>>();
    let counts = (0..3).map(|_| model.int_range(1, 3)).collect::<Vec<_>>();
    let loads = (0..3).map(|_| model.int_set(vec![2, 3, 4, 6, 7])).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::BinLoads {
        items: items.clone(),
        sizes: vec![1; 5],
        loads: counts.clone(),
    }));
    // Item order may differ across two load definitions.
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::BinLoads {
        items: items.iter().rev().copied().collect(),
        sizes: vec![2, 4, 3, 2, 1],
        loads: loads.clone(),
    }));
    model.add_constraint(Constraint::Intension(IntExpr::Gt(Box::new(items[4].into()), Box::new(items[0].into()))));
    model.add_constraint(Constraint::Linear {
        terms: counts.iter().map(|&variable| (1, variable)).collect(),
        relation: Relation::Eq,
        rhs: 5,
    });
    model.add_constraint(Constraint::Linear {
        terms: loads.iter().map(|&variable| (1, variable)).collect(),
        relation: Relation::Eq,
        rhs: 12,
    });
    let mut moment = items.iter().map(|&variable| (1, variable)).collect::<Vec<_>>();
    moment.extend(counts.iter().enumerate().map(|(bin, &variable)| (-(bin as i64), variable)));
    model.add_constraint(Constraint::Linear { terms: moment, relation: Relation::Eq, rhs: 0 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(loads.iter().copied().map(square).collect()) });
    (model, items, counts, loads)
}

/// Independent enumeration and reconstruction, with no use of the plan's delta
/// evaluator or feasibility helpers.
fn small_oracle(model: &Model, items: &[IntVarRef], counts: &[IntVarRef], loads: &[IntVarRef]) -> Vec<(Vec<i32>, i64)> {
    let weights = [1, 2, 3, 4, 2];
    let mut feasible = Vec::new();
    for mut code in 0..3usize.pow(items.len() as u32) {
        let mut values = vec![0; model.int_vars().len()];
        for &item in items {
            values[item.0] = (code % 3) as i32;
            code /= 3;
        }
        if values[items[0].0] >= values[items[4].0] {
            continue;
        }
        for (index, &item) in items.iter().enumerate() {
            let bin = values[item.0] as usize;
            values[counts[bin].0] += 1;
            values[loads[bin].0] += weights[index];
        }
        if values.iter().zip(model.int_vars()).any(|(&value, domain)| !domain.contains(i64::from(value))) {
            continue;
        }
        let objective = loads.iter().map(|variable| i64::from(values[variable.0]).pow(2)).sum();
        feasible.push((values, objective));
    }
    feasible
}

#[test]
fn load_balance_moves_match_full_holey_domain_and_precedence_oracle() {
    let (model, items, counts, loads) = small_model();
    let stop = AtomicBool::new(false);
    let plan = LoadBalancePlan::compile(&model, &stop, u64::MAX).unwrap();
    assert!(plan.estimated_bytes() > 0);
    let feasible = small_oracle(&model, &items, &counts, &loads);
    assert!(feasible.len() > 10);
    for (index, (incumbent, objective)) in feasible.iter().enumerate() {
        let solution = plan.construct(index as u64, Some(incumbent), &stop, LoadBalanceLimits { iterations: 7, restarts: 1 }).unwrap();
        let values = solution.values.into_iter().collect::<Option<Vec<_>>>().unwrap();
        assert!(solution.objective <= *objective);
        assert!(feasible.contains(&(values, solution.objective)));
        assert!(solution.iterations <= 7);
    }
}

#[test]
fn load_balance_constructs_optimal_replayable_candidates_deterministically() {
    let (model, items, counts, loads) = small_model();
    let stop = AtomicBool::new(false);
    let plan = LoadBalancePlan::compile(&model, &stop, u64::MAX).unwrap();
    let limits = LoadBalanceLimits { iterations: 400, restarts: 32 };
    let first = plan.construct(31, None, &stop, limits).unwrap();
    let second = plan.construct(31, None, &stop, limits).unwrap();
    assert_eq!(first.values, second.values);
    assert_eq!(first.objective, second.objective);
    assert_eq!(first.iterations, second.iterations);
    assert_eq!(first.improvements, second.improvements);
    let oracle = small_oracle(&model, &items, &counts, &loads);
    assert_eq!(first.objective, oracle.iter().map(|(_, objective)| *objective).min().unwrap());
    assert!(oracle.contains(&(first.values.into_iter().collect::<Option<Vec<_>>>().unwrap(), first.objective)));
}

#[test]
fn load_balance_preserves_weighted_square_objective_and_offset() {
    let mut model = Model::new();
    let items = (0..4).map(|_| model.int_set(vec![0, 2])).collect::<Vec<_>>();
    let loads = (0..3).map(|_| model.int_range(0, 8)).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::BinLoads {
        items: items.clone(),
        sizes: vec![1, 2, 2, 3],
        loads: loads.clone(),
    }));
    let terms = loads
        .iter()
        .enumerate()
        .map(|(bin, &variable)| IntExpr::Mul(vec![IntExpr::Constant(bin as i64 + 1), variable.into(), variable.into()]));
    model.add_objective(Objective::IntExpr {
        minimize: true,
        expr: IntExpr::Add(std::iter::once(IntExpr::Constant(-100)).chain(terms).collect()),
    });
    let stop = AtomicBool::new(false);
    let plan = LoadBalancePlan::compile(&model, &stop, u64::MAX).unwrap();
    let initial = vec![0, 0, 2, 2, 3, 0, 5];
    let solution = plan.construct(0, Some(&initial), &stop, LoadBalanceLimits { iterations: 80, restarts: 1 }).unwrap();
    let values = solution.values.into_iter().collect::<Option<Vec<_>>>().unwrap();
    assert!(items.iter().all(|variable| matches!(values[variable.0], 0 | 2)));
    assert_eq!(values[loads[1].0], 0);
    let objective =
        -100 + loads.iter().enumerate().map(|(bin, variable)| (bin as i64 + 1) * i64::from(values[variable.0]).pow(2)).sum::<i64>();
    assert_eq!(objective, solution.objective);
    assert_eq!(objective, -52);
}

#[test]
fn load_balance_obeys_cancellation_and_limits_and_rejects_unsupported_constraints() {
    let (mut model, items, _, _) = small_model();
    assert!(LoadBalancePlan::compile(&model, &AtomicBool::new(true), u64::MAX).is_none());
    assert!(LoadBalancePlan::compile(&model, &AtomicBool::new(false), 1).is_none());
    let stop = AtomicBool::new(false);
    let plan = LoadBalancePlan::compile(&model, &stop, u64::MAX).unwrap();
    assert!(plan.construct(0, None, &AtomicBool::new(true), LoadBalanceLimits::default()).is_none());
    assert!(plan.construct(0, None, &stop, LoadBalanceLimits { iterations: 0, restarts: 10 }).is_none());
    assert!(plan.construct(0, None, &stop, LoadBalanceLimits { iterations: 10, restarts: 0 }).is_none());
    model.add_constraint(Constraint::Linear { terms: vec![(1, items[0]), (1, items[1])], relation: Relation::Eq, rhs: 2 });
    assert!(LoadBalancePlan::compile(&model, &stop, u64::MAX).is_none());
}

#[test]
fn load_balance_rejects_duplicate_items_and_overflowing_objectives() {
    let mut model = Model::new();
    let item = model.int_range(0, 1);
    let loads = (0..2).map(|_| model.int_range(0, i32::MAX)).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::BinLoads {
        items: vec![item, item],
        sizes: vec![1, 1],
        loads: loads.clone(),
    }));
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(loads.iter().copied().map(square).collect()) });
    assert!(LoadBalancePlan::compile(&model, &AtomicBool::new(false), u64::MAX).is_none());

    let mut model = Model::new();
    let item = model.int_range(0, 1);
    let loads = (0..2).map(|_| model.int_range(0, i32::MAX)).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::BinLoads {
        items: vec![item],
        sizes: vec![i64::from(i32::MAX)],
        loads: loads.clone(),
    }));
    model.add_objective(Objective::IntExpr {
        minimize: true,
        expr: IntExpr::Add(
            loads.iter().map(|&variable| IntExpr::Mul(vec![IntExpr::Constant(i64::MAX), variable.into(), variable.into()])).collect(),
        ),
    });
    assert!(LoadBalancePlan::compile(&model, &AtomicBool::new(false), u64::MAX).is_none());
}

#[test]
fn load_balance_rejects_extreme_precedence_coefficients_without_overflow() {
    let (mut model, items, _, _) = small_model();
    model.add_constraint(Constraint::Linear { terms: vec![(i64::MIN, items[0]), (1, items[1])], relation: Relation::Le, rhs: 0 });
    assert!(LoadBalancePlan::compile(&model, &AtomicBool::new(false), u64::MAX).is_none());
}
