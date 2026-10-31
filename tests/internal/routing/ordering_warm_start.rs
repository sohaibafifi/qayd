use std::sync::atomic::AtomicBool;

use crate::engines::ls::ordering::{OrderingLimits, OrderingPlan};
use crate::model::{Constraint, IntExpr, IntGlobalConstraint, IntVarRef, Model, Objective, Relation};

fn hinge(a: IntVarRef, b: IntVarRef, variant: usize) -> IntExpr {
    let difference = IntExpr::Sub(Box::new(a.into()), Box::new(b.into()));
    match variant {
        0 => IntExpr::IfThenElse(Box::new(IntExpr::Lt(Box::new(a.into()), Box::new(b.into()))), Box::new(0.into()), Box::new(difference)),
        1 => IntExpr::Max(vec![difference, 0.into()]),
        _ => IntExpr::IfThenElse(Box::new(IntExpr::Le(Box::new(b.into()), Box::new(a.into()))), Box::new(difference), Box::new(0.into())),
    }
}

fn fixture(n: usize, arcs: &[(usize, usize, i64)], offset: i32, materialized: bool) -> (Model, Vec<IntVarRef>) {
    let mut model = Model::new();
    let x = (0..n).map(|_| model.int_set((offset..offset + n as i32).rev().collect())).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { variables: x.clone(), except: Vec::new() }));
    let mut terms = Vec::new();
    for (i, &(a, b, weight)) in arcs.iter().enumerate() {
        let scale = i as i64 % 2 + 1;
        let z = model.int_set((0..n as i32).map(|value| value * scale as i32).collect());
        let value = IntExpr::Mul(vec![scale.into(), hinge(x[a], x[b], i % 3)]);
        model.add_constraint(Constraint::Intension(IntExpr::Eq(Box::new(z.into()), Box::new(value))));
        terms.push((weight, z));
    }
    let expr = if materialized {
        let bound = arcs.iter().enumerate().map(|(i, &(_, _, w))| w * (i as i64 % 2 + 1) * (n as i64 - 1)).sum::<i64>();
        let objective = model.int_range(-7, i32::try_from(bound - 7).unwrap());
        terms.push((-1, objective));
        model.add_constraint(Constraint::Linear { terms, relation: Relation::Eq, rhs: 7 });
        objective.into()
    } else {
        IntExpr::Add(std::iter::once((-7).into()).chain(terms.into_iter().map(|(w, v)| IntExpr::Mul(vec![w.into(), v.into()]))).collect())
    };
    model.add_objective(Objective::IntExpr { minimize: true, expr });
    (model, x)
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn visit(order: &mut Vec<usize>, remaining: &mut Vec<usize>, output: &mut Vec<Vec<usize>>) {
        if remaining.is_empty() {
            output.push(order.clone());
            return;
        }
        for i in 0..remaining.len() {
            let value = remaining.remove(i);
            order.push(value);
            visit(order, remaining, output);
            order.pop();
            remaining.insert(i, value);
        }
    }
    let mut output = Vec::new();
    visit(&mut Vec::new(), &mut (0..n).collect(), &mut output);
    output
}

fn direct_cost(order: &[usize], arcs: &[(usize, usize, i64)]) -> i64 {
    let mut positions = vec![0i64; order.len()];
    for (i, &v) in order.iter().enumerate() {
        positions[v] = i as i64;
    }
    arcs.iter().enumerate().map(|(i, &(a, b, w))| w * (i as i64 % 2 + 1) * (positions[a] - positions[b]).max(0)).sum()
}

fn replay(model: &Model, values: &[Option<i32>]) -> i64 {
    assert_eq!(values.len(), model.int_vars().len());
    for (value, domain) in values.iter().zip(model.int_vars()) {
        assert!(domain.contains(i64::from(value.unwrap())));
    }
    for constraint in model.constraints() {
        match constraint {
            Constraint::Intension(expr) => assert_eq!(expr.evaluate(&|v| values[v.0].map(i64::from)), Some(1)),
            Constraint::Linear { terms, relation: Relation::Eq, rhs } => {
                assert_eq!(terms.iter().map(|&(w, v)| w * i64::from(values[v.0].unwrap())).sum::<i64>(), *rhs);
            }
            Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { variables, .. }) => {
                let mut actual = variables.iter().map(|v| values[v.0].unwrap()).collect::<Vec<_>>();
                actual.sort_unstable();
                actual.dedup();
                assert_eq!(actual.len(), variables.len());
            }
            _ => panic!("unexpected fixture constraint"),
        }
    }
    let [Objective::IntExpr { expr, .. }] = model.objectives() else { panic!("fixture objective") };
    expr.evaluate(&|v| values[v.0].map(i64::from)).unwrap()
}

#[test]
fn ordering_weighted_scaled_and_materialized_candidates_match_small_oracle() {
    let arcs = [(0, 1, 2), (1, 2, 3), (2, 0, 1), (2, 3, 0), (3, 1, 1)];
    let optimum = permutations(4).iter().map(|order| direct_cost(order, &arcs) - 7).min().unwrap();
    for materialized in [false, true] {
        let (model, _) = fixture(4, &arcs, -3, materialized);
        let stop = AtomicBool::new(false);
        let plan = OrderingPlan::compile(&model, &stop, u64::MAX).unwrap();
        for seed in 0..4 {
            let limits = OrderingLimits { iterations: 100, arc_evaluations: 100_000 };
            let first = plan.construct_with_limits(seed, &stop, limits).unwrap();
            let second = plan.construct_with_limits(seed, &stop, limits).unwrap();
            assert_eq!(first.values, second.values);
            assert_eq!(first.iterations, second.iterations);
            assert_eq!(first.improvements, second.improvements);
            assert_eq!(first.objective, optimum);
            assert_eq!(first.objective, replay(&model, &first.values));
        }
    }
}

#[test]
fn ordering_swap_and_composed_insertion_deltas_match_exhaustive_recalculation() {
    let arcs = [(0, 1, 4), (1, 2, 2), (2, 0, 1), (1, 0, 3), (3, 1, 2), (0, 1, 1)];
    let (model, _) = fixture(4, &arcs, 5, false);
    let plan = OrderingPlan::compile(&model, &AtomicBool::new(false), u64::MAX).unwrap();
    for order in permutations(4) {
        let base = direct_cost(&order, &arcs);
        for from in 0..4 {
            for to in 0..4 {
                if from == to {
                    continue;
                }
                let mut trial = order.clone();
                trial.swap(from, to);
                assert_eq!(plan.audit_swap_delta(&order, from, to), direct_cost(&trial, &arcs) - base);
                let mut trial = order.clone();
                let mut delta = 0;
                let mut current = from;
                while current != to {
                    let next = if current < to { current + 1 } else { current - 1 };
                    delta += plan.audit_swap_delta(&trial, current, next);
                    trial.swap(current, next);
                    current = next;
                }
                assert_eq!(delta, direct_cost(&trial, &arcs) - base);
            }
        }
    }
}

#[test]
fn ordering_scc_construction_attains_zero_for_acyclic_graphs() {
    let arcs = [(2, 0, 3), (2, 1, 1), (0, 3, 2), (1, 4, 4), (3, 4, 1)];
    let (model, _) = fixture(6, &arcs, 0, true);
    let plan = OrderingPlan::compile(&model, &AtomicBool::new(false), u64::MAX).unwrap();
    let solution = plan.construct(9, &AtomicBool::new(false)).unwrap();
    assert_eq!(solution.objective, -7);
    assert_eq!(solution.iterations, 0);
    assert_eq!(replay(&model, &solution.values), -7);
}

#[test]
fn ordering_limits_preserve_initial_candidate_and_cancellation_is_respected() {
    let (model, _) = fixture(3, &[(0, 1, 1), (1, 2, 1), (2, 0, 1)], 0, false);
    let stop = AtomicBool::new(false);
    assert!(OrderingPlan::compile(&model, &AtomicBool::new(true), u64::MAX).is_none());
    assert!(OrderingPlan::compile(&model, &stop, 0).is_none());
    let plan = OrderingPlan::compile(&model, &stop, u64::MAX).unwrap();
    assert!(OrderingPlan::compile(&model, &stop, plan.estimated_bytes() - 1).is_none());
    assert!(plan.construct(0, &AtomicBool::new(true)).is_none());
    assert!(plan.construct_with_limits(0, &stop, OrderingLimits { iterations: 0, arc_evaluations: 1_000 }).is_none());
    assert!(plan.construct_with_limits(0, &stop, OrderingLimits { iterations: 100, arc_evaluations: 0 }).is_none());
    // Exactly nine arc visits construct the three-node SCC. The hard work
    // limit must return that complete candidate rather than lose it.
    let solution = plan.construct_with_limits(0, &stop, OrderingLimits { iterations: 100, arc_evaluations: 9 }).unwrap();
    assert_eq!(solution.iterations, 0);
    assert_eq!(solution.objective, replay(&model, &solution.values));
}

#[test]
fn ordering_rejects_extra_restrictions_and_incomplete_permutation_domains() {
    let (mut model, x) = fixture(3, &[(0, 1, 1), (1, 2, 1), (2, 0, 1)], 0, false);
    model.add_constraint(Constraint::Linear { terms: vec![(1, x[0]), (-1, x[1])], relation: Relation::Eq, rhs: 1 });
    assert!(OrderingPlan::compile(&model, &AtomicBool::new(false), u64::MAX).is_none());
    let mut model = Model::new();
    let x = [model.int_set(vec![0, 2]), model.int_range(0, 1)];
    let z = model.int_range(0, 2);
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { variables: x.to_vec(), except: Vec::new() }));
    model.add_constraint(Constraint::Intension(IntExpr::Eq(Box::new(z.into()), Box::new(hinge(x[0], x[1], 0)))));
    model.add_objective(Objective::IntExpr { minimize: true, expr: z.into() });
    assert!(OrderingPlan::compile(&model, &AtomicBool::new(false), u64::MAX).is_none());
}

#[test]
fn ordering_rejects_negative_weights_overflow_and_restrictive_penalty_domains() {
    for weight in [-1, i64::MAX] {
        let (model, _) = fixture(3, &[(0, 1, weight)], 0, false);
        assert!(OrderingPlan::compile(&model, &AtomicBool::new(false), u64::MAX).is_none());
    }
    let mut model = Model::new();
    let x = (0..3).map(|_| model.int_range(0, 2)).collect::<Vec<_>>();
    let z = model.int_set(vec![0, 2]);
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { variables: x.clone(), except: Vec::new() }));
    model.add_constraint(Constraint::Intension(IntExpr::Eq(Box::new(z.into()), Box::new(hinge(x[0], x[1], 0)))));
    model.add_objective(Objective::IntExpr { minimize: true, expr: z.into() });
    assert!(OrderingPlan::compile(&model, &AtomicBool::new(false), u64::MAX).is_none());
}
