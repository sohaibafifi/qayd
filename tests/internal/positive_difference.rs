use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;

use crate::constraints::intension::{intension, intension_interruptible};
use crate::lcg::lit::LitOrConst;
use crate::lcg::trail::{Cdcl, Reason};
use crate::lcg::view::Tri;
use crate::model::{CompiledCp, Constraint, IntExpr, Model};
use crate::{optimize_with, solve_search, Expr, SearchControl, Solver, VarId};

fn expression(target: VarId, left: VarId, right: VarId, form: usize) -> Expr {
    let difference = Expr::Sub(Box::new(Expr::Var(left)), Box::new(Expr::Var(right)));
    let value = match form % 4 {
        0 => Expr::IfThenElse(
            Box::new(Expr::Lt(Box::new(Expr::Var(left)), Box::new(Expr::Var(right)))),
            Box::new(Expr::Const(0)),
            Box::new(difference),
        ),
        1 => Expr::IfThenElse(
            Box::new(Expr::Le(Box::new(Expr::Var(left)), Box::new(Expr::Var(right)))),
            Box::new(Expr::Const(0)),
            Box::new(difference),
        ),
        2 => Expr::Max(vec![Expr::Const(0), difference]),
        _ => Expr::Max(vec![difference, Expr::Const(0)]),
    };
    if form < 4 {
        Expr::Eq(Box::new(Expr::Var(target)), Box::new(value))
    } else {
        Expr::Eq(Box::new(value), Box::new(Expr::Var(target)))
    }
}

fn assignments(domains: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let mut result = vec![Vec::new()];
    for domain in domains {
        result = result
            .into_iter()
            .flat_map(|prefix| {
                domain.iter().map(move |&value| {
                    let mut assignment = prefix.clone();
                    assignment.push(value);
                    assignment
                })
            })
            .collect();
    }
    result
}

#[test]
fn all_native_forms_and_aliases_match_exhaustive_sparse_signed_assignments() {
    let domains = [vec![-3, -1, 0, 1, 3], vec![-2, 0, 2], vec![-1, 0, 2, 4], vec![0, 1, 3]];
    let patterns = [(2, 0, 1), (0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)];
    for case in 0..16 {
        let declarations = vec![domains[case % 4].clone(), domains[(case / 4) % 4].clone(), domains[(case + 2) % 4].clone()];
        let all = assignments(&declarations);
        for (target, left, right) in patterns {
            let expected = all
                .iter()
                .filter(|assignment| i64::from(assignment[target]) == (i64::from(assignment[left]) - i64::from(assignment[right])).max(0))
                .cloned()
                .collect::<BTreeSet<_>>();
            for form in 0..8 {
                let mut solver = Solver::new();
                let variables = declarations.iter().map(|domain| solver.new_var_set(domain)).collect::<Vec<_>>();
                intension(&mut solver, expression(variables[target], variables[left], variables[right], form));
                let mut actual = BTreeSet::new();
                solve_search(&mut solver, &variables, |solver| {
                    actual.insert(variables.iter().map(|&variable| solver.store.value(variable)).collect::<Vec<_>>());
                    SearchControl::Continue
                });
                assert_eq!(actual, expected, "case={case}, aliases={target}/{left}/{right}, form={form}");
            }
        }
    }
}

#[test]
fn output_cutoffs_and_sparse_jumps_propagate_to_a_backtrackable_fixpoint() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 6);
    let y = solver.new_var_range(0, 6);
    let z = solver.new_var_range(0, 6);
    intension(&mut solver, expression(z, x, y, 0));
    solver.propagate().unwrap();
    solver.store.push_level();
    solver.store.fix(z, 3).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.min(x), 3);
    assert_eq!(solver.store.max(y), 3);
    solver.store.push_level();
    solver.store.fix(y, 2).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.value(x), 5);
    solver.store.pop_level();
    solver.store.push_level();
    solver.store.fix(y, 3).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.value(x), 6);
    solver.store.pop_level();
    solver.store.pop_level();
    solver.store.fix(z, 0).unwrap();
    solver.store.fix(x, 4).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.min(y), 4);

    let mut solver = Solver::new();
    let x = solver.new_var_set(&[1, 4, 7]);
    let y = solver.new_var_set(&[0, 3]);
    let z = solver.new_var_set(&[2, 3]);
    intension(&mut solver, expression(z, x, y, 0));
    assert!(solver.propagate().is_err(), "sparse endpoint jumps must be reflected in the other domains");
}

#[test]
fn extreme_differences_and_aliased_outputs_use_widened_arithmetic() {
    for (x_value, y_value, expected) in [(i32::MIN, i32::MAX, Some(0)), (i32::MAX, i32::MIN, None)] {
        let mut solver = Solver::new();
        let x = solver.new_var_range(x_value, x_value);
        let y = solver.new_var_range(y_value, y_value);
        let z = solver.new_var_range(0, i32::MAX);
        intension(&mut solver, expression(z, x, y, 0));
        match expected {
            Some(expected) => {
                solver.propagate().unwrap();
                assert_eq!(solver.store.value(z), expected);
            }
            None => assert!(solver.propagate().is_err()),
        }
    }
    for value in [i32::MAX, i32::MAX - 1] {
        let mut solver = Solver::new();
        let x = solver.new_var_range(value, value);
        let y = solver.new_var_range(0, i32::MAX);
        intension(&mut solver, expression(y, x, y, 0));
        if value % 2 == 0 {
            solver.propagate().unwrap();
            assert_eq!(solver.store.value(y), value / 2);
        } else {
            assert!(solver.propagate().is_err());
        }
    }
    let mut solver = Solver::new();
    let x = solver.new_var_range(1, i32::MAX);
    let y = solver.new_var_range(i32::MIN, i32::MAX);
    intension(&mut solver, expression(x, x, y, 0));
    solver.propagate().unwrap();
    assert_eq!(solver.store.value(y), 0);
}

#[test]
fn native_lowering_keeps_semantic_variables_and_rejects_near_matches() {
    let mut model = Model::new();
    let x = model.int_range(-3, 3);
    let y = model.int_range(-3, 3);
    let z = model.int_range(-3, 3);
    let make = |condition_left, then_value| {
        Constraint::Intension(IntExpr::Eq(
            Box::new(IntExpr::Variable(z)),
            Box::new(IntExpr::IfThenElse(
                Box::new(IntExpr::Lt(Box::new(IntExpr::Variable(condition_left)), Box::new(IntExpr::Variable(y)))),
                Box::new(IntExpr::Constant(then_value)),
                Box::new(IntExpr::Sub(Box::new(IntExpr::Variable(x)), Box::new(IntExpr::Variable(y)))),
            )),
        ))
    };
    for (constraint, native) in [(make(x, 0), true), (make(y, 0), false), (make(x, 1), false)] {
        let mut model = model.clone();
        model.add_constraint(constraint);
        let compiled = CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).unwrap().unwrap();
        let stats = compiled.compilation_stats();
        assert_eq!(stats.native_intensions, u64::from(native));
        assert_eq!(stats.fallback_intensions, u64::from(!native));
        assert_eq!(compiled.problem().solver.store.num_vars(), 3, "positive-difference lowering must not introduce an auxiliary");
    }
}

#[test]
fn short_circuiting_i64_constant_branches_are_not_rewritten() {
    let mut model = Model::new();
    let target = model.int_range(0, 0);
    model.add_constraint(Constraint::Intension(IntExpr::Eq(
        Box::new(IntExpr::Variable(target)),
        Box::new(IntExpr::IfThenElse(
            Box::new(IntExpr::Lt(Box::new(IntExpr::Constant(i64::MIN)), Box::new(IntExpr::Constant(i64::MAX)))),
            Box::new(IntExpr::Constant(0)),
            Box::new(IntExpr::Sub(Box::new(IntExpr::Constant(i64::MIN)), Box::new(IntExpr::Constant(i64::MAX)))),
        )),
    )));
    let compiled = CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).unwrap().unwrap();
    assert_eq!(compiled.compilation_stats().native_intensions, 0);
    let mut solver = compiled.problem().solver.clone();
    solver.propagate().unwrap();
}

#[test]
fn selector_guards_remain_part_of_the_native_constraint() {
    for enabled in [0, 1] {
        let mut solver = Solver::new();
        let selector = solver.new_var_range(enabled, enabled);
        let x = solver.new_var_range(3, 3);
        let y = solver.new_var_range(1, 1);
        let z = solver.new_var_range(0, 0);
        solver.set_selector(Some(selector));
        intension(&mut solver, expression(z, x, y, 0));
        solver.set_selector(None);
        assert_eq!(solver.propagate().is_ok(), enabled == 0);
    }
}

#[test]
fn lcg_positive_output_reason_uses_the_output_bound() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 4);
    let y = solver.new_var_range(0, 4);
    let z = solver.new_var_range(0, 4);
    intension(&mut solver, expression(z, x, y, 0));
    let mut cdcl = Cdcl::new(&mut solver, &[x, y, z]);
    assert!(cdcl.init());
    let LitOrConst::Lit(positive) = cdcl.atoms.ge(z, 3) else { panic!("expected output-bound atom") };
    cdcl.decide(positive).unwrap();
    cdcl.propagate().unwrap();
    assert_eq!(cdcl.solver.store.min(x), 3);
    assert_eq!(cdcl.solver.store.max(y), 1);
    let LitOrConst::Lit(inferred) = cdcl.atoms.ge(x, 3) else { panic!("expected input-bound atom") };
    let Reason::Generic(reason) = cdcl.reason_of(inferred.atom()) else { panic!("expected whole-scope reason") };
    assert!(reason.iter().all(|&premise| cdcl.tvalue(premise) == Tri::True));
    assert!(reason.iter().any(|&premise| cdcl.atoms.var_of(premise.atom()) == z));
}

#[test]
fn native_aliases_optimize_correctly_with_both_lcg_reason_modes() {
    for (target, left, right) in [(2, 0, 1), (0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)] {
        for force_scope in [false, true] {
            for minimizing in [false, true] {
                let domains = vec![vec![-2, 0, 1, 3], vec![-1, 0, 2, 3], vec![0, 1, 3]];
                let mut possible = assignments(&domains)
                    .into_iter()
                    .filter(|assignment| assignment[target] == (assignment[left] - assignment[right]).max(0))
                    .map(|assignment| assignment[target])
                    .collect::<Vec<_>>();
                possible.sort_unstable();
                if !minimizing {
                    possible.reverse();
                }
                let mut solver = Solver::new();
                solver.set_force_scope_reasons(force_scope);
                let variables = domains.iter().map(|domain| solver.new_var_set(domain)).collect::<Vec<_>>();
                intension(&mut solver, expression(variables[target], variables[left], variables[right], 0));
                let (best, _) = optimize_with(&mut solver, &variables, variables[target], minimizing, &AtomicBool::new(false), |_| {});
                assert_eq!(best.map(|(_, value)| value), possible.first().copied());
            }
        }
    }
}

#[test]
fn positive_difference_posting_obeys_a_prearmed_stop() {
    let mut solver = Solver::new();
    let variables = (0..3).map(|_| solver.new_var_range(0, 4)).collect::<Vec<_>>();
    assert!(!intension_interruptible(&mut solver, expression(variables[2], variables[0], variables[1], 0), &AtomicBool::new(true)));
    assert_eq!(solver.num_propagators(), 0);
}
