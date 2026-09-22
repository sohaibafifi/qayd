use std::sync::atomic::AtomicBool;

use crate::constraints::linear::Relation;
use crate::engines::ls::cop::{audit_invariant_transactions, solve_ls_capped, LocalSearchSpec, LsConfig};
use crate::engines::ls::integer;
use crate::expr::Expr;
use crate::model::{CompiledCp, Constraint, IntGlobalConstraint, Model, Objective as SemanticObjective};
use crate::orchestrator::{compile_cp_plan, solve_cp_plan, IgnoreEvents, SolveBudget, SolveMode, SolveRequest, SolveStatus};
use crate::problem::{Objective, Problem};
use crate::{Solver, VarId};

fn spec_for(variables: &[VarId]) -> LocalSearchSpec {
    let mut spec = LocalSearchSpec::default();
    for &variable in variables {
        spec.add_var(variable);
    }
    spec
}

#[test]
fn constant_element_matches_fixed_assignment_oracle_with_offsets_and_extremes() {
    let array = [i32::MIN, -7, 3, 3];
    for start in [i32::MIN, -3, 0, i32::MAX - 1] {
        let mut indices = vec![i32::MIN, i32::MAX, -4, 0, 5];
        indices.extend((-1..=4).filter_map(|delta| start.checked_add(delta)));
        indices.sort_unstable();
        indices.dedup();
        for index_value in indices {
            for allowed in [vec![i32::MIN, -7, 3, 99], vec![-7, 3], vec![99]] {
                let mut solver = Solver::new();
                let index = solver.new_var_range(index_value, index_value);
                let target = solver.new_var_set(&allowed);
                let mut spec = spec_for(&[index, target]);
                spec.add_element_const(array.to_vec(), index, target, start);
                assert!(spec.is_derived(target));
                let problem = Problem { solver, search: vec![index, target], objective: Some(Objective::Var(true, target)) };
                let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 1, |_, _, _| {});
                let expected = array
                    .iter()
                    .enumerate()
                    .find(|(position, _)| i64::from(start) + *position as i64 == i64::from(index_value))
                    .map(|(_, &value)| value)
                    .filter(|value| allowed.contains(value));
                assert_eq!(outcome.best.as_ref().map(|(values, _)| values[1]), expected, "start={start}, index={index_value}");
                if let Some((values, objective)) = outcome.best {
                    assert_eq!(values[0], index_value);
                    assert_eq!(objective, i64::from(expected.unwrap()));
                }
                assert_eq!(outcome.functionals, 1);
                assert_eq!(outcome.constraints, 1);
            }
        }
    }
}

#[test]
fn constant_element_empty_array_and_literal_star_are_not_supports() {
    for array in [Vec::new(), vec![i32::MIN]] {
        let mut solver = Solver::new();
        let index = solver.new_var_range(0, 0);
        let target = solver.new_var_range(7, 7);
        let mut spec = spec_for(&[index, target]);
        spec.add_element_const(array, index, target, 0);
        let problem = Problem { solver, search: vec![index, target], objective: Some(Objective::Var(true, target)) };
        let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 1, |_, _, _| {});
        assert!(outcome.best.is_none(), "constant i32::MIN must retain its literal value instead of acting as a table STAR");
    }
}

#[test]
fn constant_element_sparse_domains_emit_only_exact_objectives() {
    let mut solver = Solver::new();
    let index = solver.new_var_set(&[-3, -1, 0, 2, 4]);
    let target = solver.new_var_set(&[-5, 4]);
    let mut spec = spec_for(&[index, target]);
    let array = [7, 4, -5, 9, 4];
    spec.add_element_const(array.to_vec(), index, target, -2);
    let problem = Problem { solver, search: vec![index, target], objective: Some(Objective::Var(true, target)) };
    let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 32, |objective, values, _| {
        let expected = array[usize::try_from(values[0] + 2).unwrap()];
        assert_eq!(values[1], expected);
        assert_eq!(objective, i64::from(expected));
    });
    assert_eq!(outcome.best.unwrap().1, -5);
}

#[test]
fn constant_element_incremental_chain_updates_and_rolls_back_objective() {
    let mut solver = Solver::new();
    let index = solver.new_var_range(0, 2);
    let middle = solver.new_var_range(0, 2);
    let target = solver.new_var_set(&[-4, 2, 5]);
    let independent = solver.new_var_range(0, 3);
    let mut spec = spec_for(&[index, middle, target, independent]);
    // Deliberately reverse the posting order to exercise dependency sorting.
    spec.add_element_const(vec![5, -4, 2], middle, target, 0);
    spec.add_element_const(vec![2, 0, 1], index, middle, 0);
    let problem = Problem {
        solver,
        search: vec![index, middle, target, independent],
        objective: Some(Objective::Linear(true, vec![3, 7], vec![target, independent])),
    };
    let audit = audit_invariant_transactions(&problem, spec, vec![0, 2, 2, 0], &[(index, 1), (independent, 2), (index, 2)])
        .expect("lookup chain trial, rollback, and commit must agree with full recomputation");
    assert_eq!(audit.trials, 3);
    assert_eq!(audit.rollbacks, 3);
    assert_eq!(audit.commits, 3);
    assert_eq!(audit.functional_evaluations, 8, "two index moves each visit two functionals in trial and commit");
    assert_eq!(audit.objective_evaluations, 6);
    assert_eq!(audit.objective_full_evaluations, 1);
    assert_eq!(audit.full_rebuilds, 1);
}

#[test]
fn constant_element_unchanged_output_stops_dependency_work() {
    let mut solver = Solver::new();
    let index = solver.new_var_range(0, 1);
    let middle = solver.new_var_range(0, 1);
    let target = solver.new_var_range(2, 7);
    let mut spec = spec_for(&[index, middle, target]);
    spec.add_element_const(vec![0, 0], index, middle, 0);
    spec.add_element_const(vec![2, 7], middle, target, 0);
    let problem = Problem { solver, search: vec![index, middle, target], objective: Some(Objective::Var(true, target)) };
    let audit = audit_invariant_transactions(&problem, spec, vec![0, 0, 2], &[(index, 1)]).expect("unchanged lookup output drift");
    assert_eq!(audit.functional_evaluations, 2);
    assert_eq!(audit.objective_evaluations, 0);
    assert_eq!(audit.full_rebuilds, 1);
}

#[test]
fn constant_element_invalid_lookup_and_excluded_output_roll_back() {
    for excluded_output in [false, true] {
        let mut solver = Solver::new();
        let index = solver.new_var_range(0, 1);
        let target = solver.new_var_range(0, 0);
        let mut spec = spec_for(&[index, target]);
        spec.add_element_const(if excluded_output { vec![0, 1] } else { vec![0] }, index, target, 0);
        let problem = Problem { solver, search: vec![index, target], objective: Some(Objective::Var(true, target)) };
        let audit = audit_invariant_transactions(&problem, spec, vec![0, 0], &[(index, 1), (index, 0)])
            .expect("invalid constant-element values must not leave stale valid state after rollback or repair");
        assert_eq!(audit.trials, 2);
        assert_eq!(audit.rollbacks, 2);
        assert_eq!(audit.commits, 2);
        assert_eq!(audit.full_rebuilds, 1);
    }
}

#[test]
fn constant_element_alias_checks_fixed_points_without_self_dependency() {
    let array = [0, 2, 2];
    for value in -1..=3 {
        let mut solver = Solver::new();
        let variable = solver.new_var_range(value, value);
        let mut spec = spec_for(&[variable]);
        spec.add_element_const(array.to_vec(), variable, variable, 0);
        assert!(!spec.is_derived(variable));
        let problem = Problem { solver, search: vec![variable], objective: Some(Objective::Var(true, variable)) };
        let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 1, |_, _, _| {});
        assert_eq!(outcome.best.is_some(), [0, 2].contains(&value));
        assert_eq!(outcome.functionals, 0);
        assert_eq!(outcome.unsupported, 0);
    }
}

#[test]
fn constant_element_multiple_definitions_retain_every_relation() {
    for left in 0..=1 {
        for right in 0..=1 {
            let mut solver = Solver::new();
            let first = solver.new_var_range(left, left);
            let second = solver.new_var_range(right, right);
            let target = solver.new_var_range(0, 1);
            let mut spec = spec_for(&[first, second, target]);
            spec.add_element_const(vec![0, 1], first, target, 0);
            spec.add_element_const(vec![1, 1], second, target, 0);
            let problem = Problem { solver, search: vec![first, second, target], objective: Some(Objective::Var(true, target)) };
            let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 1, |_, _, _| {});
            assert_eq!(outcome.best.is_some(), left == 1);
            assert_eq!(outcome.functionals, 1, "second definitions must not install another producer");
            assert_eq!(outcome.constraints, 2);
        }
    }
}

#[test]
fn constant_element_respects_existing_expression_producer() {
    for value in 0..=1 {
        let mut solver = Solver::new();
        let index = solver.new_var_range(value, value);
        let target = solver.new_var_range(0, 1);
        let mut spec = spec_for(&[index, target]);
        spec.add_expr(Expr::Eq(Box::new(Expr::Var(target)), Box::new(Expr::Var(index))));
        spec.add_element_const(vec![1, 1], index, target, 0);
        let problem = Problem { solver, search: vec![index, target], objective: Some(Objective::Var(true, target)) };
        let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 1, |_, _, _| {});
        assert_eq!(outcome.best.is_some(), value == 1);
        assert_eq!(outcome.functionals, 1);
    }
}

#[test]
fn constant_element_cycles_fall_back_to_searchable_exact_relations() {
    for other_definition in 0..3 {
        let mut solver = Solver::new();
        let left = solver.new_var_range(0, 1);
        let right = solver.new_var_range(0, 1);
        let mut spec = spec_for(&[left, right]);
        spec.add_element_const(vec![0, 1], left, right, 0);
        match other_definition {
            0 => spec.add_element_const(vec![0, 1], right, left, 0),
            1 => spec.add_expr(Expr::Eq(Box::new(Expr::Var(left)), Box::new(Expr::Var(right)))),
            _ => spec.add_linear(vec![-1, 1], vec![left, right], Relation::Eq, 0),
        }
        let problem = Problem { solver, search: vec![left, right], objective: Some(Objective::Var(true, left)) };
        let audit = audit_invariant_transactions(&problem, spec.clone(), vec![0, 0], &[(right, 1), (right, 0)])
            .expect("a demoted lookup must preserve incremental checks in pure and mixed cycles");
        assert_eq!(audit.functional_evaluations, 4);
        let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 16, |objective, values, _| {
            assert_eq!(values[0], values[1]);
            assert_eq!(objective, i64::from(values[0]));
        });
        assert_eq!(outcome.best.unwrap().1, 0);
        assert_eq!(outcome.unsupported, 0);
    }
}

#[test]
fn constant_element_later_expression_and_linear_definitions_keep_both_constraints() {
    for linear in [false, true] {
        for index_value in 0..=1 {
            let mut solver = Solver::new();
            let index = solver.new_var_range(index_value, index_value);
            let target = solver.new_var_range(0, 1);
            let mut spec = spec_for(&[index, target]);
            spec.add_element_const(vec![1, 0], index, target, 0);
            if linear {
                spec.add_linear(vec![1], vec![target], Relation::Eq, 0);
            } else {
                spec.add_expr(Expr::Eq(Box::new(Expr::Var(target)), Box::new(Expr::Const(0))));
            }
            let problem = Problem { solver, search: vec![index, target], objective: Some(Objective::Var(true, target)) };
            let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 1, |_, _, _| {});
            assert_eq!(outcome.best.is_some(), index_value == 1);
            assert_eq!(outcome.unsupported, 0, "a later producer must not turn a supported lookup into a rejected model");
            assert_eq!(outcome.constraints, 2);
        }
    }
}

#[test]
fn constant_element_guarded_relation_does_not_derive_unconditionally() {
    for active in 0..=1 {
        let mut solver = Solver::new();
        let index = solver.new_var_range(-1, -1);
        let target = solver.new_var_range(7, 7);
        let selector = solver.new_var_range(active, active);
        let mut spec = spec_for(&[index, target, selector]);
        let start = spec.begin_guarded_constraints();
        spec.add_element_const(vec![0], index, target, 0);
        spec.finish_guarded_constraints(start, selector);
        assert!(!spec.is_derived(target));
        let problem = Problem { solver, search: vec![index, target, selector], objective: Some(Objective::Var(true, target)) };
        let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 1, |_, _, _| {});
        assert_eq!(outcome.best.is_some(), active == 0);
        assert_eq!(outcome.functionals, 0);
    }
}

#[test]
fn constant_element_semantic_lowering_derives_cost_with_budget_and_stop_gates() {
    let mut model = Model::new();
    let index = model.int_set(vec![1, 3]);
    let target = model.int_set(vec![0, 1, 2]);
    let array = vec![2, 0, 2, 1];
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::ElementConst { array: array.clone(), index, value: target }));
    model.add_objective(SemanticObjective::IntExpr { minimize: true, expr: target.into() });
    let stop = AtomicBool::new(false);
    let compiled = CompiledCp::compile_interruptible(&model, &stop).unwrap().unwrap();
    let local = integer::compile_interruptible(&model, &compiled, &stop, u64::MAX).unwrap().unwrap();
    let physical_target = compiled.int_variables()[target.0];
    assert!(local.spec.is_derived(physical_target));
    assert!(integer::compile_interruptible(&model, &compiled, &stop, local.estimated_bytes).unwrap().is_some());
    assert!(integer::compile_interruptible(&model, &compiled, &stop, local.estimated_bytes - 1).unwrap().is_none());
    assert!(integer::compile_interruptible(&model, &compiled, &AtomicBool::new(true), u64::MAX).unwrap().is_none());
    let index_position = compiled.problem().search.iter().position(|&v| v == compiled.int_variables()[index.0]).unwrap();
    let target_position = compiled.problem().search.iter().position(|&v| v == physical_target).unwrap();
    let outcome = solve_ls_capped(compiled.problem().clone(), local.spec, &stop, 0, LsConfig::default(), 16, |objective, values, _| {
        let expected = array[values[index_position] as usize];
        assert_eq!(values[target_position], expected);
        assert_eq!(objective, i64::from(expected));
    });
    assert_eq!(outcome.best.unwrap().1, 0);
    assert_eq!(outcome.functionals, 1);
}

#[test]
fn constant_element_prearmed_search_stop_emits_no_candidate() {
    let mut solver = Solver::new();
    let index = solver.new_var_range(0, 1);
    let target = solver.new_var_range(0, 2);
    let mut spec = spec_for(&[index, target]);
    spec.add_element_const(vec![0, 2], index, target, 0);
    let problem = Problem { solver, search: vec![index, target], objective: Some(Objective::Var(true, target)) };
    let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(true), 0, LsConfig::default(), 8, |_, _, _| {
        panic!("prearmed stop must not emit an incumbent");
    });
    assert!(outcome.best.is_none());
    assert_eq!(outcome.iterations, 0);
}

#[test]
fn constant_element_semantic_cycles_and_duplicate_producers_preserve_repair_ownership() {
    for other_definition in 0..5 {
        let mut model = Model::new();
        let left = model.int_range(0, 1);
        let right = model.int_range(0, 1);
        model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::ElementConst { array: vec![0, 1], index: left, value: right }));
        model.add_constraint(match other_definition {
            0 => Constraint::IntegerGlobal(IntGlobalConstraint::ElementConst { array: vec![0, 1], index: right, value: left }),
            1 => Constraint::Intension(crate::model::IntExpr::Eq(Box::new(left.into()), Box::new(right.into()))),
            2 => Constraint::Linear { terms: vec![(-1, left), (1, right)], relation: crate::model::Relation::Eq, rhs: 0 },
            3 => Constraint::Intension(crate::model::IntExpr::Eq(Box::new(right.into()), Box::new(0.into()))),
            _ => Constraint::Linear { terms: vec![(1, right)], relation: crate::model::Relation::Eq, rhs: 0 },
        });
        model.add_objective(SemanticObjective::IntExpr { minimize: true, expr: left.into() });
        let stop = AtomicBool::new(false);
        let compiled = CompiledCp::compile_interruptible(&model, &stop).unwrap().unwrap();
        let local = integer::compile_interruptible(&model, &compiled, &stop, u64::MAX).unwrap().unwrap();
        assert_eq!(
            [left, right].iter().filter(|v| local.spec.is_derived(compiled.int_variables()[v.0])).count(),
            1,
            "the stored spec must expose the same single producer retained for execution"
        );
        let mut normalized_again = local.spec.clone();
        assert!(normalized_again.normalize_functionals());
        for variable in [left, right] {
            let physical = compiled.int_variables()[variable.0];
            assert_eq!(normalized_again.is_derived(physical), local.spec.is_derived(physical), "normalization must be idempotent");
        }
        let mut request = SolveRequest { mode: SolveMode::LocalSearch, ..SolveRequest::default() };
        request.limits.iterations = Some(16);
        let budget = SolveBudget::new(Some(std::time::Duration::from_secs(2)));
        let plan = compile_cp_plan(&model, &request, &budget).unwrap();
        let result = solve_cp_plan(&model, &plan, &request, &budget, &mut IgnoreEvents).unwrap();
        assert_eq!(result.status(), SolveStatus::Satisfiable);
        assert_eq!(result.primal().unwrap().objectives(), [0]);
    }
}

#[test]
fn constant_element_normalization_honors_stop_and_invalidates_after_new_producer() {
    let left = VarId(0);
    let right = VarId(1);
    let mut spec = spec_for(&[left, right]);
    spec.add_element_const(vec![0, 1], left, right, 0);
    assert!(!spec.normalize_functionals_interruptible(&AtomicBool::new(true)));
    assert_eq!(spec.unsupported(), 0, "prearmed cancellation must leave the pending spec untouched");
    assert!(spec.normalize_functionals_interruptible(&AtomicBool::new(false)));
    assert!(!spec.is_derived(left));
    assert!(spec.is_derived(right));
    assert!(!spec.normalize_functionals_interruptible(&AtomicBool::new(true)), "cached normalization must still honor stop");

    // A later producer closes the cycle, so the cached ordering is no longer
    // applicable and the existing lookup must return to an exact predicate.
    spec.add_expr(Expr::Eq(Box::new(Expr::Var(left)), Box::new(Expr::Var(right))));
    assert!(spec.normalize_functionals_interruptible(&AtomicBool::new(false)));
    assert!(spec.is_derived(left));
    assert!(!spec.is_derived(right));
    assert_eq!(spec.unsupported(), 0);
}

#[test]
fn duplicate_expression_outputs_match_a_fixed_assignment_oracle() {
    for x_value in 0..=2 {
        for y_value in 0..=2 {
            for output in 0..=2 {
                let mut solver = Solver::new();
                let x = solver.new_var_range(x_value, x_value);
                let y = solver.new_var_range(y_value, y_value);
                let target = solver.new_var_range(output, output);
                let mut spec = spec_for(&[x, y, target]);
                for input in [x, y] {
                    spec.add_expr(Expr::Eq(Box::new(Expr::Var(target)), Box::new(Expr::Add(vec![Expr::Var(input), Expr::Const(0)]))));
                }
                assert!(spec.normalize_functionals());
                assert!(!spec.is_derived(target), "ambiguous expression outputs must remain searchable");
                assert!(spec.normalize_functionals(), "normalization must remain idempotent");
                let problem = Problem { solver, search: vec![x, y, target], objective: Some(Objective::Var(true, target)) };
                let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 1, |_, _, _| {});
                let feasible = output == x_value && output == y_value;
                assert_eq!(outcome.best.is_some(), feasible, "values=({x_value},{y_value},{output})");
                assert_eq!(outcome.constraints, 2, "both defining equations must still be checked");
                assert_eq!(outcome.functionals, 0);
                assert_eq!(outcome.unsupported, 0);
                if let Some((values, objective)) = outcome.best {
                    assert_eq!(values, [x_value, y_value, output]);
                    assert_eq!(objective, i64::from(output));
                }
            }
        }
    }
}

#[test]
fn duplicate_expression_outputs_preserve_canonical_repair_ownership() {
    let mut model = Model::new();
    let x = model.int_range(0, 2);
    let y = model.int_range(1, 1);
    let target = model.int_range(0, 2);
    for input in [x, y] {
        model.add_constraint(Constraint::Intension(crate::model::IntExpr::Eq(
            Box::new(target.into()),
            Box::new(crate::model::IntExpr::Add(vec![input.into(), 0.into()])),
        )));
    }
    model.add_objective(SemanticObjective::IntExpr { minimize: true, expr: target.into() });
    let stop = AtomicBool::new(false);
    let compiled = CompiledCp::compile_interruptible(&model, &stop).unwrap().unwrap();
    let local = integer::compile_interruptible(&model, &compiled, &stop, u64::MAX).unwrap().unwrap();
    assert!(!local.spec.is_derived(compiled.int_variables()[target.0]));
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        limits: crate::orchestrator::SolveLimits { iterations: Some(64), ..crate::orchestrator::SolveLimits::default() },
        ..SolveRequest::default()
    };
    let budget = SolveBudget::new(Some(std::time::Duration::from_secs(2)));
    let plan = compile_cp_plan(&model, &request, &budget).unwrap();
    let result = solve_cp_plan(&model, &plan, &request, &budget, &mut IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.primal().unwrap().assignment().integers, [Some(1), Some(1), Some(1)]);
    assert_eq!(result.primal().unwrap().objectives(), [1]);
    assert!(result.proof().is_none());
}

#[test]
fn expression_dependency_cycles_report_unsupported_instead_of_memory_exhaustion() {
    let mut model = Model::new();
    let x = model.int_range(0, 1);
    let y = model.int_range(0, 1);
    for (target, input) in [(x, y), (y, x)] {
        model.add_constraint(Constraint::Intension(crate::model::IntExpr::Eq(
            Box::new(target.into()),
            Box::new(crate::model::IntExpr::Add(vec![input.into(), 0.into()])),
        )));
    }
    model.add_objective(SemanticObjective::IntExpr { minimize: true, expr: x.into() });
    let request = SolveRequest { mode: SolveMode::LocalSearch, ..SolveRequest::default() };
    let result = compile_cp_plan(&model, &request, &SolveBudget::new(None));
    assert!(
        matches!(result, Err(crate::orchestrator::SolveError::Unsupported(message)) if message.contains("cycle") && !message.contains("memory"))
    );
}

#[test]
fn duplicate_variable_element_outputs_remain_unsupported() {
    let left = VarId(0);
    let right = VarId(1);
    let index = VarId(2);
    let target = VarId(3);
    let mut spec = spec_for(&[left, right, index, target]);
    // Variable elements own their defining relation, so removing either
    // producer without reconstructing a constraint would weaken this model.
    spec.add_element(vec![left], index, target, 0);
    spec.add_element(vec![right], index, target, 0);
    assert!(!spec.normalize_functionals());
    assert_eq!(spec.unsupported(), 1);
}
