use std::cell::Cell;
use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;

use crate::constraints::sum_squares::fixed_sum_squares_interruptible;
use crate::lcg::lit::LitOrConst;
use crate::lcg::trail::{Cdcl, Reason};
use crate::lcg::view::Tri;
use crate::model::{CompiledCp, Constraint, IntExpr, IntVarRef, Model, Objective, Relation};
use crate::orchestrator::{compile_cp_plan, solve_cp_plan, IgnoreEvents, SolveBudget, SolveMode, SolveRequest, SolveStatus};
use crate::problem::Objective as PhysicalObjective;
use crate::{minimize, solve_search, SearchControl, Solver, VarId};

fn post(solver: &mut Solver, variables: &[VarId], objective: VarId, total: i128) {
    assert!(fixed_sum_squares_interruptible(solver, variables, objective, total, &AtomicBool::new(false)));
}

fn assignments(domains: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let mut rows = vec![Vec::new()];
    for domain in domains {
        rows = rows
            .into_iter()
            .flat_map(|row| {
                domain.iter().map(move |&value| {
                    let mut extended = row.clone();
                    extended.push(value);
                    extended
                })
            })
            .collect();
    }
    rows
}

fn square_sum(values: &[i32]) -> i32 {
    values.iter().map(|&value| value * value).sum()
}

fn squares(variables: &[IntVarRef]) -> IntExpr {
    IntExpr::Add(variables.iter().map(|&v| IntExpr::Mul(vec![IntExpr::Variable(v), IntExpr::Variable(v)])).collect())
}

fn square_model(domains: &[Vec<i32>], total: i64) -> Model {
    let mut model = Model::new();
    let variables = domains.iter().map(|domain| model.int_set(domain.clone())).collect::<Vec<_>>();
    model.add_constraint(Constraint::Linear { terms: variables.iter().map(|&v| (1, v)).collect(), relation: Relation::Eq, rhs: total });
    model.add_objective(Objective::IntExpr { minimize: true, expr: squares(&variables) });
    model
}

fn compile(model: &Model) -> CompiledCp {
    CompiledCp::compile_interruptible(model, &AtomicBool::new(false)).unwrap().unwrap()
}

#[test]
fn interval_lower_bound_is_exact_against_exhaustive_integer_assignments() {
    let boxes = [(-3, -1), (-2, 2), (-1, 0), (0, 0), (0, 3), (2, 4)];
    for &(alo, ahi) in &boxes {
        for &(blo, bhi) in &boxes {
            for &(clo, chi) in &boxes {
                let domains = vec![(alo..=ahi).collect(), (blo..=bhi).collect(), (clo..=chi).collect()];
                let all = assignments(&domains);
                for total in (alo + blo + clo - 1)..=(ahi + bhi + chi + 1) {
                    let expected = all.iter().filter(|row| row.iter().sum::<i32>() == total).map(|row| square_sum(row)).min();
                    let mut solver = Solver::new();
                    let variables = domains.iter().map(|domain| solver.new_var_set(domain)).collect::<Vec<_>>();
                    let objective = solver.new_var_range(0, 100);
                    post(&mut solver, &variables, objective, i128::from(total));
                    match expected {
                        Some(expected) => {
                            solver.propagate().unwrap();
                            assert_eq!(solver.store.min(objective), expected, "domains={domains:?}, total={total}");
                        }
                        None => assert!(solver.propagate().is_err(), "domains={domains:?}, total={total}"),
                    }
                }
            }
        }
    }
}

#[test]
fn sparse_domains_and_objective_cutoffs_preserve_the_complete_solution_set() {
    let candidate_domains = [vec![-3, -1, 2], vec![-2, 0, 3], vec![0, 1, 3], vec![-2, 2]];
    for case in 0..96 {
        let domains = vec![candidate_domains[case % 4].clone(), candidate_domains[(case / 4) % 4].clone(), vec![-2, 0, 1, 3]];
        let total = (case as i32 % 11) - 4;
        let objective_domain = (0..=27).filter(|value| value % 3 != case as i32 % 3 && *value <= 8 + case as i32 % 20).collect::<Vec<_>>();
        let expected = assignments(&domains)
            .into_iter()
            .filter(|row| row.iter().sum::<i32>() == total && objective_domain.contains(&square_sum(row)))
            .map(|mut row| {
                row.push(square_sum(&row));
                row
            })
            .collect::<BTreeSet<_>>();
        let mut solver = Solver::new();
        let mut variables = domains.iter().map(|domain| solver.new_var_set(domain)).collect::<Vec<_>>();
        let objective = solver.new_var_set(&objective_domain);
        post(&mut solver, &variables, objective, i128::from(total));
        variables.push(objective);
        let mut actual = BTreeSet::new();
        solve_search(&mut solver, &variables, |solver| {
            actual.insert(variables.iter().map(|&variable| solver.store.value(variable)).collect::<Vec<_>>());
            SearchControl::Continue
        });
        assert_eq!(actual, expected, "case={case}, domains={domains:?}, total={total}");
    }
}

#[test]
fn balanced_load_bound_and_cutoff_are_exact() {
    let mut solver = Solver::new();
    let variables = (0..50).map(|_| solver.new_var_range(2, 100)).collect::<Vec<_>>();
    let objective = solver.new_var_range(0, 500_000);
    post(&mut solver, &variables, objective, 1_315);
    solver.propagate().unwrap();
    assert_eq!(solver.store.min(objective), 34_595);
    solver.store.remove_above(objective, 34_595).unwrap();
    solver.propagate().unwrap();
    assert!(variables.iter().all(|&variable| solver.store.min(variable) == 26 && solver.store.max(variable) == 27));
}

#[test]
fn backtracking_restores_cutoff_filtering_and_exact_objective_link() {
    let mut solver = Solver::new();
    let left = solver.new_var_range(0, 4);
    let right = solver.new_var_range(0, 4);
    let objective = solver.new_var_range(0, 32);
    post(&mut solver, &[left, right], objective, 4);
    solver.propagate().unwrap();
    assert_eq!(solver.store.min(objective), 8);

    solver.store.push_level();
    solver.store.remove_above(objective, 8).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.value(left), 2);
    assert_eq!(solver.store.value(right), 2);
    solver.store.pop_level();
    assert_eq!((solver.store.min(left), solver.store.max(left)), (0, 4));

    solver.store.push_level();
    solver.store.fix(left, 0).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.value(right), 4);
    assert_eq!(solver.store.value(objective), 16);
    solver.store.pop_level();
    assert_eq!(minimize(&mut solver, &[left, right, objective], objective).unwrap().1, 8);
}

#[test]
fn empty_scope_and_numeric_extremes_are_safe() {
    for (total, feasible) in [(0, true), (1, false)] {
        let mut solver = Solver::new();
        let objective = solver.new_var_range(-1, 1);
        post(&mut solver, &[], objective, total);
        assert_eq!(solver.propagate().is_ok(), feasible);
        if feasible {
            assert_eq!(solver.store.value(objective), 0);
        }
    }
    let mut solver = Solver::new();
    let left = solver.new_var_range(i32::MIN, i32::MAX);
    let right = solver.new_var_range(i32::MIN, i32::MAX);
    let objective = solver.new_var_range(0, i32::MAX);
    post(&mut solver, &[left, right], objective, 0);
    solver.propagate().unwrap();
    assert_eq!(solver.store.min(objective), 0);
    assert!(solver.store.min(left) >= -46_340 && solver.store.max(left) <= 46_340);

    let mut solver = Solver::new();
    let left = solver.new_var_range(i32::MIN, i32::MIN);
    let right = solver.new_var_range(i32::MAX, i32::MAX);
    let objective = solver.new_var_range(0, i32::MAX);
    post(&mut solver, &[left, right], objective, -1);
    assert!(solver.propagate().is_err());
}

#[test]
fn registration_and_water_filling_obey_interruption() {
    let mut solver = Solver::new();
    let variables = (0..64).map(|_| solver.new_var_range(-10_000, 10_000)).collect::<Vec<_>>();
    let objective = solver.new_var_range(0, i32::MAX);
    assert!(!fixed_sum_squares_interruptible(&mut solver, &variables, objective, 0, &AtomicBool::new(true)));
    post(&mut solver, &variables, objective, 0);
    let polls = Cell::new(0);
    solver
        .propagate_until(|| {
            polls.set(polls.get() + 1);
            polls.get() >= 180
        })
        .unwrap();
    assert!(polls.get() >= 180);
    assert_eq!(solver.store.min(objective), 0);
    assert_eq!(solver.store.max(objective), i32::MAX);
    assert!(solver.fd_at_fixpoint());
}

#[test]
fn compiler_materializes_only_supported_unconditional_square_sums() {
    let baseline = square_model(&[vec![-2, 0, 3], vec![-1, 1, 4]], 2);
    let compiled = compile(&baseline);
    let PhysicalObjective::Var(true, objective) = compiled.objectives()[0] else {
        panic!("fixed-sum square objective was not materialized");
    };
    assert!(compiled.problem().search.contains(&objective));
    assert_eq!(compiled.problem().search.len(), baseline.int_vars.len() + 1);

    let mut duplicate_linear = baseline.clone();
    duplicate_linear.constraints[0] =
        Constraint::Linear { terms: vec![(3, IntVarRef(0)), (-2, IntVarRef(0)), (1, IntVarRef(1))], relation: Relation::Eq, rhs: 2 };
    assert!(matches!(compile(&duplicate_linear).objectives()[0], PhysicalObjective::Var(true, _)));
    let mut negative_total = baseline.clone();
    negative_total.constraints[0] =
        Constraint::Linear { terms: vec![(-1, IntVarRef(0)), (-1, IntVarRef(1))], relation: Relation::Eq, rhs: -2 };
    assert!(matches!(compile(&negative_total).objectives()[0], PhysicalObjective::Var(true, _)));

    let mut variants = Vec::new();
    let mut repeated = baseline.clone();
    repeated.objectives[0] = Objective::IntExpr { minimize: true, expr: squares(&[IntVarRef(0), IntVarRef(0), IntVarRef(1)]) };
    variants.push(repeated);
    let mut selected = baseline.clone();
    let selector = selected.bool_var();
    selected.constraints[0] = Constraint::Selected { selector, constraint: Box::new(selected.constraints[0].clone()) };
    variants.push(selected);
    let mut inequality = baseline.clone();
    let Constraint::Linear { relation, .. } = &mut inequality.constraints[0] else { unreachable!() };
    *relation = Relation::Le;
    variants.push(inequality);
    let mut weighted = baseline.clone();
    let Constraint::Linear { terms, .. } = &mut weighted.constraints[0] else { unreachable!() };
    terms[0].0 = 2;
    variants.push(weighted);
    let mut different_scope = baseline.clone();
    let extra = different_scope.int_range(0, 2);
    let Constraint::Linear { terms, .. } = &mut different_scope.constraints[0] else { unreachable!() };
    terms[0].1 = extra;
    variants.push(different_scope);
    let mut maximizing = baseline.clone();
    maximizing.objectives[0] = Objective::IntExpr { minimize: false, expr: squares(&[IntVarRef(0), IntVarRef(1)]) };
    variants.push(maximizing);
    let mut nonsquare = baseline;
    nonsquare.objectives[0] = Objective::IntExpr {
        minimize: true,
        expr: IntExpr::Add(vec![IntExpr::Mul(vec![IntExpr::Variable(IntVarRef(0)), IntExpr::Variable(IntVarRef(1))])]),
    };
    variants.push(nonsquare);
    variants.push(square_model(&[vec![46_341], vec![0]], 46_341));
    variants.push(square_model(&[vec![-32_768], vec![32_768]], 0));
    variants.push(square_model(&[vec![i32::MIN], vec![i32::MAX]], -1));
    for (index, model) in variants.iter().enumerate() {
        assert!(matches!(compile(model).objectives()[0], PhysicalObjective::Expr(_, _)), "unsupported variant {index} materialized");
    }
}

#[test]
fn compiled_optimization_and_semantic_replay_match_brute_force() {
    let domains = vec![vec![-3, -1, 1, 3], vec![-2, 0, 2], vec![-1, 2, 4]];
    let rows = assignments(&domains);
    for total in -6..=9 {
        let expected = rows.iter().filter(|row| row.iter().sum::<i32>() == total).map(|row| square_sum(row)).min();
        let model = square_model(&domains, i64::from(total));
        let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };
        let budget = SolveBudget::new(None);
        let plan = compile_cp_plan(&model, &request, &budget).unwrap();
        let result = solve_cp_plan(&model, &plan, &request, &budget, &mut IgnoreEvents).unwrap();
        match expected {
            Some(expected) => {
                assert_eq!(result.status(), SolveStatus::Optimal);
                assert_eq!(result.primal().unwrap().objectives(), [i64::from(expected)]);
            }
            None => assert_eq!(result.status(), SolveStatus::Unsatisfiable),
        }
    }
}

#[test]
fn square_objective_materialization_survives_lexicographic_tier_fixing() {
    let mut model = square_model(&[(0..=3).collect(), (0..=3).collect()], 3);
    model.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Variable(IntVarRef(0)) });
    let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };
    let budget = SolveBudget::new(None);
    let plan = compile_cp_plan(&model, &request, &budget).unwrap();
    let result = solve_cp_plan(&model, &plan, &request, &budget, &mut IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [5, 2]);
}

#[test]
fn prevalidation_estimate_rejects_invalid_references_without_panicking() {
    let mut model = square_model(&[vec![0, 1]], 1);
    model.objectives[0] = Objective::IntExpr { minimize: true, expr: squares(&[IntVarRef(5)]) };
    model.constraints[0] = Constraint::Linear { terms: vec![(1, IntVarRef(5))], relation: Relation::Eq, rhs: 1 };
    assert!(CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).is_err());
}

#[test]
fn materialized_objective_estimate_includes_the_dense_auxiliary_domain() {
    let model = square_model(&[(0..=45).collect(), (0..=45).collect()], 45);
    let mut unmaterialized = model.clone();
    unmaterialized.objectives[0] = Objective::IntExpr { minimize: false, expr: squares(&[IntVarRef(0), IntVarRef(1)]) };
    let stop = AtomicBool::new(false);
    let materialized_bytes = CompiledCp::estimate_semantic_bytes_interruptible(&model, &stop).unwrap();
    let unmaterialized_bytes = CompiledCp::estimate_semantic_bytes_interruptible(&unmaterialized, &stop).unwrap();
    assert!(materialized_bytes > unmaterialized_bytes + 90_000, "dense derived-domain and inference storage was omitted");
}

#[test]
fn local_search_reports_the_semantic_square_objective() {
    let model = square_model(&[(0..=3).collect(), (0..=3).collect()], 3);
    let mut request = SolveRequest { mode: SolveMode::LocalSearch, ..SolveRequest::default() };
    request.limits.iterations = Some(32);
    let budget = SolveBudget::new(Some(std::time::Duration::from_secs(2)));
    let plan = compile_cp_plan(&model, &request, &budget).unwrap();
    let result = solve_cp_plan(&model, &plan, &request, &budget, &mut IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.primal().unwrap().objectives(), [5]);
}

#[test]
fn local_search_materialized_square_is_a_functional_with_canonical_scores() {
    use crate::engines::ls::cop::{solve_ls_capped, LsConfig};
    use crate::engines::ls::integer;

    for values in [[0, 3], [-3, 2], [-32_767, 32_767], [0, 46_340]] {
        let model = square_model(&[vec![values[0]], vec![values[1]]], values.iter().map(|&value| i64::from(value)).sum());
        let compiled = compile(&model);
        let stop = AtomicBool::new(false);
        let local = integer::compile_interruptible(&model, &compiled, &stop, u64::MAX).unwrap().unwrap();
        assert!(integer::compile_interruptible(&model, &compiled, &stop, local.estimated_bytes - 1).unwrap().is_none());
        let mut candidates = 0;
        let outcome =
            solve_ls_capped(compiled.problem().clone(), local.spec, &stop, 0, LsConfig::default(), 2, |objective, solution, _| {
                candidates += 1;
                assert_eq!(objective, i64::from(square_sum(&solution[..2])), "LS optimized a detached physical objective");
            });
        assert!(candidates > 0);
        assert_eq!(outcome.best.unwrap().1, i64::from(square_sum(&values)));
        assert_eq!(outcome.functionals, 1, "square objective must be derived from the semantic variables");
    }
}

#[test]
fn lcg_cutoff_reason_includes_the_objective_bound() {
    let mut solver = Solver::new();
    let left = solver.new_var_range(0, 4);
    let right = solver.new_var_range(0, 4);
    let objective = solver.new_var_range(0, 32);
    post(&mut solver, &[left, right], objective, 4);
    let mut cdcl = Cdcl::new(&mut solver, &[left, right, objective]);
    assert!(cdcl.init());
    let LitOrConst::Lit(above_cutoff) = cdcl.atoms.ge(objective, 9) else { panic!("expected cutoff atom") };
    cdcl.decide(above_cutoff.negate()).unwrap();
    cdcl.propagate().unwrap();
    assert_eq!(cdcl.solver.store.value(left), 2);
    let LitOrConst::Lit(bound) = cdcl.atoms.ge(left, 2) else { panic!("expected inferred bound atom") };
    let Reason::Generic(reason) = cdcl.reason_of(bound.atom()) else { panic!("expected whole-scope propagator reason") };
    assert!(reason.iter().all(|&premise| cdcl.tvalue(premise) == Tri::True));
    assert!(reason.iter().any(|&premise| cdcl.atoms.var_of(premise.atom()) == objective));
}
