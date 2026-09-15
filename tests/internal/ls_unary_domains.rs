use std::sync::atomic::AtomicBool;

use crate::constraints::table::STAR;
use crate::engines::ls::cop::{audit_local_domains, solve_ls_capped, LocalSearchSpec, LsConfig, MAX_DOMAIN_VALUES};
use crate::problem::{Objective, Problem};
use crate::{Solver, VarId};

fn fixture(values: &[i32], tuples: Vec<Vec<i32>>, positive: bool) -> (Problem, LocalSearchSpec, VarId) {
    let mut solver = Solver::new();
    let variable = solver.new_var_set(values);
    let mut spec = LocalSearchSpec::default();
    spec.add_var(variable);
    spec.add_extension(vec![variable], tuples, positive);
    let problem = Problem { solver, search: vec![variable], objective: Some(Objective::Var(true, variable)) };
    (problem, spec, variable)
}

#[test]
fn unary_table_domains_match_independent_sparse_oracle() {
    let original = [-3, -1, 0, 2, 5];
    let tables = [vec![], vec![-3], vec![2, 2], vec![STAR], vec![STAR, 99], vec![-100, 99], vec![-3, 2, 5]];
    for positive in [false, true] {
        for literals in &tables {
            let expected = original
                .iter()
                .copied()
                .filter(|value| literals.iter().any(|literal| *literal == STAR || literal == value) == positive)
                .collect::<Vec<_>>();
            let (problem, spec, _) = fixture(&original, literals.iter().map(|&value| vec![value]).collect(), positive);
            let domains = audit_local_domains(&problem, spec.clone(), &AtomicBool::new(false));
            if expected.is_empty() {
                assert!(domains.is_none());
            } else {
                assert_eq!(domains.unwrap(), vec![expected.clone()]);
            }
            let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 1, |objective, values, _| {
                assert!(expected.contains(&values[0]));
                assert_eq!(objective, i64::from(values[0]));
            });
            assert_eq!(outcome.best.is_some(), !expected.is_empty());
            assert_eq!(outcome.constraints, 1, "domain filtering must retain the original predicate");
        }
    }
}

#[test]
fn negative_unary_tables_make_the_first_random_assignment_feasible() {
    let allowed = [2, 5, 8];
    let forbidden = (0..10).filter(|value| !allowed.contains(value)).map(|value| vec![value]).collect::<Vec<_>>();
    let mut solver = Solver::new();
    let variables = (0..24).map(|_| solver.new_var_range(0, 9)).collect::<Vec<_>>();
    let mut spec = LocalSearchSpec::default();
    for &variable in &variables {
        spec.add_var(variable);
        spec.add_extension(vec![variable], forbidden.clone(), false);
    }
    let problem =
        Problem { solver, search: variables.clone(), objective: Some(Objective::Linear(true, vec![1; variables.len()], variables)) };
    for seed in [0, 1, 42] {
        let mut first = None;
        let outcome = solve_ls_capped(
            problem.clone(),
            spec.clone(),
            &AtomicBool::new(false),
            seed,
            LsConfig::default(),
            1,
            |objective, values, source| {
                assert_eq!(source, "local-search", "negative unary tables use the existing random initializer");
                assert!(values.iter().all(|value| allowed.contains(value)));
                assert_eq!(objective, values.iter().map(|&value| i64::from(value)).sum::<i64>());
                first = Some(values.to_vec());
            },
        );
        let first = first.expect("the first random assignment must already satisfy every unary prohibition");
        assert_eq!(outcome.iterations, 1);
        let repeated = solve_ls_capped(problem.clone(), spec.clone(), &AtomicBool::new(false), seed, LsConfig::default(), 1, |_, _, _| {});
        assert_eq!(repeated.best.unwrap().0, first, "filtered initialization must preserve seeded reproducibility");
    }
}

#[test]
fn unary_table_intersections_filter_before_mutable_selection() {
    let (problem, mut spec, variable) = fixture(&[0, 1, 2, 3, 4], vec![vec![0], vec![2], vec![4]], true);
    spec.add_extension(vec![variable], vec![vec![0], vec![4]], false);
    assert_eq!(audit_local_domains(&problem, spec.clone(), &AtomicBool::new(false)).unwrap(), vec![vec![2]]);
    let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 17, LsConfig::default(), 1, |_, _, _| {});
    assert_eq!(outcome.best.unwrap(), (vec![2], 2));
    assert_eq!(outcome.iterations, 0, "a domain narrowed to one value is no longer mutable");
    assert_eq!(outcome.constraints, 2);
}

#[test]
fn contradictory_unary_tables_return_no_incumbent() {
    let (problem, mut spec, variable) = fixture(&[0, 1], vec![vec![0]], true);
    spec.add_extension(vec![variable], vec![vec![0]], false);
    let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 4, |_, _, _| {
        panic!("contradictory unary domains cannot emit an incumbent");
    });
    assert!(outcome.best.is_none());
    assert_eq!(outcome.iterations, 0);
}

#[test]
fn unary_star_preserves_table_wildcard_semantics() {
    for positive in [false, true] {
        let (problem, spec, _) = fixture(&[i32::MIN, -2, 7], vec![vec![STAR]], positive);
        let domains = audit_local_domains(&problem, spec, &AtomicBool::new(false));
        if positive {
            assert_eq!(domains.unwrap(), vec![vec![i32::MIN, -2, 7]]);
        } else {
            assert!(domains.is_none(), "a negative STAR prohibits every value");
        }
    }
}

#[test]
fn selected_unary_tables_do_not_filter_unconditionally() {
    for active in [0, 1] {
        for positive in [false, true] {
            let mut solver = Solver::new();
            let variable = solver.new_var_range(0, 3);
            let selector = solver.new_var_range(active, active);
            let mut spec = LocalSearchSpec::default();
            spec.add_var(variable);
            spec.add_var(selector);
            let start = spec.begin_guarded_constraints();
            spec.add_extension(vec![variable], vec![vec![2]], positive);
            spec.finish_guarded_constraints(start, selector);
            let problem = Problem { solver, search: vec![variable, selector], objective: Some(Objective::Var(true, variable)) };
            assert_eq!(audit_local_domains(&problem, spec.clone(), &AtomicBool::new(false)).unwrap()[0], vec![0, 1, 2, 3]);
            let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 16, |_, values, _| {
                if active == 1 {
                    assert_eq!(values[0] == 2, positive);
                }
            });
            assert!(outcome.best.is_some());
        }
    }
}

#[test]
fn malformed_unary_rows_skip_domain_filtering_conservatively() {
    for positive in [false, true] {
        for malformed in [vec![], vec![0, 7]] {
            let (problem, spec, _) = fixture(&[0, 1, 2, 3], vec![vec![2], malformed], positive);
            assert_eq!(audit_local_domains(&problem, spec, &AtomicBool::new(false)).unwrap(), vec![vec![0, 1, 2, 3]]);
        }
    }
}

#[test]
fn wide_range_domains_keep_their_original_table_predicates() {
    for positive in [false, true] {
        let mut solver = Solver::new();
        let upper = MAX_DOMAIN_VALUES as i32 + 7;
        let variable = solver.new_var_range(0, upper);
        let mut spec = LocalSearchSpec::default();
        spec.add_var(variable);
        spec.add_extension(vec![variable], vec![vec![upper]], positive);
        let problem = Problem { solver, search: vec![variable], objective: Some(Objective::Var(true, variable)) };
        assert_eq!(audit_local_domains(&problem, spec.clone(), &AtomicBool::new(false)).unwrap(), vec![Vec::<i32>::new()]);
        let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 8, |_, values, _| {
            assert_eq!(values[0] == upper, positive);
        });
        assert_eq!(outcome.constraints, 1);
        assert!(outcome.best.is_some());
    }
}

#[test]
fn unary_filtering_checks_derived_output_domains() {
    let mut solver = Solver::new();
    let index = solver.new_var_range(0, 3);
    let target = solver.new_var_range(0, 3);
    let mut spec = LocalSearchSpec::default();
    spec.add_var(index);
    spec.add_var(target);
    spec.add_element_const(vec![0, 1, 2, 3], index, target, 0);
    spec.add_extension(vec![target], vec![vec![2]], true);
    let problem = Problem { solver, search: vec![index, target], objective: Some(Objective::Var(true, target)) };
    assert_eq!(audit_local_domains(&problem, spec.clone(), &AtomicBool::new(false)).unwrap(), vec![vec![0, 1, 2, 3], vec![2]]);
    let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(false), 0, LsConfig::default(), 16, |objective, values, _| {
        assert_eq!(values, [2, 2]);
        assert_eq!(objective, 2);
    });
    assert_eq!(outcome.best.unwrap().1, 2);
}

#[test]
fn unary_domain_construction_honors_prearmed_stop() {
    let (problem, spec, _) = fixture(&[0, 1, 2], vec![vec![0]], false);
    assert!(audit_local_domains(&problem, spec.clone(), &AtomicBool::new(true)).is_none());
    let outcome = solve_ls_capped(problem, spec, &AtomicBool::new(true), 0, LsConfig::default(), 8, |_, _, _| {
        panic!("cancelled construction must not publish an incumbent");
    });
    assert!(outcome.best.is_none());
    assert_eq!(outcome.iterations, 0);
}
