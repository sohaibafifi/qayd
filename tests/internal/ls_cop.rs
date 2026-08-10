use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use qayd::constraints::linear::Relation;
use qayd::engines::ls::cop::{solve_ls, LocalSearchSpec, LsConfig, MAX_DOMAIN_VALUES};
use qayd::expr::Expr;
use qayd::problem::{Objective, Problem};
use qayd::store::Solver;

/// A search variable whose domain is too large to materialise (> 4096 values)
/// and whose only feasible region sits far from its lower bound must still be
/// moved off that bound: LS finds a feasible incumbent by sampling the range.
#[test]
fn large_range_domain_finds_feasible_incumbent() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 5000);
    assert!(solver.store.size(x) > MAX_DOMAIN_VALUES, "test needs an unmaterialised domain");
    let problem = Problem { solver, search: vec![x], objective: Some(Objective::Var(true, x)) };

    let mut spec = LocalSearchSpec::default();
    spec.add_var(x);
    spec.add_linear(vec![1], vec![x], Relation::Ge, 4500);

    let stop = AtomicBool::new(false);
    let mut incumbent: Option<(i64, Vec<i32>)> = None;
    std::thread::scope(|scope| {
        // Bound the run: cap wall time, but exit promptly once search stops.
        scope.spawn(|| {
            for _ in 0..500 {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            stop.store(true, Ordering::Relaxed);
        });
        let _ = solve_ls(problem, spec, &stop, 1, LsConfig::default(), |value, solution, _| {
            if incumbent.is_none() {
                incumbent = Some((value, solution.to_vec()));
            }
            stop.store(true, Ordering::Relaxed);
        });
        stop.store(true, Ordering::Relaxed);
    });

    let (value, solution) = incumbent.expect("LS never found a feasible incumbent for a >4096 domain");
    assert!(solution[0] >= 4500, "incumbent {solution:?} violates x >= 4500");
    assert_eq!(value, i64::from(solution[0]));
}

#[test]
fn symbolic_affine_objective_guides_guarded_element_construction() {
    let mut solver = Solver::new();
    let cells = [solver.new_var_range(0, 1), solver.new_var_range(0, 1)];
    let guard = solver.new_var_range(0, 1);
    let index = solver.new_var_range(0, 1);
    let letter = solver.new_var_range(0, 1);
    let search = vec![cells[0], cells[1], guard, index, letter];
    let objective = Expr::Add(vec![Expr::Const(3), Expr::Mul(vec![Expr::Const(7), Expr::Var(guard)])]);
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Expr(false, objective)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    spec.add_element(cells.to_vec(), index, letter, 0);
    spec.add_expr(Expr::Or(vec![
        Expr::Eq(Box::new(Expr::Var(guard)), Box::new(Expr::Const(0))),
        Expr::Eq(Box::new(Expr::Var(letter)), Box::new(Expr::Const(1))),
    ]));
    // The real AlteredStates models contain extension constraints between word
    // positions. This harmless table activates the same constructive path.
    spec.add_extension(vec![index], vec![vec![0], vec![1]], true);

    let stop = AtomicBool::new(false);
    let mut incumbent = None;
    let _ = solve_ls(problem, spec, &stop, 0, LsConfig::default(), |value, solution, source| {
        incumbent = Some((value, solution.to_vec(), source));
        stop.store(true, Ordering::Relaxed);
    });

    let (value, solution, source) = incumbent.expect("guarded constructor did not publish an incumbent");
    assert_eq!(value, 10);
    assert_eq!(solution[2], 1);
    assert_eq!(source, "constructive");
}

#[test]
fn guarded_mismatch_counter_allows_one_inexact_letter() {
    let mut solver = Solver::new();
    let cell = solver.new_var_range(0, 0);
    let guard = solver.new_var_range(0, 1);
    let index = solver.new_var_range(0, 0);
    let letter = solver.new_var_range(0, 1);
    let mismatch = solver.new_var_range(0, 1);
    let mismatch_count = solver.new_var_range(0, 1);
    let search = vec![cell, guard, index, letter, mismatch, mismatch_count];
    let objective = Expr::Mul(vec![Expr::Const(7), Expr::Var(guard)]);
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Expr(false, objective)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    spec.add_element(vec![cell], index, letter, 0);
    spec.add_expr(Expr::Eq(Box::new(Expr::Var(mismatch)), Box::new(Expr::Ne(Box::new(Expr::Var(letter)), Box::new(Expr::Const(1))))));
    spec.add_linear(vec![1, -1], vec![mismatch, mismatch_count], Relation::Eq, 0);
    assert!(!spec.has_guarded_mismatch_structure(), "a mismatch counter alone is not a guarded structure");
    spec.add_expr(Expr::Or(vec![
        Expr::Eq(Box::new(Expr::Var(guard)), Box::new(Expr::Const(0))),
        Expr::Le(Box::new(Expr::Var(mismatch_count)), Box::new(Expr::Const(1))),
    ]));
    assert!(spec.has_guarded_mismatch_structure(), "the complete guarded mismatch chain was not detected");
    spec.add_extension(vec![index], vec![vec![0]], true);

    let stop = AtomicBool::new(false);
    let mut incumbent = None;
    let _ = solve_ls(problem, spec, &stop, 0, LsConfig::default(), |value, solution, source| {
        incumbent = Some((value, solution.to_vec(), source));
        stop.store(true, Ordering::Relaxed);
    });

    let (value, solution, source) = incumbent.expect("mismatch-aware constructor did not publish an incumbent");
    assert_eq!(value, 7);
    assert_eq!(solution[1], 1);
    assert_eq!(solution[5], 1);
    assert_eq!(source, "constructive");
}

#[test]
fn guarded_grid_optimization_observes_external_stop() {
    let mut solver = Solver::new();
    let cell = solver.new_var_range(0, 1);
    let guard = solver.new_var_range(0, 1);
    let index = solver.new_var_range(0, 0);
    let letter = solver.new_var_range(0, 1);
    let mismatch = solver.new_var_range(0, 1);
    let mismatch_count = solver.new_var_range(0, 1);
    let activation = solver.new_var_range(0, 0);
    let search = vec![cell, guard, index, letter, mismatch, mismatch_count, activation];
    let objective = Expr::Mul(vec![Expr::Const(7), Expr::Var(guard)]);
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Expr(false, objective)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    spec.add_element(vec![cell], index, letter, 0);
    spec.add_expr(Expr::Eq(Box::new(Expr::Var(mismatch)), Box::new(Expr::Ne(Box::new(Expr::Var(letter)), Box::new(Expr::Const(1))))));
    spec.add_linear(vec![1, -1], vec![mismatch, mismatch_count], Relation::Eq, 0);
    spec.add_expr(Expr::Or(vec![
        Expr::Eq(Box::new(Expr::Var(guard)), Box::new(Expr::Const(0))),
        Expr::Le(Box::new(Expr::Var(mismatch_count)), Box::new(Expr::Const(1))),
    ]));
    spec.add_extension(vec![activation], vec![vec![0]], true);

    let stop = AtomicBool::new(false);
    let started = Instant::now();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(1));
            stop.store(true, Ordering::Relaxed);
        });
        let _ = solve_ls(problem, spec, &stop, 0, LsConfig::default(), |_, _, _| {});
    });
    assert!(started.elapsed() < Duration::from_secs(1), "guarded-grid optimization ignored its stop signal");
}

#[test]
fn guarded_word_branch_and_bound_reuses_cells_for_later_words() {
    let mut solver = Solver::new();
    let cells = [solver.new_var_range(0, 2), solver.new_var_range(0, 2), solver.new_var_range(0, 2)];
    let guards = [solver.new_var_range(0, 1), solver.new_var_range(0, 1), solver.new_var_range(0, 1)];
    let indices = [solver.new_var_range(2, 2), solver.new_var_range(0, 2), solver.new_var_range(0, 0)];
    let letters = [solver.new_var_range(0, 2), solver.new_var_range(0, 2), solver.new_var_range(0, 2)];
    let activation = solver.new_var_range(0, 0);
    let mut search = cells.into_iter().chain(guards).chain(indices).chain(letters).collect::<Vec<_>>();
    search.push(activation);
    let objective = Expr::Add(vec![
        Expr::Mul(vec![Expr::Const(3), Expr::Var(guards[0])]),
        Expr::Mul(vec![Expr::Const(2), Expr::Var(guards[1])]),
        Expr::Var(guards[2]),
    ]);
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Expr(false, objective)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    for ((guard, index), (letter, value)) in guards.into_iter().zip(indices).zip(letters.into_iter().zip([1, 1, 2])) {
        spec.add_element(cells.to_vec(), index, letter, 0);
        spec.add_expr(Expr::Or(vec![
            Expr::Eq(Box::new(Expr::Var(guard)), Box::new(Expr::Const(0))),
            Expr::Eq(Box::new(Expr::Var(letter)), Box::new(Expr::Const(value))),
        ]));
    }
    spec.add_extension(vec![activation], vec![vec![0]], true);

    let stop = AtomicBool::new(false);
    let mut incumbent = None;
    let _ = solve_ls(problem, spec, &stop, 0, LsConfig::default(), |value, solution, source| {
        incumbent = Some((value, solution.to_vec(), source));
        stop.store(true, Ordering::Relaxed);
    });

    let (value, solution, source) = incumbent.expect("branch-and-bound constructor did not publish an incumbent");
    assert_eq!(value, 6);
    assert_eq!(&solution[3..6], &[1, 1, 1]);
    assert_eq!(&solution[6..9], &[2, 2, 0], "the flexible word should reuse the occupied matching cell");
    assert_eq!(&solution[..3], &[2, 0, 1]);
    assert_eq!(source, "constructive");
}

#[test]
fn guarded_grid_lex_canonicalization_remaps_element_indices() {
    let mut solver = Solver::new();
    let cells = [solver.new_var_range(0, 1), solver.new_var_range(0, 1)];
    let guard = solver.new_var_range(0, 1);
    let index = solver.new_var_range(0, 1);
    let letter = solver.new_var_range(0, 1);
    let search = vec![cells[0], cells[1], guard, index, letter];
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Var(false, guard)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    spec.add_element(cells.to_vec(), index, letter, 0);
    spec.add_expr(Expr::Or(vec![
        Expr::Eq(Box::new(Expr::Var(guard)), Box::new(Expr::Const(0))),
        Expr::Eq(Box::new(Expr::Var(letter)), Box::new(Expr::Const(1))),
    ]));
    spec.add_lex_chain(vec![cells.to_vec(), vec![cells[1], cells[0]]], false);
    spec.add_extension(vec![index], vec![vec![0], vec![1]], true);

    let stop = AtomicBool::new(false);
    let mut incumbent = None;
    let _ = solve_ls(problem, spec, &stop, 0, LsConfig::default(), |value, solution, source| {
        incumbent = Some((value, solution.to_vec(), source));
        stop.store(true, Ordering::Relaxed);
    });

    let (value, solution, source) = incumbent.expect("lex-canonical constructor did not publish an incumbent");
    assert_eq!(value, 1);
    assert_eq!(&solution[..2], &[0, 1], "the grid was not moved to its lexicographically least image");
    assert_eq!(solution[3], 1, "the Element index was not remapped by the inverse grid permutation");
    assert_eq!(solution[4], 1, "the remapped Element no longer reads the requested letter");
    assert_eq!(source, "constructive");
}

#[test]
fn guarded_grid_lex_keeps_inactive_element_indices_at_their_sentinel() {
    let mut solver = Solver::new();
    let cells = [solver.new_var_range(0, 1), solver.new_var_range(0, 1)];
    let guards = [solver.new_var_range(0, 1), solver.new_var_range(0, 1)];
    let indices = [solver.new_var_range(0, 1), solver.new_var_range(0, 0)];
    let letters = [solver.new_var_range(0, 1), solver.new_var_range(0, 1)];
    let search = cells.into_iter().chain(guards).chain(indices).chain(letters).collect::<Vec<_>>();
    let objective = Expr::Add(vec![Expr::Mul(vec![Expr::Const(2), Expr::Var(guards[0])]), Expr::Var(guards[1])]);
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Expr(false, objective)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    for ((guard, index), (letter, value)) in guards.into_iter().zip(indices).zip(letters.into_iter().zip([1, 2])) {
        spec.add_element(cells.to_vec(), index, letter, 0);
        spec.add_expr(Expr::Or(vec![
            Expr::Eq(Box::new(Expr::Var(guard)), Box::new(Expr::Const(0))),
            Expr::Eq(Box::new(Expr::Var(letter)), Box::new(Expr::Const(value))),
        ]));
    }
    spec.add_expr(Expr::Or(vec![Expr::Var(guards[1]), Expr::Eq(Box::new(Expr::Var(indices[1])), Box::new(Expr::Const(0)))]));
    spec.add_lex_chain(vec![cells.to_vec(), vec![cells[1], cells[0]]], false);
    spec.add_extension(vec![indices[0]], vec![vec![0], vec![1]], true);

    let stop = AtomicBool::new(false);
    let mut incumbent = None;
    let _ = solve_ls(problem, spec, &stop, 0, LsConfig::default(), |value, solution, source| {
        incumbent = Some((value, solution.to_vec(), source));
        stop.store(true, Ordering::Relaxed);
    });

    let (value, solution, source) = incumbent.expect("lex-canonical constructor did not publish an incumbent");
    assert_eq!(value, 2);
    assert_eq!(&solution[..2], &[0, 1]);
    assert_eq!(solution[4], 1, "the selected word index was not remapped");
    assert_eq!(solution[5], 0, "the inactive word index left its sentinel");
    assert_eq!(source, "constructive");
}

#[test]
fn malformed_grid_lex_is_not_deferred() {
    let mut solver = Solver::new();
    let cells = [solver.new_var_range(0, 1), solver.new_var_range(0, 1)];
    let guard = solver.new_var_range(0, 1);
    let index = solver.new_var_range(0, 0);
    let letter = solver.new_var_range(0, 1);
    let search = vec![cells[0], cells[1], guard, index, letter];
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Var(false, guard)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    spec.add_element(cells.to_vec(), index, letter, 0);
    spec.add_expr(Expr::Or(vec![
        Expr::Eq(Box::new(Expr::Var(guard)), Box::new(Expr::Const(0))),
        Expr::Eq(Box::new(Expr::Var(letter)), Box::new(Expr::Const(1))),
    ]));
    spec.add_lex_chain(vec![cells.to_vec(), vec![cells[1], cells[1]]], false);
    spec.add_extension(vec![index], vec![vec![0]], true);

    let stop = AtomicBool::new(false);
    let mut incumbent = None;
    let _ = solve_ls(problem, spec, &stop, 0, LsConfig::default(), |value, solution, source| {
        incumbent = Some((value, solution.to_vec(), source));
        stop.store(true, Ordering::Relaxed);
    });

    let (value, solution, source) = incumbent.expect("fallback constructor did not publish its feasible minimum");
    assert_eq!(value, 0, "a non-bijective Lex row was incorrectly treated as a symmetry");
    assert_eq!(solution[2], 0);
    assert_eq!(source, "constructive");
}

#[test]
fn guarded_grid_materialization_preserves_nonzero_index_domains() {
    let mut solver = Solver::new();
    let cells = [solver.new_var_range(0, 0), solver.new_var_range(0, 0)];
    let guard = solver.new_var_range(0, 1);
    let index = solver.new_var_range(1, 1);
    let letter = solver.new_var_range(0, 1);
    let activation = solver.new_var_range(0, 0);
    let search = vec![cells[0], cells[1], guard, index, letter, activation];
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Var(false, guard)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    spec.add_element(cells.to_vec(), index, letter, 0);
    spec.add_expr(Expr::Or(vec![
        Expr::Eq(Box::new(Expr::Var(guard)), Box::new(Expr::Const(0))),
        Expr::Eq(Box::new(Expr::Var(letter)), Box::new(Expr::Const(1))),
    ]));
    spec.add_extension(vec![activation], vec![vec![0]], true);

    let stop = AtomicBool::new(false);
    let mut incumbent = None;
    let _ = solve_ls(problem, spec, &stop, 0, LsConfig::default(), |value, solution, _| {
        incumbent = Some((value, solution.to_vec()));
        stop.store(true, Ordering::Relaxed);
    });

    let (value, solution) = incumbent.expect("guarded-grid fallback did not publish an incumbent");
    assert_eq!(value, 0);
    assert_eq!(solution[3], 1, "materialization wrote an out-of-domain zero to the Element index");
}
