use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use qayd::constraints::linear::Relation;
use qayd::engines::ls::cop::{
    audit_invariant_transactions, solve_ls, solve_ls_capped, solve_ls_capped_borrowed, LocalSearchSpec, LsConfig, MAX_DOMAIN_VALUES,
};
use qayd::expr::Expr;
use qayd::ids::VarId;
use qayd::problem::{Objective, Problem};
use qayd::store::Solver;

struct SignedProductSquareFixture {
    problem: Problem,
    spec: LocalSearchSpec,
    signs: Vec<VarId>,
    aggregates: Vec<VarId>,
}

#[test]
fn integer_invariant_graph_deltas_rollback_and_commit_exactly() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 5);
    let y = solver.new_var_range(0, 5);
    let z = solver.new_var_range(0, 5);
    let independent = solver.new_var_range(0, 5);
    let problem =
        Problem { solver, search: vec![x, y, z, independent], objective: Some(Objective::Linear(true, vec![3, 7], vec![z, independent])) };
    let mut spec = LocalSearchSpec::default();
    for variable in [x, y, z, independent] {
        spec.add_var(variable);
    }
    // Deliberately post the chain backwards. Graph compilation must establish
    // x -> y -> z before any speculative move is scored.
    spec.add_expr(Expr::Eq(Box::new(Expr::Var(z)), Box::new(Expr::Var(y))));
    spec.add_expr(Expr::Eq(Box::new(Expr::Var(y)), Box::new(Expr::Var(x))));
    spec.add_linear(vec![1], vec![z], Relation::Ge, 0);
    spec.add_linear(vec![1], vec![independent], Relation::Ge, 0);

    let moves = [(x, 4), (independent, 3), (x, 1), (independent, 5)];
    let audit = audit_invariant_transactions(&problem, spec, vec![0; 4], &moves).expect("invariant transaction drift");

    assert_eq!(audit.trials, moves.len() as u64);
    assert_eq!(audit.rollbacks, audit.trials);
    assert_eq!(audit.commits, audit.trials);
    assert_eq!(audit.full_rebuilds, 1, "speculative moves must not trigger full rebuilds");
    assert_eq!(audit.functional_evaluations, 8, "only the two x moves should traverse the two-node functional chain");
    assert_eq!(audit.constraint_evaluations, 16, "each transaction should refresh only its dependent constraints");
    assert_eq!(audit.objective_evaluations, 8);
    assert_eq!(audit.objective_full_evaluations, 1, "affine trials must update the objective by delta");
}

#[test]
fn integer_invariant_graph_rolls_back_invalid_derived_values() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let derived = solver.new_var_range(0, 0);
    let problem = Problem { solver, search: vec![x, derived], objective: Some(Objective::Var(true, derived)) };
    let mut spec = LocalSearchSpec::default();
    spec.add_var(x);
    spec.add_var(derived);
    spec.add_expr(Expr::Eq(Box::new(Expr::Var(derived)), Box::new(Expr::Var(x))));

    let moves = [(x, 1), (x, 0)];
    let audit = audit_invariant_transactions(&problem, spec, vec![0, 0], &moves).expect("invalid functional rollback drift");

    assert_eq!(audit.trials, 2);
    assert_eq!(audit.rollbacks, 2);
    assert_eq!(audit.commits, 2);
    assert_eq!(audit.full_rebuilds, 1);
}

#[test]
fn integer_invariant_graph_work_is_proportional_to_the_affected_subgraph() {
    const CHAINS: usize = 64;
    let mut solver = Solver::new();
    let decisions = (0..CHAINS).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
    let derived = (0..CHAINS).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
    let mut search = decisions.clone();
    search.extend_from_slice(&derived);
    let problem = Problem { solver, search, objective: Some(Objective::Linear(true, vec![1; CHAINS], derived.clone())) };
    let mut spec = LocalSearchSpec::default();
    for variable in decisions.iter().chain(&derived) {
        spec.add_var(*variable);
    }
    for (&decision, &target) in decisions.iter().zip(&derived) {
        spec.add_expr(Expr::Eq(Box::new(Expr::Var(target)), Box::new(Expr::Var(decision))));
    }

    let audit = audit_invariant_transactions(&problem, spec, vec![0; CHAINS * 2], &[(decisions[37], 1)])
        .expect("independent invariant subgraph drift");

    assert_eq!(audit.functional_evaluations, 2, "trial plus commit should touch one functional, not all 64");
    assert_eq!(audit.constraint_evaluations, 2, "trial plus commit should touch one defining constraint, not all 64");
    assert_eq!(audit.objective_evaluations, 2);
    assert_eq!(audit.objective_full_evaluations, 1, "the 64-term objective should use its compiled delta after initialization");
}

#[test]
fn integer_invariant_graph_stores_direct_arcs_and_stops_on_unchanged_outputs() {
    const CHAIN: usize = 512;
    let mut solver = Solver::new();
    let variables = (0..CHAIN).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
    let problem = Problem { solver, search: variables.clone(), objective: Some(Objective::Var(true, variables[CHAIN - 1])) };
    let mut spec = LocalSearchSpec::default();
    for variable in &variables {
        spec.add_var(*variable);
    }
    spec.add_expr(Expr::Eq(Box::new(Expr::Var(variables[1])), Box::new(Expr::Mul(vec![Expr::Const(0), Expr::Var(variables[0])]))));
    for pair in variables[1..].windows(2) {
        spec.add_expr(Expr::Eq(Box::new(Expr::Var(pair[1])), Box::new(Expr::Var(pair[0]))));
    }

    let audit = audit_invariant_transactions(&problem, spec, vec![0; CHAIN], &[(variables[0], 1)])
        .expect("unchanged functional output transaction drift");

    assert_eq!(audit.functional_arcs, CHAIN - 1, "a chain must be stored as direct arcs, not as a quadratic transitive closure");
    assert_eq!(audit.functional_evaluations, 2, "trial plus commit should stop at the first unchanged derived value");
    assert_eq!(audit.objective_evaluations, 0, "an unchanged objective dependency must not be rescored");
    assert_eq!(audit.objective_full_evaluations, 1);
}

#[test]
fn integer_invariant_objective_uses_affine_deltas_and_safe_fallback() {
    let mut affine_solver = Solver::new();
    let affine_var = affine_solver.new_var_range(0, 3);
    let affine_objective = Expr::Add(vec![Expr::Const(5), Expr::Mul(vec![Expr::Const(3), Expr::Var(affine_var)])]);
    let affine_problem =
        Problem { solver: affine_solver, search: vec![affine_var], objective: Some(Objective::Expr(false, affine_objective)) };
    let mut affine_spec = LocalSearchSpec::default();
    affine_spec.add_var(affine_var);
    let moves = [(affine_var, 2), (affine_var, 1)];
    let affine = audit_invariant_transactions(&affine_problem, affine_spec, vec![0], &moves).expect("max affine objective delta drift");
    assert_eq!(affine.objective_evaluations, 4);
    assert_eq!(affine.objective_full_evaluations, 1);

    let mut nonlinear_solver = Solver::new();
    let nonlinear_var = nonlinear_solver.new_var_range(0, 3);
    let nonlinear_objective = Expr::Mul(vec![Expr::Var(nonlinear_var), Expr::Var(nonlinear_var)]);
    let nonlinear_problem =
        Problem { solver: nonlinear_solver, search: vec![nonlinear_var], objective: Some(Objective::Expr(true, nonlinear_objective)) };
    let mut nonlinear_spec = LocalSearchSpec::default();
    nonlinear_spec.add_var(nonlinear_var);
    let nonlinear_moves = [(nonlinear_var, 2), (nonlinear_var, 1)];
    let nonlinear = audit_invariant_transactions(&nonlinear_problem, nonlinear_spec, vec![0], &nonlinear_moves)
        .expect("nonlinear objective fallback drift");
    assert_eq!(nonlinear.objective_evaluations, 4);
    assert_eq!(nonlinear.objective_full_evaluations, 5);
}

fn unit_sum_values(terms: usize) -> Vec<i32> {
    let width = i32::try_from(terms).unwrap();
    (-width..=width).step_by(2).collect()
}

fn signed_product_square_problem(
    size: usize,
    groups: &[Vec<(usize, usize)>],
    aggregate_domains: Option<&[Vec<i32>]>,
    coefficient_override: Option<(usize, usize, i64)>,
) -> SignedProductSquareFixture {
    let mut solver = Solver::new();
    let mut spec = LocalSearchSpec::default();
    let signs = (0..size).map(|_| solver.new_var_set(&[-1, 1])).collect::<Vec<_>>();
    let mut search = signs.clone();
    for &variable in &signs {
        spec.add_var(variable);
    }

    let mut product_by_edge = HashMap::new();
    let mut aggregates = Vec::new();
    for (group_index, group) in groups.iter().enumerate() {
        let mut products = Vec::new();
        for &(left, right) in group {
            assert!(left < size && right < size);
            let edge = if left <= right { (left, right) } else { (right, left) };
            let product = *product_by_edge.entry(edge).or_insert_with(|| {
                let product = solver.new_var_set(&[-1, 1]);
                spec.add_var(product);
                spec.add_expr(Expr::Eq(
                    Box::new(Expr::Var(product)),
                    Box::new(Expr::Mul(vec![Expr::Var(signs[edge.0]), Expr::Var(signs[edge.1])])),
                ));
                search.push(product);
                product
            });
            products.push(product);
        }
        let domain =
            aggregate_domains.and_then(|domains| domains.get(group_index)).cloned().unwrap_or_else(|| unit_sum_values(group.len()));
        let aggregate = solver.new_var_set(&domain);
        spec.add_var(aggregate);
        let mut coefficients = (0..products.len())
            .map(|term| {
                coefficient_override.filter(|&(group, position, _)| group == group_index && position == term).map_or(1, |(_, _, c)| c)
            })
            .collect::<Vec<_>>();
        coefficients.push(-1);
        products.push(aggregate);
        spec.add_linear(coefficients, products, Relation::Eq, 0);
        aggregates.push(aggregate);
        search.push(aggregate);
    }

    let objective = Expr::Add(aggregates.iter().map(|&aggregate| Expr::Mul(vec![Expr::Var(aggregate), Expr::Var(aggregate)])).collect());
    SignedProductSquareFixture {
        problem: Problem { solver, search, objective: Some(Objective::Expr(true, objective)) },
        spec,
        signs,
        aggregates,
    }
}

fn signed_product_square_energy(signs: &[i32], groups: &[Vec<(usize, usize)>]) -> i64 {
    groups
        .iter()
        .map(|group| {
            let sum = group.iter().map(|&(left, right)| signs[left] * signs[right]).sum::<i32>();
            i64::from(sum) * i64::from(sum)
        })
        .sum()
}

#[test]
fn arbitrary_signed_product_groups_use_incremental_search() {
    let groups = vec![vec![(0, 2), (1, 5), (3, 4)], vec![(0, 1), (0, 4)], vec![(1, 5), (2, 3), (2, 4), (4, 5)]];
    let fixture = signed_product_square_problem(6, &groups, None, None);
    assert!(fixture.spec.has_signed_product_square_structure(&fixture.problem));

    let stop = AtomicBool::new(false);
    let mut sources = Vec::new();
    let outcome = solve_ls_capped(fixture.problem, fixture.spec, &stop, 0, LsConfig::default(), 160, |objective, solution, source| {
        assert_eq!(objective, signed_product_square_energy(&solution[..6], &groups));
        sources.push(source);
    });

    let (solution, objective) = outcome.best.expect("signed-product-square search produced no incumbent");
    assert_eq!(objective, signed_product_square_energy(&solution[..6], &groups));
    assert_eq!(sources.last(), Some(&"signed-product-squares"));
    assert_eq!(outcome.iterations, 160);
}

#[test]
fn the_signed_product_square_recognizer_has_no_size_floor() {
    let fixture = signed_product_square_problem(2, &[vec![(0, 1)]], None, None);
    assert!(fixture.spec.has_signed_product_square_structure(&fixture.problem));
}

#[test]
fn aggregate_domains_must_not_restrict_the_sign_space() {
    let restrictive = vec![vec![-2, 2]];
    let fixture = signed_product_square_problem(3, &[vec![(0, 1), (1, 2)]], Some(&restrictive), None);
    assert!(!fixture.spec.has_signed_product_square_structure(&fixture.problem));
}

#[test]
fn near_matches_do_not_activate_signed_product_square_search() {
    let groups = vec![vec![(0, 1), (1, 2)]];

    let weighted = signed_product_square_problem(3, &groups, None, Some((0, 0, 2)));
    assert!(!weighted.spec.has_signed_product_square_structure(&weighted.problem));

    let mut nonsquare = signed_product_square_problem(3, &groups, None, None);
    nonsquare.problem.objective = Some(Objective::Expr(true, Expr::Add(nonsquare.aggregates.iter().copied().map(Expr::Var).collect())));
    assert!(!nonsquare.spec.has_signed_product_square_structure(&nonsquare.problem));

    let self_product = signed_product_square_problem(2, &[vec![(0, 0)]], None, None);
    assert!(!self_product.spec.has_signed_product_square_structure(&self_product.problem));

    let mut constrained = signed_product_square_problem(3, &groups, None, None);
    constrained.spec.add_linear(vec![1], vec![constrained.signs[0]], Relation::Eq, 1);
    assert!(!constrained.spec.has_signed_product_square_structure(&constrained.problem));
}

#[test]
fn duplicated_square_term_is_not_a_signed_product_square_plan() {
    let mut fixture = signed_product_square_problem(2, &[vec![(0, 1)]], None, None);
    let aggregate = fixture.aggregates[0];
    let square = Expr::Mul(vec![Expr::Var(aggregate), Expr::Var(aggregate)]);
    fixture.problem.objective = Some(Objective::Expr(true, Expr::Add(vec![square.clone(), square])));

    assert!(!fixture.spec.has_signed_product_square_structure(&fixture.problem));
}

#[test]
fn duplicate_derived_target_is_not_a_signed_product_square_plan() {
    let mut fixture = signed_product_square_problem(2, &[vec![(0, 1)]], None, None);
    fixture.spec.add_expr(Expr::Eq(
        Box::new(Expr::Var(fixture.aggregates[0])),
        Box::new(Expr::Mul(vec![Expr::Var(fixture.signs[0]), Expr::Var(fixture.signs[1])])),
    ));

    assert!(!fixture.spec.has_signed_product_square_structure(&fixture.problem));
}

#[test]
fn unused_product_is_not_a_signed_product_square_plan() {
    let mut fixture = signed_product_square_problem(3, &[vec![(0, 1)]], None, None);
    let unused = fixture.problem.solver.new_var_set(&[-1, 1]);
    fixture.problem.search.push(unused);
    fixture.spec.add_var(unused);
    fixture.spec.add_expr(Expr::Eq(
        Box::new(Expr::Var(unused)),
        Box::new(Expr::Mul(vec![Expr::Var(fixture.signs[1]), Expr::Var(fixture.signs[2])])),
    ));

    assert!(!fixture.spec.has_signed_product_square_structure(&fixture.problem));
}

#[test]
fn irrelevant_mutable_variable_keeps_the_signed_product_plan_valid() {
    let mut fixture = signed_product_square_problem(2, &[vec![(0, 1)]], None, None);
    let unrelated = fixture.problem.solver.new_var_range(-7, 7);
    fixture.problem.search.push(unrelated);
    fixture.spec.add_var(unrelated);

    assert!(fixture.spec.has_signed_product_square_structure(&fixture.problem));
}

#[test]
fn a_product_can_feed_multiple_signed_product_groups() {
    let fixture = signed_product_square_problem(2, &[vec![(0, 1)], vec![(0, 1)]], None, None);

    assert!(fixture.spec.has_signed_product_square_structure(&fixture.problem));
}

#[test]
fn interrupted_signed_product_square_search_returns_without_work() {
    let fixture = signed_product_square_problem(4, &[vec![(0, 1), (1, 2), (2, 3)]], None, None);
    let stop = AtomicBool::new(true);
    let outcome = solve_ls_capped(fixture.problem, fixture.spec, &stop, 0, LsConfig::default(), 100, |_, _, _| {});
    assert!(outcome.best.is_none());
    assert_eq!(outcome.iterations, 0);
}

#[test]
fn borrowed_local_search_keeps_the_physical_problem_with_its_caller() {
    let fixture = signed_product_square_problem(3, &[vec![(0, 1), (1, 2)]], None, None);
    let SignedProductSquareFixture { problem, spec, .. } = fixture;
    let stop = AtomicBool::new(false);

    let outcome = solve_ls_capped_borrowed(&problem, spec, &stop, 0, LsConfig::default(), 4, |_, _, _| {});

    assert!(outcome.best.is_some());
    assert!(!problem.search.is_empty(), "borrowed LS must not consume or clone the physical root at its call boundary");
}

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
    let symbol = solver.new_var_range(0, 1);
    let search = vec![cells[0], cells[1], guard, index, symbol];
    let objective = Expr::Add(vec![Expr::Const(3), Expr::Mul(vec![Expr::Const(7), Expr::Var(guard)])]);
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Expr(false, objective)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    spec.add_element(cells.to_vec(), index, symbol, 0);
    spec.add_expr(Expr::Or(vec![
        Expr::Eq(Box::new(Expr::Var(guard)), Box::new(Expr::Const(0))),
        Expr::Eq(Box::new(Expr::Var(symbol)), Box::new(Expr::Const(1))),
    ]));
    // The real AlteredStates models contain extension constraints between sequence
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
fn guarded_mismatch_counter_allows_one_inexact_symbol() {
    let mut solver = Solver::new();
    let cell = solver.new_var_range(0, 0);
    let guard = solver.new_var_range(0, 1);
    let index = solver.new_var_range(0, 0);
    let symbol = solver.new_var_range(0, 1);
    let mismatch = solver.new_var_range(0, 1);
    let mismatch_count = solver.new_var_range(0, 1);
    let search = vec![cell, guard, index, symbol, mismatch, mismatch_count];
    let objective = Expr::Mul(vec![Expr::Const(7), Expr::Var(guard)]);
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Expr(false, objective)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    spec.add_element(vec![cell], index, symbol, 0);
    spec.add_expr(Expr::Eq(Box::new(Expr::Var(mismatch)), Box::new(Expr::Ne(Box::new(Expr::Var(symbol)), Box::new(Expr::Const(1))))));
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
fn guarded_array_optimization_observes_external_stop() {
    let mut solver = Solver::new();
    let cell = solver.new_var_range(0, 1);
    let guard = solver.new_var_range(0, 1);
    let index = solver.new_var_range(0, 0);
    let symbol = solver.new_var_range(0, 1);
    let mismatch = solver.new_var_range(0, 1);
    let mismatch_count = solver.new_var_range(0, 1);
    let activation = solver.new_var_range(0, 0);
    let search = vec![cell, guard, index, symbol, mismatch, mismatch_count, activation];
    let objective = Expr::Mul(vec![Expr::Const(7), Expr::Var(guard)]);
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Expr(false, objective)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    spec.add_element(vec![cell], index, symbol, 0);
    spec.add_expr(Expr::Eq(Box::new(Expr::Var(mismatch)), Box::new(Expr::Ne(Box::new(Expr::Var(symbol)), Box::new(Expr::Const(1))))));
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
    assert!(started.elapsed() < Duration::from_secs(1), "guarded-shared_array optimization ignored its stop signal");
}

#[test]
fn guarded_sequence_branch_and_bound_reuses_cells_for_later_sequences() {
    let mut solver = Solver::new();
    let cells = [solver.new_var_range(0, 2), solver.new_var_range(0, 2), solver.new_var_range(0, 2)];
    let guards = [solver.new_var_range(0, 1), solver.new_var_range(0, 1), solver.new_var_range(0, 1)];
    let indices = [solver.new_var_range(2, 2), solver.new_var_range(0, 2), solver.new_var_range(0, 0)];
    let symbols = [solver.new_var_range(0, 2), solver.new_var_range(0, 2), solver.new_var_range(0, 2)];
    let activation = solver.new_var_range(0, 0);
    let mut search = cells.into_iter().chain(guards).chain(indices).chain(symbols).collect::<Vec<_>>();
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
    for ((guard, index), (symbol, value)) in guards.into_iter().zip(indices).zip(symbols.into_iter().zip([1, 1, 2])) {
        spec.add_element(cells.to_vec(), index, symbol, 0);
        spec.add_expr(Expr::Or(vec![
            Expr::Eq(Box::new(Expr::Var(guard)), Box::new(Expr::Const(0))),
            Expr::Eq(Box::new(Expr::Var(symbol)), Box::new(Expr::Const(value))),
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
    assert_eq!(&solution[6..9], &[2, 2, 0], "the flexible sequence should reuse the occupied matching cell");
    assert_eq!(&solution[..3], &[2, 0, 1]);
    assert_eq!(source, "constructive");
}

#[test]
fn guarded_array_lex_canonicalization_remaps_element_indices() {
    let mut solver = Solver::new();
    let cells = [solver.new_var_range(0, 1), solver.new_var_range(0, 1)];
    let guard = solver.new_var_range(0, 1);
    let index = solver.new_var_range(0, 1);
    let symbol = solver.new_var_range(0, 1);
    let search = vec![cells[0], cells[1], guard, index, symbol];
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Var(false, guard)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    spec.add_element(cells.to_vec(), index, symbol, 0);
    spec.add_expr(Expr::Or(vec![
        Expr::Eq(Box::new(Expr::Var(guard)), Box::new(Expr::Const(0))),
        Expr::Eq(Box::new(Expr::Var(symbol)), Box::new(Expr::Const(1))),
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
    assert_eq!(&solution[..2], &[0, 1], "the shared_array was not moved to its lexicographically least image");
    assert_eq!(solution[3], 1, "the Element index was not remapped by the inverse shared_array permutation");
    assert_eq!(solution[4], 1, "the remapped Element no longer reads the requested symbol");
    assert_eq!(source, "constructive");
}

#[test]
fn guarded_array_lex_keeps_inactive_element_indices_at_their_sentinel() {
    let mut solver = Solver::new();
    let cells = [solver.new_var_range(0, 1), solver.new_var_range(0, 1)];
    let guards = [solver.new_var_range(0, 1), solver.new_var_range(0, 1)];
    let indices = [solver.new_var_range(0, 1), solver.new_var_range(0, 0)];
    let symbols = [solver.new_var_range(0, 1), solver.new_var_range(0, 1)];
    let search = cells.into_iter().chain(guards).chain(indices).chain(symbols).collect::<Vec<_>>();
    let objective = Expr::Add(vec![Expr::Mul(vec![Expr::Const(2), Expr::Var(guards[0])]), Expr::Var(guards[1])]);
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Expr(false, objective)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    for ((guard, index), (symbol, value)) in guards.into_iter().zip(indices).zip(symbols.into_iter().zip([1, 2])) {
        spec.add_element(cells.to_vec(), index, symbol, 0);
        spec.add_expr(Expr::Or(vec![
            Expr::Eq(Box::new(Expr::Var(guard)), Box::new(Expr::Const(0))),
            Expr::Eq(Box::new(Expr::Var(symbol)), Box::new(Expr::Const(value))),
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
    assert_eq!(solution[4], 1, "the selected sequence index was not remapped");
    assert_eq!(solution[5], 0, "the inactive sequence index left its sentinel");
    assert_eq!(source, "constructive");
}

#[test]
fn malformed_shared_array_lex_is_not_deferred() {
    let mut solver = Solver::new();
    let cells = [solver.new_var_range(0, 1), solver.new_var_range(0, 1)];
    let guard = solver.new_var_range(0, 1);
    let index = solver.new_var_range(0, 0);
    let symbol = solver.new_var_range(0, 1);
    let search = vec![cells[0], cells[1], guard, index, symbol];
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Var(false, guard)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    spec.add_element(cells.to_vec(), index, symbol, 0);
    spec.add_expr(Expr::Or(vec![
        Expr::Eq(Box::new(Expr::Var(guard)), Box::new(Expr::Const(0))),
        Expr::Eq(Box::new(Expr::Var(symbol)), Box::new(Expr::Const(1))),
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
fn guarded_array_materialization_preserves_nonzero_index_domains() {
    let mut solver = Solver::new();
    let cells = [solver.new_var_range(0, 0), solver.new_var_range(0, 0)];
    let guard = solver.new_var_range(0, 1);
    let index = solver.new_var_range(1, 1);
    let symbol = solver.new_var_range(0, 1);
    let activation = solver.new_var_range(0, 0);
    let search = vec![cells[0], cells[1], guard, index, symbol, activation];
    let problem = Problem { solver, search: search.clone(), objective: Some(Objective::Var(false, guard)) };

    let mut spec = LocalSearchSpec::default();
    for variable in search {
        spec.add_var(variable);
    }
    spec.add_element(cells.to_vec(), index, symbol, 0);
    spec.add_expr(Expr::Or(vec![
        Expr::Eq(Box::new(Expr::Var(guard)), Box::new(Expr::Const(0))),
        Expr::Eq(Box::new(Expr::Var(symbol)), Box::new(Expr::Const(1))),
    ]));
    spec.add_extension(vec![activation], vec![vec![0]], true);

    let stop = AtomicBool::new(false);
    let mut incumbent = None;
    let _ = solve_ls(problem, spec, &stop, 0, LsConfig::default(), |value, solution, _| {
        incumbent = Some((value, solution.to_vec()));
        stop.store(true, Ordering::Relaxed);
    });

    let (value, solution) = incumbent.expect("guarded-shared_array fallback did not publish an incumbent");
    assert_eq!(value, 0);
    assert_eq!(solution[3], 1, "materialization wrote an out-of-domain zero to the Element index");
}
