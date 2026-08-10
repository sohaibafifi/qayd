use crate::expr::Expr;
use crate::ids::VarId;
use crate::lcg::guarded_sum::GuardedSum;
use crate::Solver;
use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

fn var(variable: VarId) -> Expr {
    Expr::Var(variable)
}

fn constant(value: i64) -> Expr {
    Expr::Const(value)
}

fn eq(left: Expr, right: Expr) -> Expr {
    Expr::Eq(Box::new(left), Box::new(right))
}

fn ne(left: Expr, right: Expr) -> Expr {
    Expr::Ne(Box::new(left), Box::new(right))
}

fn and(atoms: Vec<Expr>) -> Expr {
    Expr::And(atoms)
}

#[test]
fn compact_bounds_follow_holey_domains_and_forced_values() {
    let mut solver = Solver::new();
    let x = solver.new_var_set(&[0, 2]);
    let y = solver.new_var_range(0, 1);
    let expression =
        Expr::Add(vec![and(vec![eq(var(x), constant(2)), var(y)]), and(vec![ne(constant(2), var(x)), Expr::Not(Box::new(var(y)))])]);
    let guarded = GuardedSum::compile(&expression).expect("supported guarded sum");

    assert_eq!(guarded.vars(), &[x, y]);
    assert_eq!(guarded.bounds(&solver.store), (0, 2));
    assert_eq!(guarded.bounds_with_value(&solver.store, x, 2), (0, 1));
    assert_eq!(guarded.bounds_with_value(&solver.store, y, 1), (0, 1));

    solver.store.fix(x, 2).unwrap();
    solver.store.fix(y, 1).unwrap();
    assert_eq!(guarded.bounds(&solver.store), (1, 1));
}

#[test]
fn compilation_folds_constants_duplicates_and_contradictions_exactly() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let expression = Expr::Add(vec![
        and(vec![eq(var(x), constant(i64::MAX))]),
        and(vec![ne(var(x), constant(i64::MAX))]),
        and(vec![eq(var(x), constant(1)), ne(var(x), constant(1))]),
        and(vec![eq(var(x), constant(1)), ne(var(x), constant(2)), var(x), var(x)]),
        and(vec![constant(-3), Expr::Not(Box::new(constant(0)))]),
    ]);
    let guarded = GuardedSum::compile(&expression).expect("supported guarded sum");

    // Two terms are constant true and the normalized x == 1 term is open.
    assert_eq!(guarded.vars(), &[x]);
    assert_eq!(guarded.bounds(&solver.store), (2, 3));
    assert_eq!(guarded.bounds_with_value(&solver.store, x, 0), (2, 2));
    assert_eq!(guarded.bounds_with_value(&solver.store, x, 1), (3, 3));

    solver.store.fix(x, 0).unwrap();
    assert_eq!(guarded.bounds(&solver.store), (2, 2));
}

#[test]
fn unsupported_shapes_are_rejected_as_a_whole() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);

    assert!(GuardedSum::compile(&and(vec![var(x)])).is_none());
    assert!(GuardedSum::compile(&Expr::Add(vec![var(x)])).is_none());
    assert!(GuardedSum::compile(&Expr::Add(vec![and(vec![Expr::Or(vec![var(x)])])])).is_none());
    assert!(GuardedSum::compile(&Expr::Add(vec![and(vec![eq(Expr::Add(vec![var(x), constant(1)]), constant(1),)])])).is_none());
}

#[test]
fn minimization_hint_returns_an_exact_supported_assignment() {
    let mut solver = Solver::new();
    let x = solver.new_var_set(&[-2, 0, 3]);
    let y = solver.new_var_range(0, 2);
    let expression = Expr::Add(vec![
        and(vec![eq(var(x), constant(-2)), ne(var(y), constant(1))]),
        and(vec![ne(var(x), constant(3)), eq(constant(2), var(y))]),
        and(vec![var(x), Expr::Not(Box::new(var(y)))]),
    ]);
    let guarded = GuardedSum::compile(&expression).unwrap();
    let (objective, assignment) = guarded
        .minimize_hint(&solver.store, 17, &AtomicBool::new(false), 2_000)
        .expect("the work budget covers at least one complete assignment");
    let assigned = assignment.iter().copied().collect::<BTreeMap<_, _>>();

    assert_eq!(assignment.len(), 2);
    assert!(assignment.iter().all(|&(variable, value)| solver.store.contains(variable, value)));
    assert_eq!(expression.eval(&|variable| i64::from(assigned[&variable])), Some(objective));
}

#[test]
fn coordinate_descent_improves_the_domain_feasible_start() {
    let mut solver = Solver::new();
    let variables = (0..8).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
    let expression = Expr::Add(variables.iter().map(|&variable| and(vec![Expr::Not(Box::new(var(variable)))])).collect());
    let guarded = GuardedSum::compile(&expression).unwrap();
    let (objective, assignment) = guarded.minimize_hint(&solver.store, 0, &AtomicBool::new(false), 1_000).unwrap();

    assert_eq!(objective, 0);
    assert!(assignment.iter().all(|&(_, value)| value == 1));
}

#[test]
fn two_variable_perturbation_escapes_a_coordinate_local_minimum() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    let at = |x_value, y_value| and(vec![eq(var(x), constant(x_value)), eq(var(y), constant(y_value))]);
    let expression = Expr::Add(vec![at(0, 0), at(1, 0), at(1, 0), at(0, 1), at(0, 1)]);
    let guarded = GuardedSum::compile(&expression).unwrap();
    let (objective, assignment) = guarded.minimize_hint(&solver.store, 0, &AtomicBool::new(false), 100).unwrap();

    assert_eq!(objective, 0);
    assert_eq!(assignment, vec![(x, 1), (y, 1)]);
}

#[test]
fn minimization_hint_is_seed_deterministic() {
    let mut solver = Solver::new();
    let variables = (0..6).map(|_| solver.new_var_range(0, 2)).collect::<Vec<_>>();
    let expression = Expr::Add(
        (0..variables.len())
            .flat_map(|left| {
                let variables = &variables;
                (left + 1..variables.len())
                    .map(move |right| and(vec![eq(var(variables[left]), constant(1)), ne(var(variables[right]), constant(2))]))
            })
            .collect(),
    );
    let guarded = GuardedSum::compile(&expression).unwrap();
    let stop = AtomicBool::new(false);

    let first = guarded.minimize_hint(&solver.store, 91, &stop, 4_000);
    let second = guarded.minimize_hint(&solver.store, 91, &stop, 4_000);
    assert_eq!(first, second);
}

#[test]
fn minimization_hint_honors_interruption_and_literal_work_budget() {
    let mut solver = Solver::new();
    let variables = (0..4).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
    let expression = Expr::Add(variables.iter().map(|&variable| and(vec![var(variable)])).collect());
    let guarded = GuardedSum::compile(&expression).unwrap();

    assert_eq!(guarded.minimize_hint(&solver.store, 0, &AtomicBool::new(true), 100), None);
    // Constructing the first exact state needs one literal evaluation per term.
    assert_eq!(guarded.minimize_hint(&solver.store, 0, &AtomicBool::new(false), 3), None);
    assert!(guarded.minimize_hint(&solver.store, 0, &AtomicBool::new(false), 4).is_some());
}
