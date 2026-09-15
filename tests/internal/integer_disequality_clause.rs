use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;

use crate::constraints::intension::{intension, intension_interruptible};
use crate::lcg::lit::LitOrConst;
use crate::lcg::trail::{Cdcl, Reason};
use crate::lcg::view::Tri;
use crate::model::{CompiledCp, Constraint, IntExpr, Model};
use crate::{optimize_with, solve_search, Expr, SearchControl, Solver, VarId};

fn ne(left: Expr, right: Expr) -> Expr {
    Expr::Ne(Box::new(left), Box::new(right))
}

fn clause(variables: &[VarId], endpoints: [usize; 4]) -> Expr {
    let operand = |index| match index {
        0..=3 => Expr::Var(variables[index]),
        4 => Expr::Const(-1),
        _ => Expr::Const(2),
    };
    Expr::Or(vec![ne(operand(endpoints[0]), operand(endpoints[1])), ne(operand(endpoints[2]), operand(endpoints[3]))])
}

fn assignments(domain: &[i32], count: usize) -> Vec<Vec<i32>> {
    let mut result = vec![Vec::new()];
    for _ in 0..count {
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

fn satisfies(assignment: &[i32], endpoints: [usize; 4]) -> bool {
    let value = |index| match index {
        0..=3 => assignment[index],
        4 => -1,
        _ => 2,
    };
    value(endpoints[0]) != value(endpoints[1]) || value(endpoints[2]) != value(endpoints[3])
}

#[test]
fn all_endpoint_aliases_and_constants_match_exhaustive_sparse_solutions() {
    // Every arrangement of four variables and two constants includes repeated
    // variables within/across alternatives, reversed duplicates and tautologies.
    for domain in [vec![-1, 0, 2], vec![-3, 2]] {
        let all = assignments(&domain, 4);
        for encoded in 0usize..6usize.pow(4) {
            let endpoints = [encoded % 6, (encoded / 6) % 6, (encoded / 36) % 6, (encoded / 216) % 6];
            let expected = all.iter().filter(|values| satisfies(values, endpoints)).cloned().collect::<BTreeSet<_>>();
            let mut solver = Solver::new();
            let variables = (0..4).map(|_| solver.new_var_set(&domain)).collect::<Vec<_>>();
            intension(&mut solver, clause(&variables, endpoints));
            let mut actual = BTreeSet::new();
            solve_search(&mut solver, &variables, |solver| {
                actual.insert(variables.iter().map(|&variable| solver.store.value(variable)).collect::<Vec<_>>());
                SearchControl::Continue
            });
            assert_eq!(actual, expected, "domain={domain:?}, endpoints={endpoints:?}");
        }
    }
}

#[test]
fn unit_filtering_removes_interior_values_and_restores_after_backtracking() {
    let mut solver = Solver::new();
    let variables = (0..4).map(|_| solver.new_var_set(&[-3, 0, 4])).collect::<Vec<_>>();
    intension(&mut solver, clause(&variables, [0, 1, 2, 3]));
    solver.propagate().unwrap();
    solver.store.push_level();
    for &variable in &variables[..3] {
        solver.store.fix(variable, 0).unwrap();
    }
    solver.propagate().unwrap();
    assert_eq!(solver.store.values(variables[3]).collect::<Vec<_>>(), [-3, 4]);
    solver.store.pop_level();
    assert!(solver.store.contains(variables[3], 0));
    solver.store.push_level();
    solver.store.fix(variables[0], -3).unwrap();
    solver.store.fix(variables[1], 4).unwrap();
    solver.store.fix(variables[2], 0).unwrap();
    solver.store.fix(variables[3], 0).unwrap();
    solver.propagate().unwrap();
    solver.store.pop_level();
    for &variable in &variables {
        solver.store.fix(variable, 0).unwrap();
    }
    assert!(solver.propagate().is_err());
}

#[test]
fn wide_domains_and_i64_constants_never_require_value_enumeration() {
    let mut solver = Solver::new();
    let variables = (0..4).map(|_| solver.new_var_range(i32::MIN, i32::MAX)).collect::<Vec<_>>();
    intension(&mut solver, clause(&variables, [0, 1, 2, 3]));
    solver.propagate().unwrap();
    solver.store.fix(variables[0], i32::MIN).unwrap();
    solver.store.fix(variables[1], i32::MIN).unwrap();
    solver.store.fix(variables[2], i32::MAX).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.max(variables[3]), i32::MAX - 1);
    assert_eq!(solver.store.min(variables[3]), i32::MIN);

    for constant in [i64::MIN, i64::MAX, i64::from(i32::MIN) - 1, i64::from(i32::MAX) + 1] {
        let mut solver = Solver::new();
        let x = solver.new_var_range(i32::MIN, i32::MAX);
        intension(&mut solver, Expr::Or(vec![ne(Expr::Var(x), Expr::Const(constant)), ne(Expr::Var(x), Expr::Var(x))]));
        solver.propagate().unwrap();
        assert_eq!((solver.store.min(x), solver.store.max(x)), (i32::MIN, i32::MAX));
    }
    for constant in [i32::MIN, i32::MAX] {
        let mut solver = Solver::new();
        let x = solver.new_var_range(i32::MIN, i32::MAX);
        intension(&mut solver, Expr::Or(vec![ne(Expr::Const(i64::from(constant)), Expr::Var(x))]));
        solver.propagate().unwrap();
        assert!(!solver.store.contains(x, constant));
    }
}

#[test]
fn arbitrary_clause_lengths_and_reflexive_terms_keep_their_semantics() {
    for count in [0, 1, 2, 7, 70] {
        let mut solver = Solver::new();
        let x = solver.new_var_set(&[-1, 0, 2]);
        let mut alternatives = vec![ne(Expr::Var(x), Expr::Var(x)); count];
        alternatives.extend([ne(Expr::Var(x), Expr::Const(0)), ne(Expr::Const(0), Expr::Var(x))]);
        intension(&mut solver, Expr::Or(alternatives));
        solver.propagate().unwrap();
        assert_eq!(solver.store.values(x).collect::<Vec<_>>(), [-1, 2]);
    }
    for expression in [Expr::Or(vec![]), Expr::Or(vec![ne(Expr::Const(i64::MAX), Expr::Const(i64::MAX))])] {
        let mut solver = Solver::new();
        intension(&mut solver, expression);
        assert!(solver.propagate().is_err());
    }
    let mut solver = Solver::new();
    let variables = (0..12).map(|_| solver.new_var_range(0, 3)).collect::<Vec<_>>();
    let alternatives = variables.chunks_exact(2).map(|pair| ne(Expr::Var(pair[0]), Expr::Var(pair[1]))).collect();
    intension(&mut solver, Expr::Or(alternatives));
    for &variable in &variables[..11] {
        solver.store.fix(variable, 2).unwrap();
    }
    solver.propagate().unwrap();
    assert!(!solver.store.contains(variables[11], 2));
}

#[test]
fn integer_clause_lowering_keeps_variables_and_rejects_arithmetic_or_equality_alternatives() {
    for form in 0..3 {
        let mut model = Model::new();
        let variables = (0..4).map(|_| model.int_range(-2, 3)).collect::<Vec<_>>();
        let first = IntExpr::Ne(Box::new(IntExpr::Variable(variables[0])), Box::new(IntExpr::Variable(variables[1])));
        let second = match form {
            0 => IntExpr::Ne(Box::new(IntExpr::Variable(variables[2])), Box::new(IntExpr::Variable(variables[3]))),
            1 => IntExpr::Eq(Box::new(IntExpr::Variable(variables[2])), Box::new(IntExpr::Variable(variables[3]))),
            _ => IntExpr::Ne(
                Box::new(IntExpr::Add(vec![IntExpr::Variable(variables[2]), IntExpr::Constant(1)])),
                Box::new(IntExpr::Variable(variables[3])),
            ),
        };
        model.add_constraint(Constraint::Intension(IntExpr::Or(vec![first, second])));
        let compiled = CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).unwrap().unwrap();
        assert_eq!(compiled.compilation_stats().native_intensions, u64::from(form == 0));
        assert_eq!(compiled.compilation_stats().fallback_intensions, u64::from(form != 0));
        assert_eq!(compiled.problem().solver.store.num_vars(), 4);
        assert_eq!(compiled.problem().solver.num_propagators(), 1);
    }
}

#[test]
fn selector_guards_disable_conflicts_and_survive_lcg_backjumps() {
    for enabled in [0, 1] {
        let mut solver = Solver::new();
        let selector = solver.new_var_range(enabled, enabled);
        let variables = (0..4).map(|_| solver.new_var_range(2, 2)).collect::<Vec<_>>();
        solver.set_selector(Some(selector));
        intension(&mut solver, clause(&variables, [0, 1, 2, 3]));
        solver.set_selector(None);
        assert_eq!(solver.propagate().is_err(), enabled == 1);
    }
    for force_scope in [false, true] {
        let mut solver = Solver::new();
        solver.set_force_scope_reasons(force_scope);
        let selector = solver.new_var_range(0, 1);
        let variables = (0..4).map(|_| solver.new_var_set(&[-1, 0, 2])).collect::<Vec<_>>();
        solver.set_selector(Some(selector));
        intension(&mut solver, clause(&variables, [0, 1, 2, 3]));
        solver.set_selector(None);
        let mut scope = variables.clone();
        scope.push(selector);
        let mut cdcl = Cdcl::new(&mut solver, &scope);
        assert!(cdcl.init());
        for variable in [variables[0], variables[1], variables[2]] {
            let LitOrConst::Lit(fixed) = cdcl.atoms.eq(variable, 0) else { panic!("expected nonconstant equality") };
            cdcl.decide(fixed).unwrap();
            cdcl.propagate().unwrap();
        }
        assert!(cdcl.solver.store.contains(variables[3], 0), "an undecided selector does not enforce the clause");
        let LitOrConst::Lit(enabled) = cdcl.atoms.eq(selector, 1) else { panic!("expected selector equality") };
        cdcl.decide(enabled).unwrap();
        cdcl.propagate().unwrap();
        let LitOrConst::Lit(removed) = cdcl.atoms.eq(variables[3], 0) else { panic!("expected removed-value equality") };
        assert_eq!(cdcl.tvalue(removed), Tri::False);
        let Reason::Generic(reason) = cdcl.reason_of(removed.atom()) else { panic!("expected propagator reason") };
        assert!(reason.iter().all(|&literal| cdcl.tvalue(literal) == Tri::True));
        for variable in [selector, variables[0], variables[1], variables[2]] {
            assert!(reason.iter().any(|literal| cdcl.atoms.var_of(literal.atom()) == variable));
        }
        if !force_scope {
            assert!(reason.iter().all(|literal| cdcl.atoms.var_of(literal.atom()) != variables[3]));
        }
        cdcl.backjump_to(0);
        assert!(cdcl.solver.store.contains(variables[3], 0));
        let LitOrConst::Lit(disabled) = cdcl.atoms.eq(selector, 0) else { panic!("expected selector equality") };
        cdcl.decide(disabled).unwrap();
        cdcl.propagate().unwrap();
        for &variable in &variables {
            let LitOrConst::Lit(fixed) = cdcl.atoms.eq(variable, 0) else { panic!("expected nonconstant equality") };
            cdcl.decide(fixed).unwrap();
            cdcl.propagate().unwrap();
        }
    }
}

#[test]
fn lcg_optimization_matches_brute_force_with_aliases_and_both_reason_modes() {
    let domains = [-1, 0, 2];
    for endpoints in [[0, 1, 2, 3], [0, 1, 1, 2], [0, 1, 1, 0], [0, 0, 2, 3], [0, 4, 0, 5], [0, 0, 1, 1]] {
        let all = assignments(&domains, 4);
        let feasible = all.iter().filter(|assignment| satisfies(assignment, endpoints) && assignment[0] == assignment[1]);
        let expected = feasible.map(|assignment| assignment[3]).collect::<Vec<_>>();
        for force_scope in [false, true] {
            for minimizing in [false, true] {
                let mut solver = Solver::new();
                solver.set_force_scope_reasons(force_scope);
                let variables = (0..4).map(|_| solver.new_var_set(&domains)).collect::<Vec<_>>();
                intension(&mut solver, clause(&variables, endpoints));
                intension(&mut solver, Expr::Eq(Box::new(Expr::Var(variables[0])), Box::new(Expr::Var(variables[1]))));
                let (best, _) = optimize_with(&mut solver, &variables, variables[3], minimizing, &AtomicBool::new(false), |_| {});
                let objective = if minimizing { expected.iter().min() } else { expected.iter().max() }.copied();
                assert_eq!(best.map(|(_, value)| value), objective, "endpoints={endpoints:?}, scope={force_scope}, min={minimizing}");
            }
        }
    }
}

#[test]
fn native_clause_posting_obeys_a_prearmed_stop() {
    let mut solver = Solver::new();
    let variables = (0..4).map(|_| solver.new_var_range(0, 4)).collect::<Vec<_>>();
    assert!(!intension_interruptible(&mut solver, clause(&variables, [0, 1, 2, 3]), &AtomicBool::new(true)));
    assert_eq!(solver.num_propagators(), 0);
}
