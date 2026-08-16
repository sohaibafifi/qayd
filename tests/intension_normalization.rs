use qayd::constraints::intension::intension;
use qayd::{count_solutions, Expr, Solver, VarId};

fn fixed(solver: &mut Solver, value: i32) -> VarId {
    solver.new_var_range(value, value)
}

fn accepts(expression: impl FnOnce(&mut Solver) -> Expr) -> bool {
    let mut solver = Solver::new();
    let expression = expression(&mut solver);
    intension(&mut solver, expression);
    count_solutions(&mut solver, &[]) == 1
}

#[test]
fn affine_relations_match_their_algebraic_definition() {
    for x in -2..=2 {
        for y in -2..=2 {
            let accepted = accepts(|solver| {
                let x_var = fixed(solver, x);
                let y_var = fixed(solver, y);
                Expr::Le(
                    Box::new(Expr::Sub(Box::new(Expr::Mul(vec![Expr::Const(2), Expr::Var(x_var)])), Box::new(Expr::Var(y_var)))),
                    Box::new(Expr::Const(1)),
                )
            });
            assert_eq!(accepted, 2 * x - y <= 1, "x={x}, y={y}");
        }
    }
}

#[test]
fn positive_and_negative_reified_affine_forms_are_exact() {
    for selector in 0..=1 {
        for x in -1..=1 {
            for y in -1..=1 {
                let equivalence = accepts(|solver| {
                    let selector_var = fixed(solver, selector);
                    let x_var = fixed(solver, x);
                    let y_var = fixed(solver, y);
                    Expr::Iff(
                        Box::new(Expr::Not(Box::new(Expr::Var(selector_var)))),
                        Box::new(Expr::Ge(Box::new(Expr::Add(vec![Expr::Var(x_var), Expr::Var(y_var)])), Box::new(Expr::Const(1)))),
                    )
                });
                assert_eq!(equivalence, (selector == 0) == (x + y >= 1), "iff selector={selector}, x={x}, y={y}");

                let implication = accepts(|solver| {
                    let selector_var = fixed(solver, selector);
                    let x_var = fixed(solver, x);
                    let y_var = fixed(solver, y);
                    Expr::Imp(
                        Box::new(Expr::Not(Box::new(Expr::Var(selector_var)))),
                        Box::new(Expr::Eq(
                            Box::new(Expr::Sub(Box::new(Expr::Var(x_var)), Box::new(Expr::Var(y_var)))),
                            Box::new(Expr::Const(0)),
                        )),
                    )
                });
                assert_eq!(implication, selector == 1 || x == y, "imp selector={selector}, x={x}, y={y}");
            }
        }
    }
}

#[test]
fn reified_boolean_formulas_match_truth_tables() {
    for selector in 0..=1 {
        for left in 0..=1 {
            for right in 0..=1 {
                let or_equivalence = accepts(|solver| {
                    let selector_var = fixed(solver, selector);
                    let left_var = fixed(solver, left);
                    let right_var = fixed(solver, right);
                    Expr::Iff(
                        Box::new(Expr::Var(selector_var)),
                        Box::new(Expr::Or(vec![Expr::Not(Box::new(Expr::Var(left_var))), Expr::Var(right_var)])),
                    )
                });
                assert_eq!(or_equivalence, (selector == 1) == (left == 0 || right == 1));

                let and_equivalence = accepts(|solver| {
                    let selector_var = fixed(solver, selector);
                    let left_var = fixed(solver, left);
                    let right_var = fixed(solver, right);
                    Expr::Iff(
                        Box::new(Expr::Not(Box::new(Expr::Var(selector_var)))),
                        Box::new(Expr::And(vec![Expr::Var(left_var), Expr::Not(Box::new(Expr::Var(right_var)))])),
                    )
                });
                assert_eq!(and_equivalence, (selector == 0) == (left == 1 && right == 0));
            }
        }
    }
}

#[test]
fn nonlinear_near_match_remains_semantically_exact() {
    let mut accepted = 0;
    for x in -1..=1 {
        for y in -1..=1 {
            accepted += usize::from(accepts(|solver| {
                let x_var = fixed(solver, x);
                let y_var = fixed(solver, y);
                Expr::Le(Box::new(Expr::Mul(vec![Expr::Var(x_var), Expr::Var(y_var)])), Box::new(Expr::Const(0)))
            }));
        }
    }
    assert_eq!(accepted, 7);
}
