//! Expr unit tests (moved out of `src/expr.rs`). Boolean-interval sentinels are
//! spelled as literals here since they are private to the module.

use qayd::expr::{abs, add, and, div, int, lt, mul, ne, var};
use qayd::VarId;

fn fixed(vals: &[(VarId, i64)]) -> impl Fn(VarId) -> i64 + '_ {
    move |v| vals.iter().find(|(u, _)| *u == v).unwrap().1
}

#[test]
fn eval_arithmetic() {
    let x = VarId(0);
    let e = add(vec![mul(vec![int(2), var(x)]), int(1)]); // 2x + 1
    assert_eq!(e.eval(&fixed(&[(x, 3)])), Some(7));
}

#[test]
fn eval_relational_and_logical() {
    let x = VarId(0);
    let e = and(vec![lt(var(x), int(5)), ne(var(x), int(2))]);
    assert_eq!(e.eval(&fixed(&[(x, 3)])), Some(1));
    assert_eq!(e.eval(&fixed(&[(x, 2)])), Some(0));
    assert_eq!(e.eval(&fixed(&[(x, 7)])), Some(0));
}

#[test]
fn eval_div_by_zero_is_none() {
    let x = VarId(0);
    let e = div(int(10), var(x));
    assert_eq!(e.eval(&fixed(&[(x, 0)])), None);
    assert_eq!(e.eval(&fixed(&[(x, 2)])), Some(5));
}

#[test]
fn bounds_are_sound_supersets() {
    let x = VarId(0);
    let dom = |v: VarId| if v == x { (-2, 3) } else { (0, 0) };
    // 2x in [-4, 6]
    assert_eq!(mul(vec![int(2), var(x)]).bounds(&dom), (-4, 6));
    // |x| in [0, 3]
    assert_eq!(abs(var(x)).bounds(&dom), (0, 3));
    // x < 10 is certainly true => (1, 1)
    assert_eq!(lt(var(x), int(10)).bounds(&dom), (1, 1));
    // x < -5 is certainly false => (0, 0)
    assert_eq!(lt(var(x), int(-5)).bounds(&dom), (0, 0));
    // x < 1 is undetermined => (0, 1)
    assert_eq!(lt(var(x), int(1)).bounds(&dom), (0, 1));
}
