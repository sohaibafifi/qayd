//! Expression trees for `intension` constraints.
//!
//! An [`Expr`] is an arithmetic/relational/logical tree over variables. Two
//! interpretations drive the `intension` propagator:
//!
//! - [`Expr::eval`] — exact evaluation under a full assignment. Returns `None`
//!   for an undefined result (division/modulo by zero), so the solver never
//!   panics on the search path. Relational and logical nodes return `0`/`1`.
//! - [`Expr::bounds`] — a *sound* interval `[lo, hi]` of the possible values
//!   given current variable bounds. "Sound" means it never excludes a feasible
//!   value, so it is safe to use for disentailment: an arithmetic-tree bound may
//!   be wider than reality (division/modulo are widened conservatively), but it
//!   is never narrower. A boolean node's interval is a subset of `[0, 1]`.

use crate::ids::VarId;

/// A node in an `intension` expression tree.
#[derive(Clone)]
pub enum Expr {
    /// Integer constant.
    Const(i64),
    /// A variable's value.
    Var(VarId),

    // arithmetic
    /// Arithmetic negation.
    Neg(Box<Expr>),
    /// Absolute value.
    Abs(Box<Expr>),
    /// Sum of operands.
    Add(Vec<Expr>),
    /// `a - b`.
    Sub(Box<Expr>, Box<Expr>),
    /// Product of operands.
    Mul(Vec<Expr>),
    /// Integer division `a / b` (truncated toward zero).
    Div(Box<Expr>, Box<Expr>),
    /// Remainder `a % b` (truncated, sign of dividend).
    Mod(Box<Expr>, Box<Expr>),
    /// Minimum of operands.
    Min(Vec<Expr>),
    /// Maximum of operands.
    Max(Vec<Expr>),

    // relational (yield 0/1)
    /// `a == b`.
    Eq(Box<Expr>, Box<Expr>),
    /// `a != b`.
    Ne(Box<Expr>, Box<Expr>),
    /// `a < b`.
    Lt(Box<Expr>, Box<Expr>),
    /// `a <= b`.
    Le(Box<Expr>, Box<Expr>),
    /// `a > b`.
    Gt(Box<Expr>, Box<Expr>),
    /// `a >= b`.
    Ge(Box<Expr>, Box<Expr>),

    // logical (operands and result are 0/1; non-zero counts as true)
    /// Logical negation.
    Not(Box<Expr>),
    /// Conjunction.
    And(Vec<Expr>),
    /// Disjunction.
    Or(Vec<Expr>),
    /// Implication `a -> b`.
    Imp(Box<Expr>, Box<Expr>),
    /// Bi-implication `a <-> b`.
    Iff(Box<Expr>, Box<Expr>),

    /// `if cond != 0 then then_ else else_`.
    IfThenElse(Box<Expr>, Box<Expr>, Box<Expr>),
}

impl Expr {
    /// Collect every variable referenced in the tree (with duplicates).
    pub fn collect_vars(&self, out: &mut Vec<VarId>) {
        match self {
            Expr::Const(_) => {}
            Expr::Var(v) => out.push(*v),
            Expr::Neg(a) | Expr::Abs(a) | Expr::Not(a) => a.collect_vars(out),
            Expr::Sub(a, b)
            | Expr::Div(a, b)
            | Expr::Mod(a, b)
            | Expr::Eq(a, b)
            | Expr::Ne(a, b)
            | Expr::Lt(a, b)
            | Expr::Le(a, b)
            | Expr::Gt(a, b)
            | Expr::Ge(a, b)
            | Expr::Imp(a, b)
            | Expr::Iff(a, b) => {
                a.collect_vars(out);
                b.collect_vars(out);
            }
            Expr::Add(es)
            | Expr::Mul(es)
            | Expr::Min(es)
            | Expr::Max(es)
            | Expr::And(es)
            | Expr::Or(es) => {
                for e in es {
                    e.collect_vars(out);
                }
            }
            Expr::IfThenElse(c, t, e) => {
                c.collect_vars(out);
                t.collect_vars(out);
                e.collect_vars(out);
            }
        }
    }

    /// Exact evaluation under a total assignment. `None` if undefined.
    pub fn eval(&self, val: &impl Fn(VarId) -> i64) -> Option<i64> {
        let b = |x: i64| if x != 0 { 1 } else { 0 };
        Some(match self {
            Expr::Const(c) => *c,
            Expr::Var(v) => val(*v),
            Expr::Neg(a) => -a.eval(val)?,
            Expr::Abs(a) => a.eval(val)?.abs(),
            Expr::Add(es) => {
                let mut s = 0i64;
                for e in es {
                    s += e.eval(val)?;
                }
                s
            }
            Expr::Sub(a, c) => a.eval(val)? - c.eval(val)?,
            Expr::Mul(es) => {
                let mut p = 1i64;
                for e in es {
                    p *= e.eval(val)?;
                }
                p
            }
            Expr::Div(a, c) => {
                let d = c.eval(val)?;
                if d == 0 {
                    return None;
                }
                a.eval(val)? / d
            }
            Expr::Mod(a, c) => {
                let d = c.eval(val)?;
                if d == 0 {
                    return None;
                }
                a.eval(val)? % d
            }
            Expr::Min(es) => {
                let mut m = i64::MAX;
                for e in es {
                    m = m.min(e.eval(val)?);
                }
                m
            }
            Expr::Max(es) => {
                let mut m = i64::MIN;
                for e in es {
                    m = m.max(e.eval(val)?);
                }
                m
            }
            Expr::Eq(a, c) => b((a.eval(val)? == c.eval(val)?) as i64),
            Expr::Ne(a, c) => b((a.eval(val)? != c.eval(val)?) as i64),
            Expr::Lt(a, c) => b((a.eval(val)? < c.eval(val)?) as i64),
            Expr::Le(a, c) => b((a.eval(val)? <= c.eval(val)?) as i64),
            Expr::Gt(a, c) => b((a.eval(val)? > c.eval(val)?) as i64),
            Expr::Ge(a, c) => b((a.eval(val)? >= c.eval(val)?) as i64),
            Expr::Not(a) => 1 - b(a.eval(val)?),
            Expr::And(es) => {
                let mut r = 1;
                for e in es {
                    if b(e.eval(val)?) == 0 {
                        r = 0;
                    }
                }
                r
            }
            Expr::Or(es) => {
                let mut r = 0;
                for e in es {
                    if b(e.eval(val)?) == 1 {
                        r = 1;
                    }
                }
                r
            }
            Expr::Imp(a, c) => {
                if b(a.eval(val)?) == 0 {
                    1
                } else {
                    b(c.eval(val)?)
                }
            }
            Expr::Iff(a, c) => b((b(a.eval(val)?) == b(c.eval(val)?)) as i64),
            Expr::IfThenElse(c, t, e) => {
                if b(c.eval(val)?) == 1 {
                    t.eval(val)?
                } else {
                    e.eval(val)?
                }
            }
        })
    }

    /// A sound interval `[lo, hi]` for the value of the expression given the
    /// per-variable bounds returned by `dom`.
    pub fn bounds(&self, dom: &impl Fn(VarId) -> (i64, i64)) -> (i64, i64) {
        match self {
            Expr::Const(c) => (*c, *c),
            Expr::Var(v) => dom(*v),
            Expr::Neg(a) => {
                let (l, h) = a.bounds(dom);
                (-h, -l)
            }
            Expr::Abs(a) => {
                let (l, h) = a.bounds(dom);
                if l >= 0 {
                    (l, h)
                } else if h <= 0 {
                    (-h, -l)
                } else {
                    (0, (-l).max(h))
                }
            }
            Expr::Add(es) => {
                let mut lo = 0i128;
                let mut hi = 0i128;
                for e in es {
                    let (l, h) = e.bounds(dom);
                    lo += l as i128;
                    hi += h as i128;
                }
                (clamp(lo), clamp(hi))
            }
            Expr::Sub(a, b) => {
                let (al, ah) = a.bounds(dom);
                let (bl, bh) = b.bounds(dom);
                (
                    clamp(al as i128 - bh as i128),
                    clamp(ah as i128 - bl as i128),
                )
            }
            Expr::Mul(es) => {
                let mut acc = (1i128, 1i128);
                for e in es {
                    let (l, h) = e.bounds(dom);
                    acc = imul(acc, (l as i128, h as i128));
                }
                (clamp(acc.0), clamp(acc.1))
            }
            Expr::Div(a, b) => {
                let (al, ah) = a.bounds(dom);
                let (bl, bh) = b.bounds(dom);
                if bl <= 0 && bh >= 0 {
                    // Divisor range straddles zero: |a / b| <= |a| for any
                    // non-zero integer divisor.
                    let m = al.abs().max(ah.abs());
                    (-m, m)
                } else {
                    let cands = [al / bl, al / bh, ah / bl, ah / bh];
                    (*cands.iter().min().unwrap(), *cands.iter().max().unwrap())
                }
            }
            Expr::Mod(_, b) => {
                let (bl, bh) = b.bounds(dom);
                let m = bl.abs().max(bh.abs());
                if m == 0 {
                    (0, 0)
                } else {
                    (-(m - 1), m - 1)
                }
            }
            Expr::Min(es) => fold_bounds(es, dom, i64::MAX, |a, b| a.min(b)),
            Expr::Max(es) => fold_bounds(es, dom, i64::MIN, |a, b| a.max(b)),

            Expr::Eq(a, b) => {
                let (al, ah) = a.bounds(dom);
                let (bl, bh) = b.bounds(dom);
                if ah < bl || bh < al {
                    FALSE
                } else if al == ah && bl == bh && al == bl {
                    TRUE
                } else {
                    MAYBE
                }
            }
            Expr::Ne(a, b) => {
                let (al, ah) = a.bounds(dom);
                let (bl, bh) = b.bounds(dom);
                if ah < bl || bh < al {
                    TRUE
                } else if al == ah && bl == bh && al == bl {
                    FALSE
                } else {
                    MAYBE
                }
            }
            Expr::Lt(a, b) => cmp_bounds(a, b, dom, Cmp::Lt),
            Expr::Le(a, b) => cmp_bounds(a, b, dom, Cmp::Le),
            Expr::Gt(a, b) => cmp_bounds(b, a, dom, Cmp::Lt),
            Expr::Ge(a, b) => cmp_bounds(b, a, dom, Cmp::Le),

            Expr::Not(a) => match a.bounds(dom) {
                TRUE => FALSE,
                FALSE => TRUE,
                _ => MAYBE,
            },
            Expr::And(es) => {
                let mut all_true = true;
                for e in es {
                    match e.bounds(dom) {
                        FALSE => return FALSE,
                        TRUE => {}
                        _ => all_true = false,
                    }
                }
                if all_true {
                    TRUE
                } else {
                    MAYBE
                }
            }
            Expr::Or(es) => {
                let mut all_false = true;
                for e in es {
                    match e.bounds(dom) {
                        TRUE => return TRUE,
                        FALSE => {}
                        _ => all_false = false,
                    }
                }
                if all_false {
                    FALSE
                } else {
                    MAYBE
                }
            }
            Expr::Imp(a, b) => {
                // a -> b  ==  (!a) or b
                match a.bounds(dom) {
                    FALSE => TRUE,
                    TRUE => b.bounds(dom),
                    _ => MAYBE,
                }
            }
            Expr::Iff(a, b) => {
                let ab = a.bounds(dom);
                let bb = b.bounds(dom);
                if (ab == TRUE || ab == FALSE) && (bb == TRUE || bb == FALSE) {
                    if ab == bb {
                        TRUE
                    } else {
                        FALSE
                    }
                } else {
                    MAYBE
                }
            }
            Expr::IfThenElse(c, t, e) => match c.bounds(dom) {
                TRUE => t.bounds(dom),
                FALSE => e.bounds(dom),
                _ => {
                    let (tl, th) = t.bounds(dom);
                    let (el, eh) = e.bounds(dom);
                    (tl.min(el), th.max(eh))
                }
            },
        }
    }
}

/// Boolean-interval sentinels (subsets of `[0, 1]`).
const FALSE: (i64, i64) = (0, 0);
const TRUE: (i64, i64) = (1, 1);
const MAYBE: (i64, i64) = (0, 1);

enum Cmp {
    Lt,
    Le,
}

/// Bounds for `a < b` or `a <= b`.
fn cmp_bounds(a: &Expr, b: &Expr, dom: &impl Fn(VarId) -> (i64, i64), cmp: Cmp) -> (i64, i64) {
    let (al, ah) = a.bounds(dom);
    let (bl, bh) = b.bounds(dom);
    match cmp {
        Cmp::Lt => {
            if ah < bl {
                TRUE
            } else if al >= bh {
                FALSE
            } else {
                MAYBE
            }
        }
        Cmp::Le => {
            if ah <= bl {
                TRUE
            } else if al > bh {
                FALSE
            } else {
                MAYBE
            }
        }
    }
}

fn fold_bounds(
    es: &[Expr],
    dom: &impl Fn(VarId) -> (i64, i64),
    init: i64,
    f: impl Fn(i64, i64) -> i64,
) -> (i64, i64) {
    let mut lo = init;
    let mut hi = init;
    for e in es {
        let (l, h) = e.bounds(dom);
        lo = f(lo, l);
        hi = f(hi, h);
    }
    (lo, hi)
}

/// Multiply two `i128` intervals (corners give the extremes).
fn imul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    let p = [a.0 * b.0, a.0 * b.1, a.1 * b.0, a.1 * b.1];
    (*p.iter().min().unwrap(), *p.iter().max().unwrap())
}

/// Saturate an `i128` into `i64`.
fn clamp(x: i128) -> i64 {
    x.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

// --- ergonomic constructors ---

/// Constant.
pub fn int(c: i64) -> Expr {
    Expr::Const(c)
}
/// Variable reference.
pub fn var(v: VarId) -> Expr {
    Expr::Var(v)
}
/// Negation.
pub fn neg(a: Expr) -> Expr {
    Expr::Neg(Box::new(a))
}
/// Absolute value.
pub fn abs(a: Expr) -> Expr {
    Expr::Abs(Box::new(a))
}
/// Sum.
pub fn add(es: Vec<Expr>) -> Expr {
    Expr::Add(es)
}
/// Difference.
pub fn sub(a: Expr, b: Expr) -> Expr {
    Expr::Sub(Box::new(a), Box::new(b))
}
/// Product.
pub fn mul(es: Vec<Expr>) -> Expr {
    Expr::Mul(es)
}
/// Integer division.
pub fn div(a: Expr, b: Expr) -> Expr {
    Expr::Div(Box::new(a), Box::new(b))
}
/// Remainder.
pub fn rem(a: Expr, b: Expr) -> Expr {
    Expr::Mod(Box::new(a), Box::new(b))
}
/// Minimum.
pub fn min_of(es: Vec<Expr>) -> Expr {
    Expr::Min(es)
}
/// Maximum.
pub fn max_of(es: Vec<Expr>) -> Expr {
    Expr::Max(es)
}
/// `a == b`.
pub fn eq(a: Expr, b: Expr) -> Expr {
    Expr::Eq(Box::new(a), Box::new(b))
}
/// `a != b`.
pub fn ne(a: Expr, b: Expr) -> Expr {
    Expr::Ne(Box::new(a), Box::new(b))
}
/// `a < b`.
pub fn lt(a: Expr, b: Expr) -> Expr {
    Expr::Lt(Box::new(a), Box::new(b))
}
/// `a <= b`.
pub fn le(a: Expr, b: Expr) -> Expr {
    Expr::Le(Box::new(a), Box::new(b))
}
/// `a > b`.
pub fn gt(a: Expr, b: Expr) -> Expr {
    Expr::Gt(Box::new(a), Box::new(b))
}
/// `a >= b`.
pub fn ge(a: Expr, b: Expr) -> Expr {
    Expr::Ge(Box::new(a), Box::new(b))
}
/// Logical negation.
pub fn not(a: Expr) -> Expr {
    Expr::Not(Box::new(a))
}
/// Conjunction.
pub fn and(es: Vec<Expr>) -> Expr {
    Expr::And(es)
}
/// Disjunction.
pub fn or(es: Vec<Expr>) -> Expr {
    Expr::Or(es)
}
/// Implication.
pub fn imp(a: Expr, b: Expr) -> Expr {
    Expr::Imp(Box::new(a), Box::new(b))
}
/// Bi-implication.
pub fn iff(a: Expr, b: Expr) -> Expr {
    Expr::Iff(Box::new(a), Box::new(b))
}
/// Conditional.
pub fn ite(cond: Expr, then_: Expr, else_: Expr) -> Expr {
    Expr::IfThenElse(Box::new(cond), Box::new(then_), Box::new(else_))
}
