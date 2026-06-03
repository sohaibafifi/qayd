//! Small decompositions used by front-ends for higher-level constraints.

use crate::constraints::intension::intension;
use crate::constraints::linear::{linear, Relation};
use crate::expr::{self, Expr};
use crate::ids::VarId;
use crate::store::Solver;

pub(crate) fn clamp_i64(x: i64) -> i32 {
    x.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

pub(crate) fn aux_for_expr(solver: &mut Solver, e: Expr) -> VarId {
    let (lo, hi) = e.bounds(&|v| (solver.store.min(v) as i64, solver.store.max(v) as i64));
    let aux = solver.new_var_range(clamp_i64(lo), clamp_i64(hi));
    intension(solver, expr::eq(expr::var(aux), e));
    aux
}

pub(crate) fn eq_const_indicator(solver: &mut Solver, var: VarId, value: i32) -> VarId {
    aux_for_expr(solver, expr::eq(expr::var(var), expr::int(value as i64)))
}

fn in_values_expr(var: VarId, values: &[i32]) -> Expr {
    match values {
        [] => expr::int(0),
        [value] => expr::eq(expr::var(var), expr::int(*value as i64)),
        _ => expr::or(values.iter().map(|&value| expr::eq(expr::var(var), expr::int(value as i64))).collect()),
    }
}

fn any_eq_expr(vars: &[VarId], value: i32) -> Expr {
    expr::or(vars.iter().map(|&var| expr::eq(expr::var(var), expr::int(value as i64))).collect())
}

pub(crate) fn post_tuple_not_equal(solver: &mut Solver, left: &[VarId], right: &[VarId]) -> Result<(), String> {
    if left.len() != right.len() {
        return Err("allDifferent-list: ragged tuples".to_string());
    }
    let diffs = left.iter().zip(right).map(|(&x, &y)| expr::ne(expr::var(x), expr::var(y))).collect();
    intension(solver, expr::or(diffs));
    Ok(())
}

pub(crate) fn post_all_different_except(solver: &mut Solver, vars: &[VarId], except: &[i32]) {
    for i in 0..vars.len() {
        for j in (i + 1)..vars.len() {
            let mut terms = vec![expr::ne(expr::var(vars[i]), expr::var(vars[j]))];
            terms.extend(except.iter().map(|&e| expr::eq(expr::var(vars[i]), expr::int(e as i64))));
            intension(solver, expr::or(terms));
        }
    }
}

pub(crate) fn post_channel_onehot_index(solver: &mut Solver, xs: &[VarId], value: VarId, start_index: i32) {
    for (i, &x) in xs.iter().enumerate() {
        let idx = start_index as i64 + i as i64;
        intension(solver, expr::iff(expr::eq(expr::var(x), expr::int(1)), expr::eq(expr::var(value), expr::int(idx))));
    }
}

pub(crate) fn post_bin_loads(solver: &mut Solver, items: &[VarId], sizes: &[i64], loads: &[VarId]) {
    for (bin, &load) in loads.iter().enumerate() {
        let mut coeffs = Vec::with_capacity(items.len() + 1);
        let mut vars = Vec::with_capacity(items.len() + 1);
        for (i, &item) in items.iter().enumerate() {
            coeffs.push(sizes[i]);
            vars.push(eq_const_indicator(solver, item, bin as i32));
        }
        coeffs.push(-1);
        vars.push(load);
        linear(solver, &coeffs, &vars, Relation::Eq, 0);
    }
}

pub(crate) fn post_diffn(solver: &mut Solver, origins: &[Vec<VarId>], lengths: &[Vec<Expr>], zero_ignored: bool) -> Result<(), String> {
    if origins.len() != lengths.len() {
        return Err("noOverlap: origins/lengths length mismatch".to_string());
    }
    for i in 0..origins.len() {
        if origins[i].len() != lengths[i].len() {
            return Err("noOverlap: box origin/length arity mismatch".to_string());
        }
    }
    for i in 0..origins.len() {
        for j in (i + 1)..origins.len() {
            let mut terms = Vec::with_capacity(2 * origins[i].len() + usize::from(zero_ignored) * (lengths[i].len() + lengths[j].len()));
            for d in 0..origins[i].len() {
                terms.push(expr::le(expr::add(vec![expr::var(origins[i][d]), lengths[i][d].clone()]), expr::var(origins[j][d])));
                terms.push(expr::le(expr::add(vec![expr::var(origins[j][d]), lengths[j][d].clone()]), expr::var(origins[i][d])));
            }
            if zero_ignored {
                terms.extend(lengths[i].iter().chain(&lengths[j]).map(|len| expr::eq(len.clone(), expr::int(0))));
            }
            intension(solver, expr::or(terms));
        }
    }
    Ok(())
}

pub(crate) fn nvalues_var(solver: &mut Solver, vars: &[VarId]) -> VarId {
    let mut vals = Vec::new();
    for &v in vars {
        vals.extend(solver.store.values(v));
    }
    vals.sort_unstable();
    vals.dedup();
    let mut present = Vec::with_capacity(vals.len() + 1);
    for val in vals {
        present.push(aux_for_expr(solver, any_eq_expr(vars, val)));
    }
    let n = present.len();
    let aux = solver.new_var_range(if vars.is_empty() { 0 } else { 1 }, n as i32);
    let mut coeffs = vec![1i64; n];
    coeffs.push(-1);
    present.push(aux);
    linear(solver, &coeffs, &present, Relation::Eq, 0);
    aux
}

pub(crate) fn count_values_to_var(solver: &mut Solver, vars: &[VarId], values: &[i32], rel: Relation, target: VarId) {
    let mut terms = Vec::with_capacity(vars.len() + 1);
    for &var in vars {
        terms.push(aux_for_expr(solver, in_values_expr(var, values)));
    }
    let mut coeffs = vec![1i64; terms.len()];
    coeffs.push(-1);
    terms.push(target);
    linear(solver, &coeffs, &terms, rel, 0);
}
