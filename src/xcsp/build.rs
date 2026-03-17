//! Build a [`Solver`] from a parsed XCSP3 DOM.
//!
//! A useful subset of XCSP3-core is interpreted; anything not yet handled
//! returns a clear `Err` rather than silently mis-modelling the instance.

use crate::constraints::count::{cardinality, count, n_values};
use crate::constraints::graph::circuit;
use crate::constraints::lex::{channel, lex_chain};
use crate::constraints::linear::{linear, Relation};
use crate::constraints::primitives::{
    all_different, all_equal, element, instantiation, maximum, minimum, ordered,
};
use crate::constraints::scheduling::{cumulative, no_overlap};
use crate::constraints::table::extension;
use crate::expr::Expr;
use crate::ids::VarId;
use crate::store::Solver;
use crate::xcsp::dom::Node;
use crate::xcsp::parse::{
    expand_indices, interval_values, parse_condition, parse_expr, parse_intervals, parse_rel,
    parse_var_refs, ArrayInfo, Condition, Operand, SymTab,
};

/// A model built from an XCSP3 instance.
pub struct Built {
    /// The solver with all variables and constraints posted.
    pub solver: Solver,
    /// All decision variables (and the objective var, if any), in declaration order.
    pub vars: Vec<VarId>,
    /// `Some((minimize, obj))` for a COP, `None` for a CSP.
    pub objective: Option<(bool, VarId)>,
}

fn create_var(solver: &mut Solver, intervals: &[(i32, i32)]) -> VarId {
    if intervals.len() == 1 {
        solver.new_var_range(intervals[0].0, intervals[0].1)
    } else {
        solver.new_var_set(&interval_values(intervals))
    }
}

/// Create an array's element variables. Domain is taken either from the
/// `<array>` element's own text (one domain for all) or from `<domain for=...>`
/// children (a per-element domain, with an optional `for="others"` default).
fn build_array_vars(
    solver: &mut Solver,
    c: &Node,
    dims: &[usize],
    total: usize,
) -> Result<Vec<VarId>, String> {
    let text = c.trimmed();
    if !text.is_empty() {
        let intervals = parse_intervals(text)?;
        return Ok((0..total).map(|_| create_var(solver, &intervals)).collect());
    }

    let mut doms: Vec<Option<Vec<(i32, i32)>>> = vec![None; total];
    let mut default: Option<Vec<(i32, i32)>> = None;
    for d in c.children_named("domain") {
        let intervals = parse_intervals(d.trimmed())?;
        match d.attr("for") {
            None => default = Some(intervals),
            Some(f) if f.trim() == "others" => default = Some(intervals),
            Some(f) => {
                for tok in f.split_whitespace() {
                    if tok.contains('[') {
                        for idx in expand_indices(dims, tok)? {
                            doms[idx] = Some(intervals.clone());
                        }
                    } else {
                        // A bare array name: applies to every element.
                        for slot in doms.iter_mut() {
                            *slot = Some(intervals.clone());
                        }
                    }
                }
            }
        }
    }

    let mut flat = Vec::with_capacity(total);
    for (i, slot) in doms.iter().enumerate() {
        let intervals = slot
            .as_ref()
            .or(default.as_ref())
            .ok_or_else(|| format!("array element {i} has no domain"))?;
        flat.push(create_var(solver, intervals));
    }
    Ok(flat)
}

/// Parse a whitespace-separated integer list, expanding XCSP `value×count`
/// repetition written as `vxk` (e.g. `0x20` = twenty zeros).
fn parse_ints(s: &str) -> Result<Vec<i64>, String> {
    let mut out = Vec::new();
    for tok in s.split_whitespace() {
        match tok.split_once('x') {
            Some((v, k)) => {
                let val: i64 = v.parse().map_err(|_| format!("bad int `{tok}`"))?;
                let count: usize = k.parse().map_err(|_| format!("bad count `{tok}`"))?;
                out.extend(std::iter::repeat_n(val, count));
            }
            None => out.push(tok.parse().map_err(|_| format!("bad int `{tok}`"))?),
        }
    }
    Ok(out)
}

fn list_refs(node: &Node, sym: &SymTab) -> Result<Vec<VarId>, String> {
    // Most constraints wrap their scope in <list>; some (e.g. circuit) put the
    // references directly in the element text.
    let (src, start_index) = match node.child("list") {
        Some(l) => (l, l.attr("startIndex")),
        None => (node, node.attr("startIndex")),
    };
    if start_index.is_some_and(|s| s.trim() != "0") {
        return Err("non-zero startIndex unsupported".to_string());
    }
    parse_var_refs(src.trimmed(), sym)
}

/// Build the model from the root `<instance>` node.
pub fn build(root: &Node) -> Result<Built, String> {
    let mut solver = Solver::new();
    let mut sym = SymTab::default();
    let mut order: Vec<VarId> = Vec::new();

    // --- variables ---
    if let Some(vars) = root.child("variables") {
        for c in &vars.children {
            match c.tag.as_str() {
                "var" => {
                    let id = c.attr("id").ok_or("<var> without id")?.to_string();
                    let intervals = parse_intervals(c.trimmed())?;
                    let v = create_var(&mut solver, &intervals);
                    sym.scalars.insert(id, v);
                    order.push(v);
                }
                "array" => {
                    let id = c.attr("id").ok_or("<array> without id")?.to_string();
                    let size = c.attr("size").ok_or("<array> without size")?;
                    let dims: Vec<usize> = size
                        .split('[')
                        .filter(|p| !p.is_empty())
                        .map(|p| {
                            p.trim_end_matches(']')
                                .trim()
                                .parse::<usize>()
                                .map_err(|_| "bad array size".to_string())
                        })
                        .collect::<Result<_, _>>()?;
                    let total: usize = dims.iter().product();
                    let flat = build_array_vars(&mut solver, c, &dims, total)?;
                    order.extend(flat.iter().copied());
                    sym.arrays.insert(id, ArrayInfo { dims, flat });
                }
                _ => {}
            }
        }
    }

    // --- constraints ---
    if let Some(cons) = root.child("constraints") {
        for c in &cons.children {
            build_constraint(&mut solver, &sym, c)?;
        }
    }

    // --- objective ---
    let objective = match root.child("objectives") {
        Some(objs) => Some(build_objective(&mut solver, &sym, objs, &mut order)?),
        None => None,
    };

    Ok(Built {
        solver,
        vars: order,
        objective,
    })
}

fn build_constraint(solver: &mut Solver, sym: &SymTab, c: &Node) -> Result<(), String> {
    match c.tag.as_str() {
        "intension" => {
            let text = c
                .child("function")
                .map_or_else(|| c.trimmed(), Node::trimmed);
            let e = parse_expr(text, sym)?;
            crate::constraints::intension::intension(solver, e);
        }
        "extension" => {
            let vars = list_refs(c, sym)?;
            let (positive, tuples_node) = match (c.child("supports"), c.child("conflicts")) {
                (Some(s), _) => (true, s),
                (_, Some(s)) => (false, s),
                _ => return Err("extension without supports/conflicts".to_string()),
            };
            let tuples = parse_tuples(tuples_node.trimmed(), vars.len())?;
            extension(solver, &vars, &tuples, positive);
        }
        "allDifferent" => all_different(solver, &list_refs(c, sym)?),
        "allEqual" => all_equal(solver, &list_refs(c, sym)?),
        "sum" => build_sum(solver, sym, c)?,
        "count" => build_count(solver, sym, c)?,
        "nValues" => {
            let vars = list_refs(c, sym)?;
            let cond = condition_of(c, sym)?;
            match cond.operand {
                Operand::Const(k) => n_values(solver, &vars, cond.rel, k),
                Operand::Var(_) => return Err("nValues with variable rhs unsupported".to_string()),
            }
        }
        "element" => build_element(solver, sym, c)?,
        "minimum" => build_min_max(solver, sym, c, true)?,
        "maximum" => build_min_max(solver, sym, c, false)?,
        "ordered" => {
            let vars = list_refs(c, sym)?;
            let op = c.child("operator").ok_or("ordered without <operator>")?;
            ordered(solver, &vars, parse_rel(op.trimmed())?);
        }
        "lex" => build_lex(solver, sym, c)?,
        "channel" => build_channel(solver, sym, c)?,
        "circuit" => circuit(solver, &list_refs(c, sym)?),
        "instantiation" => {
            let vars = list_refs(c, sym)?;
            let values = c.child("values").ok_or("instantiation without <values>")?;
            let vals: Vec<i32> = parse_ints(values.trimmed())?
                .into_iter()
                .map(|v| v as i32)
                .collect();
            instantiation(solver, &vars, &vals);
        }
        "cardinality" => build_cardinality(solver, sym, c)?,
        "noOverlap" => build_no_overlap(solver, sym, c)?,
        "cumulative" => build_cumulative(solver, sym, c)?,
        "group" => build_group(solver, sym, c)?,
        "block" => {
            for child in &c.children {
                build_constraint(solver, sym, child)?;
            }
        }
        other => return Err(format!("unsupported constraint <{other}>")),
    }
    Ok(())
}

/// Expand a `<group>`: a template constraint applied once per `<args>` line,
/// substituting `%...` (all args) and `%0`, `%1`, … (positional args) into the
/// template's text and attributes.
fn build_group(solver: &mut Solver, sym: &SymTab, c: &Node) -> Result<(), String> {
    let template = c
        .children
        .iter()
        .find(|n| n.tag != "args")
        .ok_or("group without a template constraint")?;
    for args in c.children_named("args") {
        let tokens: Vec<&str> = args.trimmed().split_whitespace().collect();
        let mut inst = template.clone();
        subst_in_place(&mut inst, &tokens);
        build_constraint(solver, sym, &inst)?;
    }
    Ok(())
}

fn subst_str(s: &str, tokens: &[&str]) -> String {
    let mut r = s.replace("%...", &tokens.join(" "));
    // Replace higher indices first so `%10` is not clobbered by `%1`.
    for i in (0..tokens.len()).rev() {
        r = r.replace(&format!("%{i}"), tokens[i]);
    }
    r
}

fn subst_in_place(n: &mut Node, tokens: &[&str]) {
    n.text = subst_str(&n.text, tokens);
    for (_, v) in n.attrs.iter_mut() {
        *v = subst_str(v, tokens);
    }
    for child in &mut n.children {
        subst_in_place(child, tokens);
    }
}

fn condition_of(c: &Node, sym: &SymTab) -> Result<Condition, String> {
    let cond = c.child("condition").ok_or("missing <condition>")?;
    parse_condition(cond.trimmed(), sym)
}

fn build_sum(solver: &mut Solver, sym: &SymTab, c: &Node) -> Result<(), String> {
    let mut vars = list_refs(c, sym)?;
    let mut coeffs = match c.child("coeffs") {
        Some(n) => parse_ints(n.trimmed())?,
        None => vec![1; vars.len()],
    };
    let cond = condition_of(c, sym)?;
    match cond.operand {
        Operand::Const(k) => linear(solver, &coeffs, &vars, cond.rel, k),
        Operand::Var(y) => {
            // sum(coeffs*vars) rel y  <=>  sum(coeffs*vars) - y rel 0
            coeffs.push(-1);
            vars.push(y);
            linear(solver, &coeffs, &vars, cond.rel, 0);
        }
    }
    Ok(())
}

fn build_count(solver: &mut Solver, sym: &SymTab, c: &Node) -> Result<(), String> {
    let vars = list_refs(c, sym)?;
    let values = c.child("values").ok_or("count without <values>")?;
    let value: i32 = values
        .trimmed()
        .parse()
        .map_err(|_| "count expects one value")?;
    let cond = condition_of(c, sym)?;
    match cond.operand {
        Operand::Const(k) => count(solver, &vars, value, cond.rel, k),
        Operand::Var(_) => return Err("count with variable rhs unsupported".to_string()),
    }
    Ok(())
}

/// Expand a list whose tokens may be variable references *or* integer
/// constants (constants become fixed singleton variables).
fn refs_or_consts(solver: &mut Solver, sym: &SymTab, s: &str) -> Result<Vec<VarId>, String> {
    let mut out = Vec::new();
    for tok in s.split_whitespace() {
        if let Ok(n) = tok.parse::<i32>() {
            out.push(solver.new_var_set(&[n]));
        } else {
            out.extend(parse_var_refs(tok, sym)?);
        }
    }
    Ok(out)
}

fn build_element(solver: &mut Solver, sym: &SymTab, c: &Node) -> Result<(), String> {
    let list_src = c
        .child("list")
        .map_or_else(|| c.trimmed().to_string(), |l| l.trimmed().to_string());
    let array = refs_or_consts(solver, sym, &list_src)?;
    let index = c.child("index").ok_or("element without <index>")?;
    let idx = sym.resolve_one(index.trimmed())?;
    let value = if let Some(v) = c.child("value") {
        let t = v.trimmed();
        match t.parse::<i32>() {
            Ok(n) => solver.new_var_set(&[n]),
            Err(_) => sym.resolve_one(t)?,
        }
    } else {
        let cond = condition_of(c, sym)?;
        if cond.rel != Relation::Eq {
            return Err("element condition must be eq".to_string());
        }
        match cond.operand {
            Operand::Const(n) => solver.new_var_set(&[n as i32]),
            Operand::Var(v) => v,
        }
    };
    element(solver, &array, idx, value);
    Ok(())
}

fn build_min_max(solver: &mut Solver, sym: &SymTab, c: &Node, is_min: bool) -> Result<(), String> {
    let xs = list_refs(c, sym)?;
    let cond = condition_of(c, sym)?;
    if cond.rel != Relation::Eq {
        return Err("minimum/maximum condition must be eq".to_string());
    }
    let y = match cond.operand {
        Operand::Const(n) => solver.new_var_set(&[n as i32]),
        Operand::Var(v) => v,
    };
    if is_min {
        minimum(solver, y, &xs);
    } else {
        maximum(solver, y, &xs);
    }
    Ok(())
}

fn build_lex(solver: &mut Solver, sym: &SymTab, c: &Node) -> Result<(), String> {
    let mut rows: Vec<Vec<VarId>> = Vec::new();
    for list in c.children_named("list") {
        rows.push(parse_var_refs(list.trimmed(), sym)?);
    }
    if rows.len() < 2 {
        return Err("lex needs at least two lists".to_string());
    }
    let op = c.child("operator").ok_or("lex without <operator>")?;
    let rel = parse_rel(op.trimmed())?;
    let strict = matches!(rel, Relation::Lt | Relation::Gt);
    if matches!(rel, Relation::Gt | Relation::Ge) {
        rows.reverse();
    }
    lex_chain(solver, &rows, strict);
    Ok(())
}

fn build_channel(solver: &mut Solver, sym: &SymTab, c: &Node) -> Result<(), String> {
    let lists: Vec<Vec<VarId>> = c
        .children_named("list")
        .map(|l| parse_var_refs(l.trimmed(), sym))
        .collect::<Result<_, _>>()?;
    match lists.len() {
        1 => channel(solver, &lists[0], &lists[0]),
        2 => channel(solver, &lists[0], &lists[1]),
        _ => return Err("channel expects one or two lists".to_string()),
    }
    Ok(())
}

fn build_cardinality(solver: &mut Solver, sym: &SymTab, c: &Node) -> Result<(), String> {
    let vars = list_refs(c, sym)?;
    let values_node = c.child("values").ok_or("cardinality without <values>")?;
    let values: Vec<i32> = values_node
        .trimmed()
        .split_whitespace()
        .map(|t| t.parse().map_err(|_| "bad value"))
        .collect::<Result<_, _>>()?;
    let occurs = c.child("occurs").ok_or("cardinality without <occurs>")?;
    let mut low = Vec::new();
    let mut high = Vec::new();
    for tok in occurs.trimmed().split_whitespace() {
        if let Some((a, b)) = tok.split_once("..") {
            low.push(a.parse().map_err(|_| "bad occurs")?);
            high.push(b.parse().map_err(|_| "bad occurs")?);
        } else {
            let v: i64 = tok.parse().map_err(|_| "bad occurs")?;
            low.push(v);
            high.push(v);
        }
    }
    let closed = c.attr("closed") == Some("true");
    cardinality(solver, &vars, &values, &low, &high, closed);
    Ok(())
}

fn build_cumulative(solver: &mut Solver, sym: &SymTab, c: &Node) -> Result<(), String> {
    let origins = c.child("origins").ok_or("cumulative without <origins>")?;
    let starts = parse_var_refs(origins.trimmed(), sym)?;
    let lengths = c.child("lengths").ok_or("cumulative without <lengths>")?;
    let durations = parse_ints(lengths.trimmed())?;
    let heights_node = c.child("heights").ok_or("cumulative without <heights>")?;
    let heights = parse_ints(heights_node.trimmed())?;
    let cond = condition_of(c, sym)?;
    if cond.rel != Relation::Le {
        return Err("cumulative condition must be <=".to_string());
    }
    let cap = match cond.operand {
        Operand::Const(k) => k,
        Operand::Var(_) => return Err("cumulative with variable capacity unsupported".to_string()),
    };
    if durations.len() != starts.len() || heights.len() != starts.len() {
        return Err("cumulative origins/lengths/heights mismatch".to_string());
    }
    cumulative(solver, &starts, &durations, &heights, cap);
    Ok(())
}

fn build_no_overlap(solver: &mut Solver, sym: &SymTab, c: &Node) -> Result<(), String> {
    let origins = c.child("origins").ok_or("noOverlap without <origins>")?;
    let starts = parse_var_refs(origins.trimmed(), sym)?;
    let lengths = c.child("lengths").ok_or("noOverlap without <lengths>")?;
    let durations = parse_ints(lengths.trimmed())?;
    if durations.len() != starts.len() {
        return Err("noOverlap origins/lengths mismatch".to_string());
    }
    no_overlap(solver, &starts, &durations);
    Ok(())
}

fn parse_tuples(s: &str, arity: usize) -> Result<Vec<Vec<i32>>, String> {
    let mut tuples = Vec::new();
    if arity == 1 {
        // Flat list of values, possibly "(v)" wrapped.
        for tok in s.split_whitespace() {
            let t = tok.trim_matches(|ch| ch == '(' || ch == ')');
            tuples.push(vec![t.parse().map_err(|_| "bad tuple value")?]);
        }
        return Ok(tuples);
    }
    // "(a,b,c)(d,e,f)..."
    for raw in s.split(')') {
        let raw = raw.trim();
        let raw = raw.trim_start_matches('(');
        if raw.is_empty() {
            continue;
        }
        let tuple: Vec<i32> = raw
            .split(',')
            .map(|t| t.trim().parse().map_err(|_| "bad tuple value"))
            .collect::<Result<_, _>>()?;
        if tuple.len() != arity {
            return Err("tuple arity mismatch".to_string());
        }
        tuples.push(tuple);
    }
    Ok(tuples)
}

fn build_objective(
    solver: &mut Solver,
    sym: &SymTab,
    objs: &Node,
    order: &mut Vec<VarId>,
) -> Result<(bool, VarId), String> {
    let (minimize, node) = match (objs.child("minimize"), objs.child("maximize")) {
        (Some(n), _) => (true, n),
        (_, Some(n)) => (false, n),
        _ => return Err("empty <objectives>".to_string()),
    };

    // type="sum" with <list> (+ optional <coeffs>): introduce an objective var
    // equal to the weighted sum.
    if node.attr("type") == Some("sum") {
        let list_src = node
            .child("list")
            .map_or_else(|| node.trimmed().to_string(), |l| l.trimmed().to_string());
        let vars = parse_var_refs(&list_src, sym)?;
        let coeffs = match node.child("coeffs") {
            Some(n) => parse_ints(n.trimmed())?,
            None => vec![1; vars.len()],
        };
        let (lo, hi) = sum_bounds(solver, &coeffs, &vars);
        let obj = solver.new_var_range(lo, hi);
        // sum(coeffs*vars) - obj = 0
        let mut cc = coeffs;
        cc.push(-1);
        let mut vv = vars;
        vv.push(obj);
        linear(solver, &cc, &vv, Relation::Eq, 0);
        order.push(obj);
        return Ok((minimize, obj));
    }

    // A bare single-variable objective.
    if let Some(list) = node.child("list") {
        let refs = parse_var_refs(list.trimmed(), sym)?;
        if refs.len() == 1 {
            return Ok((minimize, refs[0]));
        }
        return Err("only sum / single-var list objectives supported".to_string());
    }

    // An expression objective (attribute or text).
    let expr_text = node
        .attr("expression")
        .map(str::to_string)
        .unwrap_or_else(|| node.trimmed().to_string());
    if expr_text.is_empty() {
        return Err("unsupported objective form".to_string());
    }
    let e = parse_expr(&expr_text, sym)?;
    let (lo, hi) = expr_bounds(solver, &e);
    let obj = solver.new_var_range(lo, hi);
    crate::constraints::intension::intension(solver, crate::expr::eq(crate::expr::var(obj), e));
    order.push(obj);
    Ok((minimize, obj))
}

fn sum_bounds(solver: &Solver, coeffs: &[i64], vars: &[VarId]) -> (i32, i32) {
    let mut lo: i64 = 0;
    let mut hi: i64 = 0;
    for (&a, &v) in coeffs.iter().zip(vars) {
        let (vmin, vmax) = (solver.store.min(v) as i64, solver.store.max(v) as i64);
        let (tmin, tmax) = if a >= 0 {
            (a * vmin, a * vmax)
        } else {
            (a * vmax, a * vmin)
        };
        lo += tmin;
        hi += tmax;
    }
    (clamp(lo), clamp(hi))
}

fn expr_bounds(solver: &Solver, e: &Expr) -> (i32, i32) {
    let (lo, hi) = e.bounds(&|v| (solver.store.min(v) as i64, solver.store.max(v) as i64));
    (clamp(lo), clamp(hi))
}

fn clamp(x: i64) -> i32 {
    x.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}
