//! Bridge from the `xcsp3-rust-parser` callback interface to the solver.
//!
//! The parser handles all the XCSP3 surface syntax (arrays, ranges, groups,
//! slides, compact tables, …) and calls these methods with structured data;
//! each maps onto a posting helper. Unsupported forms set `error` so the caller
//! reports a clear failure instead of silently mis-modelling.

use std::collections::HashMap;

use xcsp3_rust_parser::data_structs::expression_tree::xcsp3_utils::{
    ExpressionTree, Operator as EOp, TreeNode,
};
use xcsp3_rust_parser::data_structs::xrelational_operand::xcsp3_core::Operand;
use xcsp3_rust_parser::data_structs::xrelational_operator::xcsp3_core::Operator as ROp;
use xcsp3_rust_parser::objectives::xobjective_element::xcsp3_core::XElementOperator;
use xcsp3_rust_parser::xcsp_callback::XcspCallback;
use xcsp3_rust_parser::xcsp_xml::xcsp_xml_model::xcsp3_xml::InstanceType;

use crate::constraints::count::{cardinality, count, n_values};
use crate::constraints::graph::circuit;
use crate::constraints::lex::{channel, lex_chain};
use crate::constraints::linear::{linear, Relation};
use crate::constraints::primitives::{
    all_different, all_equal, element, instantiation, maximum, minimum, ordered,
};
use crate::constraints::scheduling::{bin_packing, cumulative, cumulative_var, no_overlap};
use crate::constraints::table::{extension, mdd, regular, Dfa, Mdd, MddArc, STAR};
use crate::expr::{self, Expr};
use crate::ids::VarId;
use crate::store::Solver;

/// Accumulates the model as the parser walks the instance.
pub struct Model {
    pub solver: Solver,
    pub order: Vec<VarId>,
    pub objective: Option<(bool, VarId)>,
    pub error: Option<String>,
    ids: HashMap<String, VarId>,
    /// Declared variables in declaration order, for expanding whole-array
    /// scope references like `x[]` / `x[][]` that the parser leaves un-expanded.
    decls: Vec<(String, VarId)>,
}

/// Right-hand side of a condition.
enum Rhs {
    Const(i64),
    Var(VarId),
}

impl Model {
    pub fn new() -> Self {
        Self {
            solver: Solver::new(),
            order: Vec::new(),
            objective: None,
            error: None,
            ids: HashMap::new(),
            decls: Vec::new(),
        }
    }

    fn fail(&mut self, msg: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(msg.into());
        }
    }

    fn var_id(&self, name: &str) -> Result<VarId, String> {
        self.ids
            .get(name)
            .copied()
            .ok_or_else(|| format!("unknown variable `{name}`"))
    }

    fn scope(&self, list: &[String]) -> Result<Vec<VarId>, String> {
        let mut out = Vec::with_capacity(list.len());
        for s in list {
            if let Some(cells) = self.expand_array_ref(s) {
                out.extend(cells);
            } else {
                out.push(self.var_id(s)?);
            }
        }
        Ok(out)
    }

    /// Expand a whole-array reference (`x[]`, `x[][]`, …) into its declared
    /// cells in declaration order. Returns `None` for ordinary scalar/cell refs.
    fn expand_array_ref(&self, s: &str) -> Option<Vec<VarId>> {
        let open = s.find('[')?;
        let (base, rest) = s.split_at(open);
        let depth = rest.matches("[]").count();
        // Only handle pure trailing empty brackets: `[]`, `[][]`, …
        if depth == 0 || rest != "[]".repeat(depth) {
            return None;
        }
        let prefix = format!("{base}[");
        let cells: Vec<VarId> = self
            .decls
            .iter()
            .filter(|(nm, _)| nm.starts_with(&prefix) && nm.matches('[').count() == depth)
            .map(|(_, v)| *v)
            .collect();
        Some(cells)
    }

    /// A fresh fixed variable holding `value`.
    fn constant(&mut self, value: i32) -> VarId {
        self.solver.new_var_set(&[value])
    }

    fn rel(op: ROp) -> Result<Relation, String> {
        Ok(match op {
            ROp::Lt => Relation::Lt,
            ROp::Le => Relation::Le,
            ROp::Ge => Relation::Ge,
            ROp::Gt => Relation::Gt,
            ROp::Eq => Relation::Eq,
            ROp::Ne => Relation::Ne,
            ROp::In | ROp::Notin => return Err("set operator in condition".to_string()),
        })
    }

    fn rhs(&self, operand: &Operand) -> Result<Rhs, String> {
        match operand {
            Operand::Integer(k) => Ok(Rhs::Const(*k as i64)),
            Operand::Variable(s) => Ok(Rhs::Var(self.var_id(s)?)),
            _ => Err("unsupported condition operand".to_string()),
        }
    }

    /// Post `Σ coeffs·vars  rel  rhs` (moving a variable rhs to the lhs).
    fn post_linear(
        &mut self,
        mut coeffs: Vec<i64>,
        mut vars: Vec<VarId>,
        rel: Relation,
        rhs: Rhs,
    ) -> Result<(), String> {
        if coeffs.len() != vars.len() {
            return Err("linear: coeffs/vars length mismatch".to_string());
        }
        match rhs {
            Rhs::Const(k) => linear(&mut self.solver, &coeffs, &vars, rel, k),
            Rhs::Var(y) => {
                coeffs.push(-1);
                vars.push(y);
                linear(&mut self.solver, &coeffs, &vars, rel, 0);
            }
        }
        Ok(())
    }

    /// Convert a parser expression tree to a solver [`Expr`].
    fn tree(&self, t: &ExpressionTree) -> Result<Expr, String> {
        let root = t.first_order_iter().next().ok_or("empty expression tree")?;
        self.node(root)
    }

    fn node(&self, node: &TreeNode) -> Result<Expr, String> {
        match node {
            TreeNode::Constant(i) => Ok(expr::int(*i as i64)),
            TreeNode::Variable(name) => match self.expand_array_ref(name) {
                // A whole-array reference in an expression list means the sum of
                // its cells (the list's coefficient distributes over them).
                Some(cells) if !cells.is_empty() => {
                    Ok(expr::add(cells.into_iter().map(expr::var).collect()))
                }
                Some(_) => Err(format!("empty array reference `{name}`")),
                None => Ok(expr::var(self.var_id(name)?)),
            },
            TreeNode::Operator(op, kids) => {
                let mut args = Vec::new();
                for k in kids {
                    if matches!(k, TreeNode::LeftBracket | TreeNode::RightBracket) {
                        continue;
                    }
                    args.push(self.node(k)?);
                }
                build_op(op, args)
            }
            _ => Err("unsupported expression node".to_string()),
        }
    }

    /// Aux variable equal to `e`, linked by an intension constraint.
    fn aux_for(&mut self, e: Expr) -> VarId {
        let (lo, hi) = e.bounds(&|v| {
            (
                self.solver.store.min(v) as i64,
                self.solver.store.max(v) as i64,
            )
        });
        let aux = self.solver.new_var_range(clamp(lo), clamp(hi));
        crate::constraints::intension::intension(&mut self.solver, expr::eq(expr::var(aux), e));
        aux
    }

    /// Turn a list of parser expression trees into solver variables (aux vars
    /// for non-trivial expressions).
    fn tree_vars(&mut self, list: &[ExpressionTree]) -> Result<Vec<VarId>, String> {
        let mut out = Vec::with_capacity(list.len());
        for t in list {
            let e = self.tree(t)?;
            match e {
                Expr::Var(v) => out.push(v),
                other => out.push(self.aux_for(other)),
            }
        }
        Ok(out)
    }
}

fn clamp(x: i64) -> i32 {
    x.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

fn one(mut a: Vec<Expr>) -> Result<Expr, String> {
    if a.len() != 1 {
        return Err("operator arity".to_string());
    }
    Ok(a.pop().unwrap())
}
fn two(mut a: Vec<Expr>) -> Result<(Expr, Expr), String> {
    if a.len() != 2 {
        return Err("operator arity".to_string());
    }
    let b = a.pop().unwrap();
    let x = a.pop().unwrap();
    Ok((x, b))
}

fn build_op(op: &EOp, args: Vec<Expr>) -> Result<Expr, String> {
    use EOp::*;
    Ok(match op {
        Add => expr::add(args),
        Mul => expr::mul(args),
        Min => expr::min_of(args),
        Max => expr::max_of(args),
        And => expr::and(args),
        Or => expr::or(args),
        Neg => expr::neg(one(args)?),
        Abs => expr::abs(one(args)?),
        Not => expr::not(one(args)?),
        Sqr => {
            let a = one(args)?;
            expr::mul(vec![a.clone(), a])
        }
        Sub => {
            let (a, b) = two(args)?;
            expr::sub(a, b)
        }
        Div => {
            let (a, b) = two(args)?;
            expr::div(a, b)
        }
        Mod => {
            let (a, b) = two(args)?;
            expr::rem(a, b)
        }
        Dist => {
            let (a, b) = two(args)?;
            expr::abs(expr::sub(a, b))
        }
        Lt => {
            let (a, b) = two(args)?;
            expr::lt(a, b)
        }
        Le => {
            let (a, b) = two(args)?;
            expr::le(a, b)
        }
        Ge => {
            let (a, b) = two(args)?;
            expr::ge(a, b)
        }
        Gt => {
            let (a, b) = two(args)?;
            expr::gt(a, b)
        }
        Ne => {
            let (a, b) = two(args)?;
            expr::ne(a, b)
        }
        Eq => {
            let (a, b) = two(args)?;
            expr::eq(a, b)
        }
        Imp => {
            let (a, b) = two(args)?;
            expr::imp(a, b)
        }
        Iff => {
            let (a, b) = two(args)?;
            expr::iff(a, b)
        }
        Xor => {
            let (a, b) = two(args)?;
            expr::not(expr::iff(a, b))
        }
        If => {
            if args.len() != 3 {
                return Err("if arity".to_string());
            }
            let mut it = args.into_iter();
            expr::ite(it.next().unwrap(), it.next().unwrap(), it.next().unwrap())
        }
        Pow | Set | In => return Err("unsupported operator (pow/set/in)".to_string()),
    })
}

/// Run a fallible mapping, recording the first error.
macro_rules! guard {
    ($self:ident, $body:block) => {{
        if $self.error.is_some() {
            return;
        }
        #[allow(clippy::redundant_closure_call)]
        let r: Result<(), String> = (|| $body)();
        if let Err(e) = r {
            $self.fail(e);
        }
    }};
}

impl XcspCallback for Model {
    fn begin_instance(&mut self, _t: &InstanceType) {}

    // --- variables ---
    fn on_variable_interval(&mut self, id: String, min: i32, max: i32) {
        let v = self.solver.new_var_range(min, max);
        self.order.push(v);
        self.decls.push((id.clone(), v));
        self.ids.insert(id, v);
    }
    fn on_variable_values(&mut self, id: String, values: &[i32]) {
        let v = self.solver.new_var_set(values);
        self.order.push(v);
        self.decls.push((id.clone(), v));
        self.ids.insert(id, v);
    }

    // --- constraints ---
    fn on_constraint_extension(
        &mut self,
        list: &[String],
        tuples: &Vec<Vec<i32>>,
        is_support: bool,
        _has_star: bool,
    ) {
        guard!(self, {
            let vars = self.scope(list)?;
            let tuples: Vec<Vec<i32>> = tuples
                .iter()
                .map(|t| {
                    t.iter()
                        .map(|&v| if v == i32::MAX { STAR } else { v })
                        .collect()
                })
                .collect();
            extension(&mut self.solver, &vars, &tuples, is_support);
            Ok(())
        });
    }

    fn on_constraint_unary(&mut self, list: &String, values: &[i32], is_support: bool) {
        guard!(self, {
            let v = self.var_id(list)?;
            let tuples: Vec<Vec<i32>> = values.iter().map(|&x| vec![x]).collect();
            extension(&mut self.solver, &[v], &tuples, is_support);
            Ok(())
        });
    }

    fn on_constraint_intention(&mut self, _list: &[String], tree: &ExpressionTree) {
        guard!(self, {
            let e = self.tree(tree)?;
            crate::constraints::intension::intension(&mut self.solver, e);
            Ok(())
        });
    }

    fn on_constraint_all_different_v1(&mut self, list: &[String]) {
        guard!(self, {
            let vars = self.scope(list)?;
            all_different(&mut self.solver, &vars);
            Ok(())
        });
    }

    fn on_constraint_all_different_list(&mut self, lists: &[Vec<String>]) {
        guard!(self, {
            if lists.len() == 1 {
                let vars = self.scope(&lists[0])?;
                all_different(&mut self.solver, &vars);
            } else {
                // allDifferent on lists: the tuples must be pairwise distinct,
                // i.e. each pair differs in at least one position.
                let tuples = lists
                    .iter()
                    .map(|l| self.scope(l))
                    .collect::<Result<Vec<_>, _>>()?;
                for i in 0..tuples.len() {
                    for j in (i + 1)..tuples.len() {
                        if tuples[i].len() != tuples[j].len() {
                            return Err("allDifferent-list: ragged tuples".to_string());
                        }
                        let diffs = tuples[i]
                            .iter()
                            .zip(&tuples[j])
                            .map(|(&x, &y)| expr::ne(expr::var(x), expr::var(y)))
                            .collect::<Vec<_>>();
                        crate::constraints::intension::intension(&mut self.solver, expr::or(diffs));
                    }
                }
            }
            Ok(())
        });
    }

    fn on_constraint_all_different_except(&mut self, list: &[String], except: &[i32]) {
        guard!(self, {
            // Weak form: for every pair, they differ unless the first takes an
            // exempt value (`x_i == x_j ⟹ x_i ∈ except`).
            // TODO(strong): allDifferent-except via matching that ignores exempts.
            let vars = self.scope(list)?;
            for i in 0..vars.len() {
                for j in (i + 1)..vars.len() {
                    let mut terms = vec![expr::ne(expr::var(vars[i]), expr::var(vars[j]))];
                    for &e in except {
                        terms.push(expr::eq(expr::var(vars[i]), expr::int(e as i64)));
                    }
                    crate::constraints::intension::intension(&mut self.solver, expr::or(terms));
                }
            }
            Ok(())
        });
    }

    fn on_constraint_all_equal_v1(&mut self, list: &[String]) {
        guard!(self, {
            let vars = self.scope(list)?;
            all_equal(&mut self.solver, &vars);
            Ok(())
        });
    }

    fn on_constraint_ordered_v1(&mut self, list: &[String], operator: ROp) {
        guard!(self, {
            let vars = self.scope(list)?;
            ordered(&mut self.solver, &vars, Model::rel(operator)?);
            Ok(())
        });
    }

    fn on_constraint_instantiation(&mut self, list: &[String], values: &[i32]) {
        guard!(self, {
            let vars = self.scope(list)?;
            instantiation(&mut self.solver, &vars, values);
            Ok(())
        });
    }

    fn on_constraint_sum_v1(&mut self, list: &[String], operator: ROp, operand: Operand) {
        guard!(self, {
            let vars = self.scope(list)?;
            let coeffs = vec![1i64; vars.len()];
            let (rel, rhs) = (Model::rel(operator)?, self.rhs(&operand)?);
            self.post_linear(coeffs, vars, rel, rhs)?;
            Ok(())
        });
    }
    fn on_constraint_sum_v2(
        &mut self,
        list: &[String],
        coeffs: &[i32],
        operator: ROp,
        operand: Operand,
    ) {
        guard!(self, {
            let vars = self.scope(list)?;
            let coeffs: Vec<i64> = coeffs.iter().map(|&c| c as i64).collect();
            let (rel, rhs) = (Model::rel(operator)?, self.rhs(&operand)?);
            self.post_linear(coeffs, vars, rel, rhs)?;
            Ok(())
        });
    }
    fn on_constraint_sum_v4(&mut self, list: &[ExpressionTree], operator: ROp, operand: Operand) {
        guard!(self, {
            let vars = self.tree_vars(list)?;
            let coeffs = vec![1i64; vars.len()];
            let (rel, rhs) = (Model::rel(operator)?, self.rhs(&operand)?);
            self.post_linear(coeffs, vars, rel, rhs)?;
            Ok(())
        });
    }
    fn on_constraint_sum_v5(
        &mut self,
        list: &[ExpressionTree],
        coeffs: &[i32],
        operator: ROp,
        operand: Operand,
    ) {
        guard!(self, {
            let vars = self.tree_vars(list)?;
            let coeffs: Vec<i64> = coeffs.iter().map(|&c| c as i64).collect();
            let (rel, rhs) = (Model::rel(operator)?, self.rhs(&operand)?);
            self.post_linear(coeffs, vars, rel, rhs)?;
            Ok(())
        });
    }

    fn on_constraint_count_v2(
        &mut self,
        list: &[String],
        values: &[i32],
        operator: ROp,
        operand: Operand,
    ) {
        guard!(self, {
            let vars = self.scope(list)?;
            // `(in, lo..hi)` condition: occurrences within an interval.
            if matches!(operator, ROp::In) {
                if let Operand::Interval(lo, hi) = operand {
                    self.post_count(&vars, values, Relation::Ge, Rhs::Const(lo as i64));
                    self.post_count(&vars, values, Relation::Le, Rhs::Const(hi as i64));
                    return Ok(());
                }
                return Err("count: in <set> condition".to_string());
            }
            let rel = Model::rel(operator)?;
            let rhs = self.rhs(&operand)?;
            self.post_count(&vars, values, rel, rhs);
            Ok(())
        });
    }

    fn on_constraint_minimum_v1(&mut self, list: &[String], operator: ROp, operand: Operand) {
        guard!(self, { self.min_max(list, operator, operand, true) });
    }
    fn on_constraint_maximum_v1(&mut self, list: &[String], operator: ROp, operand: Operand) {
        guard!(self, { self.min_max(list, operator, operand, false) });
    }

    fn on_constraint_element_v1(&mut self, list: &[String], value: i32) {
        guard!(self, {
            // value belongs to the list (existential element with free index).
            let array = self.scope(list)?;
            let n = array.len() as i32;
            let idx = self.solver.new_var_range(0, n - 1);
            let val = self.constant(value);
            element(&mut self.solver, &array, idx, val);
            Ok(())
        });
    }
    fn on_constraint_element_v4(
        &mut self,
        list: &[String],
        start_index: i32,
        index: String,
        value: i32,
    ) {
        guard!(self, {
            if start_index != 0 {
                return Err("element with non-zero startIndex".to_string());
            }
            let array = self.scope(list)?;
            let idx = self.var_id(&index)?;
            let val = self.constant(value);
            element(&mut self.solver, &array, idx, val);
            Ok(())
        });
    }
    fn on_constraint_element_v3(
        &mut self,
        list: &[String],
        start_index: i32,
        index: String,
        value: String,
    ) {
        guard!(self, {
            if start_index != 0 {
                return Err("element with non-zero startIndex".to_string());
            }
            let array = self.scope(list)?;
            let idx = self.var_id(&index)?;
            let val = self.var_id(&value)?;
            element(&mut self.solver, &array, idx, val);
            Ok(())
        });
    }
    fn on_constraint_element_v5(
        &mut self,
        list: &[String],
        start_index: i32,
        index: String,
        operator: ROp,
        operand: Operand,
    ) {
        guard!(self, {
            if start_index != 0 {
                return Err("element with non-zero startIndex".to_string());
            }
            let array = self.scope(list)?;
            let idx = self.var_id(&index)?;
            self.element_cond(array, idx, operator, operand)
        });
    }
    fn on_constraint_element_v8(
        &mut self,
        list: &[i32],
        start_index: i32,
        index: String,
        operator: ROp,
        operand: Operand,
    ) {
        guard!(self, {
            if start_index != 0 {
                return Err("element with non-zero startIndex".to_string());
            }
            let array: Vec<VarId> = list.iter().map(|&c| self.constant(c)).collect();
            let idx = self.var_id(&index)?;
            self.element_cond(array, idx, operator, operand)
        });
    }
    fn on_constraint_element_v6(
        &mut self,
        list: &[i32],
        start_index: i32,
        index: String,
        value: String,
    ) {
        guard!(self, {
            if start_index != 0 {
                return Err("element with non-zero startIndex".to_string());
            }
            let array: Vec<VarId> = list.iter().map(|&c| self.constant(c)).collect();
            let idx = self.var_id(&index)?;
            let val = self.var_id(&value)?;
            element(&mut self.solver, &array, idx, val);
            Ok(())
        });
    }
    fn on_constraint_element_v7(
        &mut self,
        list: &[i32],
        start_index: i32,
        index: String,
        value: i32,
    ) {
        guard!(self, {
            if start_index != 0 {
                return Err("element with non-zero startIndex".to_string());
            }
            let array: Vec<VarId> = list.iter().map(|&c| self.constant(c)).collect();
            let idx = self.var_id(&index)?;
            let val = self.constant(value);
            element(&mut self.solver, &array, idx, val);
            Ok(())
        });
    }

    fn on_constraint_cumulative_v1(
        &mut self,
        origins: &[String],
        lengths: &[i32],
        heights: &[i32],
        operator: ROp,
        operand: Operand,
    ) {
        guard!(self, {
            if !matches!(operator, ROp::Le) {
                return Err("cumulative condition must be <=".to_string());
            }
            let starts = self.scope(origins)?;
            let d: Vec<i64> = lengths.iter().map(|&x| x as i64).collect();
            match self.rhs(&operand)? {
                // Fixed heights + constant capacity: the strong edge-finder.
                Rhs::Const(k) => {
                    let h: Vec<i64> = heights.iter().map(|&x| x as i64).collect();
                    cumulative(&mut self.solver, &starts, &d, &h, k);
                }
                // Variable capacity: the weaker variable-resource propagator.
                Rhs::Var(cap) => {
                    let h: Vec<VarId> = heights.iter().map(|&x| self.constant(x)).collect();
                    cumulative_var(&mut self.solver, &starts, &d, &h, cap);
                }
            }
            Ok(())
        });
    }
    fn on_constraint_cumulative_v2(
        &mut self,
        origins: &[String],
        lengths: &[i32],
        heights: &[String],
        operator: ROp,
        operand: Operand,
    ) {
        guard!(self, {
            if !matches!(operator, ROp::Le) {
                return Err("cumulative condition must be <=".to_string());
            }
            let starts = self.scope(origins)?;
            let d: Vec<i64> = lengths.iter().map(|&x| x as i64).collect();
            let h = self.scope(heights)?;
            let cap = match self.rhs(&operand)? {
                Rhs::Const(k) => self.constant(clamp(k)),
                Rhs::Var(v) => v,
            };
            cumulative_var(&mut self.solver, &starts, &d, &h, cap);
            Ok(())
        });
    }

    fn on_constraint_nvalues_v1(&mut self, list: &[String], operator: ROp, operand: Operand) {
        guard!(self, {
            let vars = self.scope(list)?;
            let rel = Model::rel(operator)?;
            match self.rhs(&operand)? {
                Rhs::Const(k) => n_values(&mut self.solver, &vars, rel, k),
                Rhs::Var(_) => return Err("nValues with variable rhs".to_string()),
            }
            Ok(())
        });
    }

    fn on_constraint_channel_v1(&mut self, list: &[String], _start_index: i32) {
        guard!(self, {
            let vars = self.scope(list)?;
            channel(&mut self.solver, &vars, &vars);
            Ok(())
        });
    }

    fn on_constraint_precedence_v1(&mut self, list: &[String], _covered: bool) {
        guard!(self, {
            // Default value order: the sorted distinct values across all domains.
            let vars = self.scope(list)?;
            let mut vals: Vec<i32> = Vec::new();
            for &v in &vars {
                vals.extend(self.solver.store.values(v));
            }
            vals.sort_unstable();
            vals.dedup();
            crate::constraints::primitives::precedence(&mut self.solver, &vars, &vals);
            Ok(())
        });
    }
    fn on_constraint_precedence_v2(&mut self, list: &[String], values: &[i32], _covered: bool) {
        guard!(self, {
            let vars = self.scope(list)?;
            crate::constraints::primitives::precedence(&mut self.solver, &vars, values);
            Ok(())
        });
    }

    fn on_constraint_circuit_v1(&mut self, list: &Vec<String>) {
        guard!(self, {
            let vars = self.scope(list)?;
            circuit(&mut self.solver, &vars);
            Ok(())
        });
    }

    fn on_constraint_lex(&mut self, lists: &Vec<Vec<String>>, operator: ROp) {
        guard!(self, {
            let mut rows = Vec::new();
            for l in lists {
                rows.push(self.scope(l)?);
            }
            let rel = Model::rel(operator)?;
            let strict = matches!(rel, Relation::Lt | Relation::Gt);
            if matches!(rel, Relation::Gt | Relation::Ge) {
                rows.reverse();
            }
            lex_chain(&mut self.solver, &rows, strict);
            Ok(())
        });
    }

    fn on_constraint_cardinality_v1(
        &mut self,
        list: &[String],
        values: &[i32],
        occurs: &[i32],
        closed: bool,
    ) {
        guard!(self, {
            if occurs.len() != values.len() {
                return Err("cardinality: occurs/values length mismatch".to_string());
            }
            let vars = self.scope(list)?;
            let low: Vec<i64> = occurs.iter().map(|&x| x as i64).collect();
            cardinality(&mut self.solver, &vars, values, &low, &low, closed);
            Ok(())
        });
    }
    fn on_constraint_cardinality_v3(
        &mut self,
        list: &[String],
        values: &[i32],
        occurs: &[(i32, i32)],
        closed: bool,
    ) {
        guard!(self, {
            if occurs.len() != values.len() {
                return Err("cardinality: occurs/values length mismatch".to_string());
            }
            let vars = self.scope(list)?;
            let low: Vec<i64> = occurs.iter().map(|&(l, _)| l as i64).collect();
            let high: Vec<i64> = occurs.iter().map(|&(_, h)| h as i64).collect();
            cardinality(&mut self.solver, &vars, values, &low, &high, closed);
            Ok(())
        });
    }

    fn on_constraint_bin_packing_v1(
        &mut self,
        list: &[String],
        sizes: &[i32],
        operator: ROp,
        operand: Operand,
    ) {
        guard!(self, {
            let items = self.scope(list)?;
            let s: Vec<i64> = sizes.iter().map(|&x| x as i64).collect();
            let cap = match (operator, self.rhs(&operand)?) {
                (ROp::Le, Rhs::Const(k)) => k,
                _ => return Err("binPacking needs <= constant capacity".to_string()),
            };
            // count bins from the item domains
            let nbins = (0..items.len())
                .map(|i| self.solver.store.max(items[i]) + 1)
                .max()
                .unwrap_or(0)
                .max(1) as usize;
            bin_packing(&mut self.solver, &items, &s, &vec![cap; nbins]);
            Ok(())
        });
    }
    fn on_constraint_bin_packing_v2(&mut self, list: &[String], sizes: &[i32], limits: &[i32]) {
        guard!(self, {
            let items = self.scope(list)?;
            let s: Vec<i64> = sizes.iter().map(|&x| x as i64).collect();
            let caps: Vec<i64> = limits.iter().map(|&x| x as i64).collect();
            bin_packing(&mut self.solver, &items, &s, &caps);
            Ok(())
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn on_constraint_knapsack(
        &mut self,
        list: &[String],
        weights: &[i32],
        _w_op: ROp,
        w_operand: Operand,
        profits: &[i32],
        _p_op: ROp,
        p_operand: Operand,
    ) {
        guard!(self, {
            let vars = self.scope(list)?;
            let w: Vec<i64> = weights.iter().map(|&x| x as i64).collect();
            let p: Vec<i64> = profits.iter().map(|&x| x as i64).collect();
            // Σ w·x ≤ W and Σ p·x ≥ P (W, P may be variables).
            self.post_linear(w, vars.clone(), Relation::Le, self.rhs(&w_operand)?)?;
            self.post_linear(p, vars, Relation::Ge, self.rhs(&p_operand)?)?;
            Ok(())
        });
    }

    fn on_constraint_no_overlap_v1(
        &mut self,
        list: &[String],
        lengths: &[i32],
        _zero_ignored: bool,
    ) {
        guard!(self, {
            let starts = self.scope(list)?;
            let d: Vec<i64> = lengths.iter().map(|&x| x as i64).collect();
            no_overlap(&mut self.solver, &starts, &d);
            Ok(())
        });
    }

    fn on_constraint_regular(
        &mut self,
        list: &[String],
        start: String,
        finals: &[String],
        transitions: &[(String, i32, String)],
    ) {
        guard!(self, {
            let vars = self.scope(list)?;
            let mut states: HashMap<String, usize> = HashMap::new();
            let state_id = |s: &str, m: &mut HashMap<String, usize>| {
                let n = m.len();
                *m.entry(s.to_string()).or_insert(n)
            };
            let trans: Vec<(usize, i32, usize)> = transitions
                .iter()
                .map(|(a, v, b)| (state_id(a, &mut states), *v, state_id(b, &mut states)))
                .collect();
            let start_s = state_id(&start, &mut states);
            let accept: Vec<usize> = finals.iter().map(|s| state_id(s, &mut states)).collect();
            let dfa = Dfa {
                n_states: states.len(),
                start: start_s,
                accept,
                transitions: trans,
            };
            regular(&mut self.solver, &vars, dfa);
            Ok(())
        });
    }

    fn on_constraint_mdd(&mut self, list: &[String], transitions: &Vec<(String, i32, String)>) {
        guard!(self, {
            let vars = self.scope(list)?;
            // Layer per variable, inferred from a root reachable BFS over labels.
            let m = build_mdd(vars.len(), transitions)?;
            mdd(&mut self.solver, &vars, m);
            Ok(())
        });
    }

    // --- objectives ---
    fn on_minimize_var(&mut self, var: String) {
        guard!(self, {
            self.objective = Some((true, self.var_id(&var)?));
            Ok(())
        });
    }
    fn on_maximize_var(&mut self, var: String) {
        guard!(self, {
            self.objective = Some((false, self.var_id(&var)?));
            Ok(())
        });
    }
    fn on_minimize_expression(&mut self, e: &ExpressionTree) {
        guard!(self, {
            let expr = self.tree(e)?;
            let aux = self.aux_for(expr);
            self.order.push(aux);
            self.objective = Some((true, aux));
            Ok(())
        });
    }
    fn on_maximize_expression(&mut self, e: &ExpressionTree) {
        guard!(self, {
            let expr = self.tree(e)?;
            let aux = self.aux_for(expr);
            self.order.push(aux);
            self.objective = Some((false, aux));
            Ok(())
        });
    }
    fn on_minimize_v1(&mut self, t: XElementOperator, list: &[String], coefs: &[i32]) {
        guard!(self, { self.objective_sum(t, list, coefs, true) });
    }
    fn on_maximize_v1(&mut self, t: XElementOperator, list: &[String], coefs: &[i32]) {
        guard!(self, { self.objective_sum(t, list, coefs, false) });
    }
    fn on_minimize_v5(&mut self, t: XElementOperator, list: &[String]) {
        let ones = vec![1i32; list.len()];
        guard!(self, { self.objective_sum(t, list, &ones, true) });
    }
    fn on_maximize_v5(&mut self, t: XElementOperator, list: &[String]) {
        let ones = vec![1i32; list.len()];
        guard!(self, { self.objective_sum(t, list, &ones, false) });
    }
    fn on_minimize_v3(&mut self, t: XElementOperator, list: &[ExpressionTree], coefs: &[i32]) {
        guard!(self, { self.objective_exprs(t, list, coefs, true) });
    }
    fn on_maximize_v3(&mut self, t: XElementOperator, list: &[ExpressionTree], coefs: &[i32]) {
        guard!(self, { self.objective_exprs(t, list, coefs, false) });
    }
    fn on_minimize_v6(&mut self, t: XElementOperator, list: &[ExpressionTree]) {
        let ones = vec![1i32; list.len()];
        guard!(self, { self.objective_exprs(t, list, &ones, true) });
    }
    fn on_maximize_v6(&mut self, t: XElementOperator, list: &[ExpressionTree]) {
        let ones = vec![1i32; list.len()];
        guard!(self, { self.objective_exprs(t, list, &ones, false) });
    }
}

impl Model {
    fn min_max(
        &mut self,
        list: &[String],
        operator: ROp,
        operand: Operand,
        is_min: bool,
    ) -> Result<(), String> {
        let xs = self.scope(list)?;
        let rel = Model::rel(operator)?;
        let y = match self.rhs(&operand)? {
            Rhs::Const(k) => self.constant(k as i32),
            Rhs::Var(v) => v,
        };
        if matches!(rel, Relation::Eq) {
            if is_min {
                minimum(&mut self.solver, y, &xs);
            } else {
                maximum(&mut self.solver, y, &xs);
            }
        } else {
            // aux = min/max(xs); then aux rel y.
            let (lo, hi) = (
                xs.iter()
                    .map(|&v| self.solver.store.min(v))
                    .min()
                    .unwrap_or(0),
                xs.iter()
                    .map(|&v| self.solver.store.max(v))
                    .max()
                    .unwrap_or(0),
            );
            let aux = self.solver.new_var_range(lo, hi);
            if is_min {
                minimum(&mut self.solver, aux, &xs);
            } else {
                maximum(&mut self.solver, aux, &xs);
            }
            linear(&mut self.solver, &[1, -1], &[aux, y], rel, 0);
        }
        Ok(())
    }

    /// Post a count condition `#{i : vars[i] ∈ values}  rel  rhs`, choosing the
    /// dedicated bounds propagator for the single-value/constant case.
    fn post_count(&mut self, vars: &[VarId], values: &[i32], rel: Relation, rhs: Rhs) {
        match rhs {
            Rhs::Const(k) if values.len() == 1 => count(&mut self.solver, vars, values[0], rel, k),
            Rhs::Const(k) => {
                let y = self.constant(clamp(k));
                self.count_compare(vars, values, rel, y);
            }
            Rhs::Var(y) => self.count_compare(vars, values, rel, y),
        }
    }

    /// `aux = array[index]` then `aux  rel  operand`.
    fn element_cond(
        &mut self,
        array: Vec<VarId>,
        idx: VarId,
        operator: ROp,
        operand: Operand,
    ) -> Result<(), String> {
        let lo = array
            .iter()
            .map(|&v| self.solver.store.min(v))
            .min()
            .unwrap_or(0);
        let hi = array
            .iter()
            .map(|&v| self.solver.store.max(v))
            .max()
            .unwrap_or(0);
        let aux = self.solver.new_var_range(lo, hi);
        element(&mut self.solver, &array, idx, aux);
        let rel = Model::rel(operator)?;
        let y = match self.rhs(&operand)? {
            Rhs::Const(k) => self.constant(clamp(k)),
            Rhs::Var(v) => v,
        };
        linear(&mut self.solver, &[1, -1], &[aux, y], rel, 0);
        Ok(())
    }

    /// Post `#{i : vars[i] ∈ values}  rel  y` via reified 0/1 indicators summed
    /// linearly. Correct for value sets and a variable bound `y`.
    fn count_compare(&mut self, vars: &[VarId], values: &[i32], rel: Relation, y: VarId) {
        let mut terms: Vec<VarId> = Vec::with_capacity(vars.len() + 1);
        for &v in vars {
            let hit = if values.len() == 1 {
                expr::eq(expr::var(v), expr::int(values[0] as i64))
            } else {
                expr::or(
                    values
                        .iter()
                        .map(|&val| expr::eq(expr::var(v), expr::int(val as i64)))
                        .collect(),
                )
            };
            terms.push(self.aux_for(hit));
        }
        let mut coeffs = vec![1i64; terms.len()];
        coeffs.push(-1);
        terms.push(y);
        linear(&mut self.solver, &coeffs, &terms, rel, 0);
    }

    /// Objective over named variables.
    fn objective_sum(
        &mut self,
        t: XElementOperator,
        list: &[String],
        coefs: &[i32],
        minimize: bool,
    ) -> Result<(), String> {
        let vars = self.scope(list)?;
        let coeffs: Vec<i64> = coefs.iter().map(|&c| c as i64).collect();
        self.objective_agg(t, vars, coeffs, minimize)
    }

    /// Objective over expression trees (each becomes an aux variable).
    fn objective_exprs(
        &mut self,
        t: XElementOperator,
        list: &[ExpressionTree],
        coefs: &[i32],
        minimize: bool,
    ) -> Result<(), String> {
        let vars = self.tree_vars(list)?;
        let coeffs: Vec<i64> = coefs.iter().map(|&c| c as i64).collect();
        self.objective_agg(t, vars, coeffs, minimize)
    }

    /// Build the objective variable as `aggregate_t(coeffs·vars)` and record it.
    fn objective_agg(
        &mut self,
        t: XElementOperator,
        vars: Vec<VarId>,
        coeffs: Vec<i64>,
        minimize: bool,
    ) -> Result<(), String> {
        use XElementOperator::*;
        if coeffs.len() != vars.len() {
            return Err("objective: coeffs/terms length mismatch".to_string());
        }
        let obj = match t {
            Sum => {
                let (mut lo, mut hi) = (0i64, 0i64);
                for (&c, &v) in coeffs.iter().zip(&vars) {
                    let (vmin, vmax) = (
                        self.solver.store.min(v) as i64,
                        self.solver.store.max(v) as i64,
                    );
                    let (a, b) = if c >= 0 {
                        (c * vmin, c * vmax)
                    } else {
                        (c * vmax, c * vmin)
                    };
                    lo += a;
                    hi += b;
                }
                let obj = self.solver.new_var_range(clamp(lo), clamp(hi));
                let mut cc = coeffs;
                cc.push(-1);
                let mut vv = vars;
                vv.push(obj);
                linear(&mut self.solver, &cc, &vv, Relation::Eq, 0);
                obj
            }
            Minimum | Maximum => {
                // Fold coefficients into aux terms when they aren't all 1.
                let terms: Vec<VarId> = if coeffs.iter().all(|&c| c == 1) {
                    vars
                } else {
                    vars.iter()
                        .zip(&coeffs)
                        .map(|(&v, &c)| self.aux_for(expr::mul(vec![expr::int(c), expr::var(v)])))
                        .collect()
                };
                let lo = terms
                    .iter()
                    .map(|&v| self.solver.store.min(v))
                    .min()
                    .unwrap_or(0);
                let hi = terms
                    .iter()
                    .map(|&v| self.solver.store.max(v))
                    .max()
                    .unwrap_or(0);
                let obj = self.solver.new_var_range(lo, hi);
                if matches!(t, Minimum) {
                    minimum(&mut self.solver, obj, &terms);
                } else {
                    maximum(&mut self.solver, obj, &terms);
                }
                obj
            }
            _ => return Err("unsupported objective type (product/nValues/lex)".to_string()),
        };
        self.order.push(obj);
        self.objective = Some((minimize, obj));
        Ok(())
    }
}

/// Build a layered MDD from `(src, value, dst)` transitions over `n` layers.
/// Node names are layered by BFS distance from the root.
fn build_mdd(n: usize, transitions: &[(String, i32, String)]) -> Result<Mdd, String> {
    use std::collections::{HashMap, HashSet};
    // adjacency
    let mut out: HashMap<&str, Vec<(i32, &str)>> = HashMap::new();
    let mut all: HashSet<&str> = HashSet::new();
    let mut has_in: HashSet<&str> = HashSet::new();
    for (a, v, b) in transitions {
        out.entry(a).or_default().push((*v, b));
        all.insert(a);
        all.insert(b);
        has_in.insert(b);
    }
    let root = *all
        .iter()
        .find(|s| !has_in.contains(*s))
        .ok_or("mdd: no root")?;
    // layer of each node by BFS; node id within layer
    let mut layer_of: HashMap<&str, usize> = HashMap::new();
    layer_of.insert(root, 0);
    let mut frontier = vec![root];
    for l in 0..n {
        let mut next = Vec::new();
        for node in &frontier {
            if let Some(edges) = out.get(node) {
                for (_, b) in edges {
                    if !layer_of.contains_key(b) {
                        layer_of.insert(b, l + 1);
                        next.push(*b);
                    }
                }
            }
        }
        frontier = next;
    }
    let mut nodes_per_layer = vec![0usize; n + 1];
    let mut local: HashMap<&str, usize> = HashMap::new();
    let mut by_layer: Vec<Vec<&str>> = vec![Vec::new(); n + 1];
    let mut order: Vec<(&str, usize)> = layer_of.iter().map(|(&k, &v)| (k, v)).collect();
    order.sort_by_key(|&(_, l)| l);
    for (node, l) in order {
        if l > n {
            continue;
        }
        local.insert(node, by_layer[l].len());
        by_layer[l].push(node);
        nodes_per_layer[l] += 1;
    }
    let mut layers: Vec<Vec<MddArc>> = (0..n).map(|_| Vec::new()).collect();
    for (a, v, b) in transitions {
        let la = *layer_of.get(a.as_str()).ok_or("mdd: bad node")?;
        if la >= n {
            continue;
        }
        let from = *local.get(a.as_str()).ok_or("mdd: bad node")?;
        let to = *local.get(b.as_str()).ok_or("mdd: bad node")?;
        layers[la].push(MddArc {
            from,
            value: *v,
            to,
        });
    }
    Ok(Mdd {
        layers,
        nodes_per_layer,
    })
}
