//! Bridge from the `xcsp3-rust-parser` callback interface to the solver.
//!
//! Each callback maps onto a posting helper. Unsupported forms set `error`.

use std::collections::HashMap;
use std::sync::Arc;

use xcsp3_rust_parser::data_structs::expression_tree::xcsp3_utils::{ExpressionTree, Operator as EOp, TreeNode};
use xcsp3_rust_parser::data_structs::xrelational_operand::xcsp3_core::Operand;
use xcsp3_rust_parser::data_structs::xrelational_operator::xcsp3_core::Operator as ROp;
use xcsp3_rust_parser::objectives::xobjective_element::xcsp3_core::XElementOperator;
use xcsp3_rust_parser::xcsp_callback::XcspCallback;
use xcsp3_rust_parser::xcsp_xml::xcsp_xml_model::xcsp3_xml::InstanceType;

use crate::constraints::count::{cardinality, count, n_values};
use crate::constraints::flatten;
use crate::constraints::graph::circuit;
use crate::constraints::lex::{channel, lex_chain};
use crate::constraints::linear::{linear, Relation};
use crate::constraints::primitives::{all_different, all_equal, element, instantiation, maximum, minimum, ordered, sign_products};
use crate::constraints::scheduling::{bin_packing, cumulative, cumulative_var, no_overlap};
use crate::constraints::table::{
    extension_from_template, extension_template as compile_extension_template, mdd, regular, Dfa, ExtensionTemplate, Mdd, MddArc, STAR,
};
use crate::engines::ls::cop::{LocalRhs, LocalSearchSpec};
use crate::expr::{self, Expr};
use crate::ids::VarId;
use crate::problem::Objective;
use crate::store::Solver;

const MAX_MATERIALIZED_OBJECTIVE_SPAN: i64 = 1_000_000;

fn require(ok: bool, message: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(message.to_string())
    }
}

fn i64s(values: &[i32]) -> Vec<i64> {
    values.iter().map(|&x| x as i64).collect()
}

fn ones(n: usize) -> Vec<i64> {
    vec![1; n]
}

fn weighted_sum_expr(coeffs: &[i64], vars: &[VarId]) -> Expr {
    let mut terms = Vec::with_capacity(vars.len());
    for (&coeff, &var) in coeffs.iter().zip(vars) {
        match coeff {
            0 => {}
            1 => terms.push(expr::var(var)),
            -1 => terms.push(expr::neg(expr::var(var))),
            _ => terms.push(expr::mul(vec![expr::int(coeff), expr::var(var)])),
        }
    }
    match terms.len() {
        0 => expr::int(0),
        1 => terms.pop().unwrap(),
        _ => expr::add(terms),
    }
}

fn sorted_values(values: impl IntoIterator<Item = i32>) -> Result<Vec<i32>, String> {
    let mut values: Vec<i32> = values.into_iter().collect();
    values.sort_unstable();
    values.dedup();
    require(!values.is_empty(), "empty set condition")?;
    Ok(values)
}

struct ArrayDecl {
    shape: Vec<usize>,
    /// Sparse: index-tuple → variable. XCSP arrays may declare cells out of
    /// row-major order and only for a subset (e.g. per-cell `<domain for=...>`),
    /// so a dense `Vec` keyed by a flattened index does not work in general.
    cells: HashMap<Vec<usize>, VarId>,
}

/// Accumulates the model as the parser walks the instance.
pub struct Model {
    pub solver: Solver,
    pub declared: Vec<(String, VarId)>,
    pub(super) objective: Option<Objective>,
    pub(super) local: LocalSearchSpec,
    pub error: Option<String>,
    /// Scalar variables by name. Array cells use compact row-major storage.
    ids: HashMap<String, VarId>,
    arrays: HashMap<String, ArrayDecl>,
    pending_sign_products: Vec<[VarId; 3]>,
    share_extension_template: bool,
    extension_template: Option<Arc<ExtensionTemplate>>,
}

/// Right-hand side of a condition.
enum Rhs {
    Const(i64),
    Var(VarId),
}

#[derive(Clone, Copy)]
struct MatrixAccess<'a> {
    rows: usize,
    cols: usize,
    row_index: &'a str,
    col_index: &'a str,
    start_row_index: i32,
    start_col_index: i32,
}

impl Model {
    pub fn new() -> Self {
        Self {
            solver: Solver::new(),
            declared: Vec::new(),
            objective: None,
            local: LocalSearchSpec::default(),
            error: None,
            ids: HashMap::new(),
            arrays: HashMap::new(),
            pending_sign_products: Vec::new(),
            share_extension_template: false,
            extension_template: None,
        }
    }

    fn extension_template(&mut self, arity: usize, tuples: impl FnOnce() -> Vec<Vec<i32>>) -> Arc<ExtensionTemplate> {
        if let Some(template) = &self.extension_template {
            return Arc::clone(template);
        }
        let tuples = tuples();
        let template = compile_extension_template(arity, &tuples);
        if self.share_extension_template {
            self.extension_template = Some(Arc::clone(&template));
        }
        template
    }

    fn fail(&mut self, msg: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(msg.into());
        }
    }

    fn var_id(&self, name: &str) -> Result<VarId, String> {
        if let Some((base, indices)) = cell_ref(name) {
            let array = self.arrays.get(base).ok_or_else(|| format!("unknown variable `{name}`"))?;
            return array.cells.get(&indices).copied().ok_or_else(|| format!("unknown variable `{name}`"));
        }
        self.ids.get(name).copied().ok_or_else(|| format!("unknown variable `{name}`"))
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
        let array = self.arrays.get(base).filter(|array| array.shape.len() == depth)?;
        let mut tuples = vec![Vec::new()];
        for &size in &array.shape {
            tuples = tuples
                .iter()
                .flat_map(|p| {
                    (0..size).map(move |v| {
                        let mut t = p.clone();
                        t.push(v);
                        t
                    })
                })
                .collect();
        }
        tuples.iter().map(|t| array.cells.get(t).copied()).collect()
    }

    /// Expand a ranged/partial array slice such as `x[0..2]`, `x[1][]`, or
    /// `x[][0..3]` into its cells (row-major). Each dimension is empty (whole),
    /// a single index, or an inclusive `lo..hi` range. Returns `None` for plain
    /// scalar/cell refs (no slice) so those keep flowing through `var_id`.
    fn expand_array_slice(&self, name: &str) -> Option<Vec<VarId>> {
        let open = name.find('[')?;
        let (base, mut rest) = name.split_at(open);
        let array = self.arrays.get(base)?;
        let mut dims: Vec<(usize, usize)> = Vec::new();
        let mut saw_slice = false;
        while let Some(inner) = rest.strip_prefix('[') {
            let close = inner.find(']')?;
            let spec = &inner[..close];
            let size = *array.shape.get(dims.len())?;
            let (lo, hi) = if spec.is_empty() {
                saw_slice = true;
                (0, size.checked_sub(1)?)
            } else if let Some((a, b)) = spec.split_once("..") {
                saw_slice = true;
                (a.parse().ok()?, b.parse().ok()?)
            } else {
                let i = spec.parse().ok()?;
                (i, i)
            };
            if lo > hi || hi >= size {
                return None;
            }
            dims.push((lo, hi));
            rest = &inner[(close + 1)..];
        }
        // Must consume the whole name, cover every dimension, and be a real slice
        // (plain single cells are left to `var_id`).
        if !rest.is_empty() || dims.len() != array.shape.len() || !saw_slice {
            return None;
        }
        let mut tuples = vec![Vec::new()];
        for &(lo, hi) in &dims {
            tuples = tuples
                .iter()
                .flat_map(|prefix| {
                    (lo..=hi).map(move |v| {
                        let mut t = prefix.clone();
                        t.push(v);
                        t
                    })
                })
                .collect();
        }
        tuples.iter().map(|t| array.cells.get(t).copied()).collect()
    }

    fn remember_var(&mut self, id: String, var: VarId) {
        let Some((base, indices)) = cell_ref(&id) else {
            self.ids.insert(id, var);
            return;
        };
        let array = self.arrays.entry(base.to_string()).or_insert(ArrayDecl { shape: vec![0; indices.len()], cells: HashMap::new() });
        debug_assert_eq!(array.shape.len(), indices.len(), "array rank changed");
        for (size, &index) in array.shape.iter_mut().zip(&indices) {
            *size = (*size).max(index + 1);
        }
        array.cells.insert(indices, var);
    }

    /// A fresh fixed variable holding `value`.
    fn constant(&mut self, value: i32) -> VarId {
        self.solver.new_var_set(&[value])
    }

    /// Fresh fixed variables, one per integer (for slots that take variables).
    fn consts(&mut self, values: &[i32]) -> Vec<VarId> {
        values.iter().map(|&x| self.constant(x)).collect()
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

    fn rhs_var(&mut self, operand: &Operand) -> Result<VarId, String> {
        Ok(match self.rhs(operand)? {
            Rhs::Const(k) => self.constant(clamp(k)),
            Rhs::Var(v) => v,
        })
    }

    fn local_rhs(rhs: &Rhs) -> LocalRhs {
        match rhs {
            Rhs::Const(k) => LocalRhs::Const(*k),
            Rhs::Var(v) => LocalRhs::Var(*v),
        }
    }

    fn var_or_constant(&mut self, s: &str) -> Result<VarId, String> {
        match s.parse::<i32>() {
            Ok(v) => Ok(self.constant(v)),
            Err(_) => self.var_id(s),
        }
    }

    /// Decode a condition `(operator, operand)` into the relations to post.
    /// An `(in, lo..hi)` interval becomes two bounds; `(notin, …)` and set
    /// operands are unsupported.
    fn conditions(&self, operator: ROp, operand: Operand) -> Result<Vec<(Relation, Rhs)>, String> {
        match operator {
            ROp::In => match operand {
                Operand::Interval(lo, hi) => Ok(vec![(Relation::Ge, Rhs::Const(lo as i64)), (Relation::Le, Rhs::Const(hi as i64))]),
                _ => Err("in <set> condition".to_string()),
            },
            ROp::Notin => match operand {
                // y ∉ {set}  ⟺  conjunction of disequalities. (Interval `notin`
                // is a disjunction and is handled by the individual posters.)
                Operand::SetInteger(set) => Ok(sorted_values(set)?.into_iter().map(|v| (Relation::Ne, Rhs::Const(v as i64))).collect()),
                _ => Err("notin interval condition (handled by caller)".to_string()),
            },
            _ => Ok(vec![(Model::rel(operator)?, self.rhs(&operand)?)]),
        }
    }

    /// Post `Σ coeffs·vars  (operator)  operand`, expanding interval conditions.
    fn post_sum(&mut self, coeffs: Vec<i64>, vars: Vec<VarId>, operator: ROp, operand: Operand) -> Result<(), String> {
        match (operator, operand) {
            (ROp::In, Operand::SetInteger(set)) => {
                let allowed = sorted_values(set)?;
                let y = self.solver.new_var_set(&allowed);
                self.post_linear(coeffs, vars, Relation::Eq, Rhs::Var(y))?;
            }
            (ROp::Notin, Operand::Interval(lo, hi)) => {
                // Σ ∉ [lo,hi] is a disjunction. Keep it symbolic instead of
                // materialising the complement, which can be empty or huge.
                let sum = weighted_sum_expr(&coeffs, &vars);
                let e = expr::or(vec![expr::lt(sum.clone(), expr::int(lo as i64)), expr::gt(sum, expr::int(hi as i64))]);
                self.local.add_expr(e.clone());
                crate::constraints::intension::intension(&mut self.solver, e);
            }
            (operator, operand) => {
                for (rel, rhs) in self.conditions(operator, operand)? {
                    self.post_linear(coeffs.clone(), vars.clone(), rel, rhs)?;
                }
            }
        }
        Ok(())
    }

    fn post_sum_vars(&mut self, list: &[String], coeffs: Option<&[i32]>, operator: ROp, operand: Operand) -> Result<(), String> {
        let vars = self.scope(list)?;
        let coeffs = coeffs.map(i64s).unwrap_or_else(|| ones(vars.len()));
        self.post_sum(coeffs, vars, operator, operand)
    }

    fn post_sum_exprs(&mut self, list: &[ExpressionTree], coeffs: Option<&[i32]>, operator: ROp, operand: Operand) -> Result<(), String> {
        let vars = self.tree_vars(list)?;
        let coeffs = coeffs.map(i64s).unwrap_or_else(|| ones(vars.len()));
        self.post_sum(coeffs, vars, operator, operand)
    }

    fn post_count_condition(&mut self, vars: &[VarId], values: &[i32], operator: ROp, operand: Operand) -> Result<(), String> {
        match (operator, operand) {
            (ROp::In, Operand::SetInteger(set)) => {
                let allowed = sorted_values(set)?;
                let y = self.solver.new_var_set(&allowed);
                self.local.add_count_allowed(vars.to_vec(), values.to_vec(), allowed);
                self.post_count(vars, values, Relation::Eq, Rhs::Var(y));
            }
            (operator, operand) => {
                for (rel, rhs) in self.conditions(operator, operand)? {
                    self.local.add_count(vars.to_vec(), values.to_vec(), rel, Self::local_rhs(&rhs));
                    self.post_count(vars, values, rel, rhs);
                }
            }
        }
        Ok(())
    }

    fn post_lex_rows(&mut self, lists: &[Vec<String>], operator: ROp) -> Result<(), String> {
        let mut rows = Vec::with_capacity(lists.len());
        for l in lists {
            // A `lex` tuple may mix variables and integer constants.
            let row = l.iter().map(|s| self.var_or_constant(s)).collect::<Result<Vec<_>, _>>()?;
            rows.push(row);
        }
        let rel = Model::rel(operator)?;
        let strict = matches!(rel, Relation::Lt | Relation::Gt);
        if matches!(rel, Relation::Gt | Relation::Ge) {
            rows.reverse();
        }
        self.local.add_lex_chain(rows.clone(), strict);
        lex_chain(&mut self.solver, &rows, strict);
        Ok(())
    }

    /// Post `Σ coeffs·vars  rel  rhs` (moving a variable rhs to the lhs).
    fn post_linear(&mut self, mut coeffs: Vec<i64>, mut vars: Vec<VarId>, rel: Relation, rhs: Rhs) -> Result<(), String> {
        if coeffs.len() != vars.len() {
            return Err("linear: coeffs/vars length mismatch".to_string());
        }
        match rhs {
            Rhs::Const(k) => {
                linear(&mut self.solver, &coeffs, &vars, rel, k);
                self.local.add_linear(coeffs, vars, rel, k);
            }
            Rhs::Var(y) => {
                coeffs.push(-1);
                vars.push(y);
                linear(&mut self.solver, &coeffs, &vars, rel, 0);
                self.local.add_linear(coeffs, vars, rel, 0);
            }
        }
        Ok(())
    }

    fn flush_sign_products(&mut self) {
        sign_products(&mut self.solver, &std::mem::take(&mut self.pending_sign_products));
    }

    fn sign_product(&self, e: &Expr) -> Option<[VarId; 3]> {
        let Expr::Eq(a, b) = e else {
            return None;
        };
        let (Expr::Var(y), Expr::Mul(terms)) = (&**a, &**b) else {
            return None;
        };
        let [Expr::Var(x), Expr::Var(z)] = terms.as_slice() else {
            return None;
        };
        let vars = [*y, *x, *z];
        (y != x
            && y != z
            && x != z
            && vars
                .iter()
                .all(|&v| self.solver.store.size(v) == 2 && self.solver.store.contains(v, -1) && self.solver.store.contains(v, 1)))
        .then_some(vars)
    }

    /// Convert a parser expression tree to a solver [`Expr`].
    fn tree(&self, t: &ExpressionTree) -> Result<Expr, String> {
        let root = t.first_order_iter().next().ok_or("empty expression tree")?;
        self.node(root)
    }

    fn node(&self, node: &TreeNode) -> Result<Expr, String> {
        match node {
            TreeNode::Constant(i) => Ok(expr::int(*i as i64)),
            TreeNode::Variable(name) => match self.expand_array_ref(name).or_else(|| self.expand_array_slice(name)) {
                // A whole-array/slice reference in an expression means the sum of
                // its cells (the surrounding coefficient distributes over them).
                Some(cells) if !cells.is_empty() => Ok(expr::add(cells.into_iter().map(expr::var).collect())),
                Some(_) => Err(format!("empty array reference `{name}`")),
                None => Ok(expr::var(self.var_id(name)?)),
            },
            TreeNode::Operator(op, kids) => {
                // Membership `in(x, set(...))` → OR of equalities; needs the raw
                // `Set` children, so it is handled before generic argument eval.
                if matches!(op, EOp::In) {
                    return self.build_in(kids);
                }
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

    /// Build `in(x, set(e1, e2, …))` as `x == e1 ∨ x == e2 ∨ …`.
    fn build_in(&self, kids: &[TreeNode]) -> Result<Expr, String> {
        let real: Vec<&TreeNode> = kids.iter().filter(|k| !matches!(k, TreeNode::LeftBracket | TreeNode::RightBracket)).collect();
        if real.len() != 2 {
            return Err("in: expected (value, set)".to_string());
        }
        let x = self.node(real[0])?;
        let members: Vec<Expr> = match real[1] {
            TreeNode::Operator(EOp::Set, set_kids) => set_kids
                .iter()
                .filter(|k| !matches!(k, TreeNode::LeftBracket | TreeNode::RightBracket))
                .map(|k| self.node(k))
                .collect::<Result<_, _>>()?,
            other => vec![self.node(other)?],
        };
        if members.is_empty() {
            return Ok(expr::int(0)); // x ∈ ∅ is false
        }
        Ok(expr::or(members.into_iter().map(|m| expr::eq(x.clone(), m)).collect()))
    }

    /// Aux variable equal to `e`, linked by an intension constraint.
    fn aux_for(&mut self, e: Expr) -> VarId {
        let aux = flatten::aux_for_expr(&mut self.solver, e.clone());
        self.local.add_expr(expr::eq(expr::var(aux), e));
        aux
    }

    /// Turn a list of parser expression trees into solver variables (aux vars
    /// for non-trivial expressions).
    fn tree_vars(&mut self, list: &[ExpressionTree]) -> Result<Vec<VarId>, String> {
        let mut out = Vec::with_capacity(list.len());
        for t in list {
            // A top-level whole-array reference expands to its cells, matching
            // how the parser expands the parallel `<coeffs>` list.
            if let Some(TreeNode::Variable(name)) = t.first_order_iter().next() {
                if let Some(cells) = self.expand_array_ref(name).or_else(|| self.expand_array_slice(name)) {
                    out.extend(cells);
                    continue;
                }
            }
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
    flatten::clamp_i64(x)
}

fn cell_ref(name: &str) -> Option<(&str, Vec<usize>)> {
    let open = name.find('[')?;
    let mut rest = &name[open..];
    let mut indices = Vec::new();
    while let Some(inner) = rest.strip_prefix('[') {
        let close = inner.find(']')?;
        indices.push(inner[..close].parse().ok()?);
        rest = &inner[(close + 1)..];
    }
    (!indices.is_empty() && rest.is_empty()).then_some((&name[..open], indices))
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

/// Chain a binary relation over a list: `f(a0,a1) ∧ f(a1,a2) ∧ …`. XCSP allows
/// n-ary `lt/le/gt/ge/eq` with this "sequence" meaning.
fn chain(args: Vec<Expr>, f: impl Fn(Expr, Expr) -> Expr) -> Result<Expr, String> {
    if args.len() < 2 {
        return Err("operator arity".to_string());
    }
    let mut terms: Vec<Expr> = args.windows(2).map(|w| f(w[0].clone(), w[1].clone())).collect();
    Ok(if terms.len() == 1 { terms.pop().unwrap() } else { expr::and(terms) })
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
        Lt => chain(args, expr::lt)?,
        Le => chain(args, expr::le)?,
        Ge => chain(args, expr::ge)?,
        Gt => chain(args, expr::gt)?,
        Eq => chain(args, expr::eq)?,
        Ne => {
            // n-ary `ne` means pairwise distinct.
            if args.len() < 2 {
                return Err("operator arity".to_string());
            }
            if args.len() == 2 {
                let (a, b) = two(args)?;
                expr::ne(a, b)
            } else {
                let mut terms = Vec::new();
                for i in 0..args.len() {
                    for j in (i + 1)..args.len() {
                        terms.push(expr::ne(args[i].clone(), args[j].clone()));
                    }
                }
                expr::and(terms)
            }
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
            // n-ary xor = odd parity; fold pairwise.
            if args.is_empty() {
                return Err("operator arity".to_string());
            }
            let mut it = args.into_iter();
            let mut acc = it.next().unwrap();
            for x in it {
                acc = expr::not(expr::iff(acc, x));
            }
            acc
        }
        If => {
            if args.len() != 3 {
                return Err("if arity".to_string());
            }
            let mut it = args.into_iter();
            expr::ite(it.next().unwrap(), it.next().unwrap(), it.next().unwrap())
        }
        Pow => {
            let (a, b) = two(args)?;
            match b {
                Expr::Const(0) => expr::int(1),
                Expr::Const(k) if (1..=8).contains(&k) => expr::mul(vec![a; k as usize]),
                _ => return Err("pow: non-constant or large exponent".to_string()),
            }
        }
        // `In` is handled in `node` (it needs the raw `Set` children); `Set`
        // is only meaningful as the second argument of `In`.
        Set | In => return Err("set/in operator out of context".to_string()),
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

    fn begin_group(&mut self) {
        self.share_extension_template = true;
        self.extension_template = None;
    }

    fn end_group(&mut self) {
        self.flush_sign_products();
        self.share_extension_template = false;
        self.extension_template = None;
    }

    fn begin_slide(&mut self) {
        self.begin_group();
    }

    fn end_slide(&mut self) {
        self.end_group();
    }

    // --- variables ---
    fn on_variable_interval(&mut self, id: String, min: i32, max: i32) {
        let v = self.solver.new_var_range(min, max);
        self.declared.push((id.clone(), v));
        self.local.add_var(v);
        self.remember_var(id, v);
    }
    fn on_variable_values(&mut self, id: String, values: &[i32]) {
        let v = self.solver.new_var_set(values);
        self.declared.push((id.clone(), v));
        self.local.add_var(v);
        self.remember_var(id, v);
    }

    // --- constraints ---
    fn on_constraint_extension(&mut self, list: &[String], tuples: &Vec<Vec<i32>>, is_support: bool, _has_star: bool) {
        guard!(self, {
            let vars = self.scope(list)?;
            let tuples = tuples.iter().map(|t| t.iter().map(|&v| if v == i32::MAX { STAR } else { v }).collect()).collect::<Vec<Vec<_>>>();
            self.local.add_extension(vars.clone(), tuples.clone(), is_support);
            let template = self.extension_template(vars.len(), || tuples);
            extension_from_template(&mut self.solver, &vars, template, is_support);
            Ok(())
        });
    }

    fn on_constraint_unary(&mut self, list: &String, values: &[i32], is_support: bool) {
        guard!(self, {
            let v = self.var_id(list)?;
            let tuples = values.iter().map(|&x| vec![x]).collect::<Vec<_>>();
            self.local.add_extension(vec![v], tuples.clone(), is_support);
            let template = self.extension_template(1, || tuples);
            extension_from_template(&mut self.solver, &[v], template, is_support);
            Ok(())
        });
    }

    fn on_constraint_intention(&mut self, _list: &[String], tree: &ExpressionTree) {
        guard!(self, {
            let e = self.tree(tree)?;
            self.local.add_expr(e.clone());
            if let Some(term) = self.sign_product(&e) {
                if self.share_extension_template {
                    self.pending_sign_products.push(term);
                } else {
                    sign_products(&mut self.solver, &[term]);
                }
            } else {
                crate::constraints::intension::intension(&mut self.solver, e);
            }
            Ok(())
        });
    }

    fn on_constraint_all_different_v1(&mut self, list: &[String]) {
        guard!(self, {
            let vars = self.scope(list)?;
            self.local.add_all_different(vars.clone());
            all_different(&mut self.solver, &vars);
            Ok(())
        });
    }

    fn on_constraint_all_different_v2(&mut self, list: &[ExpressionTree]) {
        guard!(self, {
            let vars = self.tree_vars(list)?;
            self.local.add_all_different(vars.clone());
            all_different(&mut self.solver, &vars);
            Ok(())
        });
    }

    fn on_constraint_all_different_list(&mut self, lists: &[Vec<String>]) {
        guard!(self, {
            if lists.len() == 1 {
                let vars = self.scope(&lists[0])?;
                self.local.add_all_different(vars.clone());
                all_different(&mut self.solver, &vars);
            } else {
                // allDifferent on lists: the tuples must be pairwise distinct,
                // i.e. each pair differs in at least one position.
                let tuples = lists.iter().map(|l| self.scope(l)).collect::<Result<Vec<_>, _>>()?;
                self.local.add_all_different_rows(tuples.clone());
                for i in 0..tuples.len() {
                    for j in (i + 1)..tuples.len() {
                        flatten::post_tuple_not_equal(&mut self.solver, &tuples[i], &tuples[j])?;
                    }
                }
            }
            Ok(())
        });
    }

    fn on_constraint_all_different_matrix(&mut self, lists: &[Vec<String>]) {
        guard!(self, {
            let (rows, cols) = self.matrix_shape(lists)?;
            let mut matrix = Vec::with_capacity(rows);
            for row in lists {
                let vars = self.scope(row)?;
                require(vars.len() == cols, "allDifferent matrix: ragged matrix")?;
                self.local.add_all_different(vars.clone());
                all_different(&mut self.solver, &vars);
                matrix.push(vars);
            }
            let columns = (0..cols).map(|c| (0..rows).map(|r| matrix[r][c]).collect::<Vec<_>>()).collect::<Vec<_>>();
            for col in columns {
                self.local.add_all_different(col.clone());
                all_different(&mut self.solver, &col);
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
            self.local.add_all_different_except(vars.clone(), except.to_vec());
            flatten::post_all_different_except(&mut self.solver, &vars, except);
            Ok(())
        });
    }

    fn on_constraint_all_equal_v1(&mut self, list: &[String]) {
        guard!(self, {
            let vars = self.scope(list)?;
            self.local.add_all_equal(vars.clone());
            all_equal(&mut self.solver, &vars);
            Ok(())
        });
    }

    fn on_constraint_ordered_v1(&mut self, list: &[String], operator: ROp) {
        guard!(self, {
            let vars = self.scope(list)?;
            let rel = Model::rel(operator)?;
            for pair in vars.windows(2) {
                self.local.add_linear(vec![1, -1], vec![pair[0], pair[1]], rel, 0);
            }
            ordered(&mut self.solver, &vars, rel);
            Ok(())
        });
    }
    fn on_constraint_ordered_v3(&mut self, list: &[String], lengths: &[String], operator: ROp) {
        guard!(self, {
            // XCSP gives either n lengths (last unused) or n-1.
            if lengths.len() + 1 < list.len() {
                return Err("ordered: too few lengths".to_string());
            }
            let vars = self.scope(list)?;
            let lens = self.scope(lengths)?;
            let rel = Model::rel(operator)?;
            // Chain: x[i] + len[i]  rel  x[i+1].
            for i in 0..vars.len().saturating_sub(1) {
                linear(&mut self.solver, &[1, 1, -1], &[vars[i], lens[i], vars[i + 1]], rel, 0);
                self.local.add_linear(vec![1, 1, -1], vec![vars[i], lens[i], vars[i + 1]], rel, 0);
            }
            Ok(())
        });
    }

    fn on_constraint_ordered_v2(&mut self, list: &[String], lengths: &[i32], operator: ROp) {
        guard!(self, {
            // XCSP gives either n lengths (last unused) or n-1.
            if lengths.len() + 1 < list.len() {
                return Err("ordered: too few lengths".to_string());
            }
            let vars = self.scope(list)?;
            let rel = Model::rel(operator)?;
            // Chain: x[i] + len[i]  rel  x[i+1]; constant len folds into the rhs.
            for i in 0..vars.len().saturating_sub(1) {
                let rhs = i64::from(-lengths[i]);
                linear(&mut self.solver, &[1, -1], &[vars[i], vars[i + 1]], rel, rhs);
                self.local.add_linear(vec![1, -1], vec![vars[i], vars[i + 1]], rel, rhs);
            }
            Ok(())
        });
    }

    fn on_constraint_instantiation(&mut self, list: &[String], values: &[i32]) {
        guard!(self, {
            let vars = self.scope(list)?;
            for (&var, &value) in vars.iter().zip(values) {
                self.local.add_linear(vec![1], vec![var], Relation::Eq, value as i64);
            }
            instantiation(&mut self.solver, &vars, values);
            Ok(())
        });
    }

    fn on_constraint_sum_v1(&mut self, list: &[String], operator: ROp, operand: Operand) {
        guard!(self, { self.post_sum_vars(list, None, operator, operand) });
    }
    fn on_constraint_sum_v2(&mut self, list: &[String], coeffs: &[i32], operator: ROp, operand: Operand) {
        guard!(self, { self.post_sum_vars(list, Some(coeffs), operator, operand) });
    }
    fn on_constraint_sum_v3(&mut self, list: &[String], coeffs: &[String], operator: ROp, operand: Operand) {
        guard!(self, {
            // Variable coefficients: materialise p_i = c_i · x_i as intension-backed
            // aux vars, then post a unit-coefficient linear sum over them.
            let prods = self.var_coeff_products(list, coeffs, "sum")?;
            let coeffs = ones(prods.len());
            self.post_sum(coeffs, prods, operator, operand)
        });
    }
    fn on_constraint_sum_v4(&mut self, list: &[ExpressionTree], operator: ROp, operand: Operand) {
        guard!(self, { self.post_sum_exprs(list, None, operator, operand) });
    }
    fn on_constraint_sum_v5(&mut self, list: &[ExpressionTree], coeffs: &[i32], operator: ROp, operand: Operand) {
        guard!(self, { self.post_sum_exprs(list, Some(coeffs), operator, operand) });
    }

    fn on_constraint_count_v2(&mut self, list: &[String], values: &[i32], operator: ROp, operand: Operand) {
        guard!(self, {
            let vars = self.scope(list)?;
            self.post_count_condition(&vars, values, operator, operand)
        });
    }
    fn on_constraint_count_v4(&mut self, list: &[String], values: &[String], operator: ROp, operand: Operand) {
        guard!(self, {
            let vars = self.scope(list)?;
            let vals = self.scope(values)?;
            // indicator_i = 1  iff  list[i] ∈ {values} (variable targets), via a
            // reified equality (OR over several targets); then count indicators == 1.
            let inds: Vec<VarId> = vars
                .iter()
                .map(|&xi| {
                    let mut memberships: Vec<Expr> = vals.iter().map(|&v| expr::eq(expr::var(xi), expr::var(v))).collect();
                    let e = if memberships.len() == 1 { memberships.pop().unwrap() } else { expr::or(memberships) };
                    self.aux_for(e)
                })
                .collect();
            self.post_count_condition(&inds, &[1], operator, operand)
        });
    }
    fn on_constraint_count_v1(&mut self, list: &[ExpressionTree], values: &[i32], operator: ROp, operand: Operand) {
        guard!(self, {
            let vars = self.tree_vars(list)?;
            self.post_count_condition(&vars, values, operator, operand)
        });
    }

    fn on_constraint_minimum_v1(&mut self, list: &[String], operator: ROp, operand: Operand) {
        guard!(self, { self.min_max(list, operator, operand, true) });
    }
    fn on_constraint_maximum_v1(&mut self, list: &[String], operator: ROp, operand: Operand) {
        guard!(self, { self.min_max(list, operator, operand, false) });
    }
    fn on_constraint_minimum_v2(&mut self, list: &[ExpressionTree], operator: ROp, operand: Operand) {
        guard!(self, {
            let xs = self.tree_vars(list)?;
            self.min_max_vars(xs, operator, operand, true)
        });
    }
    fn on_constraint_maximum_v2(&mut self, list: &[ExpressionTree], operator: ROp, operand: Operand) {
        guard!(self, {
            let xs = self.tree_vars(list)?;
            self.min_max_vars(xs, operator, operand, false)
        });
    }

    fn on_constraint_element_v1(&mut self, list: &[String], value: i32) {
        guard!(self, {
            // value belongs to the list (existential element with free index).
            let array = self.scope(list)?;
            self.local.add_element_member(array.clone(), value);
            let n = array.len() as i32;
            let idx = self.solver.new_var_range(0, n - 1);
            let val = self.constant(value);
            element(&mut self.solver, &array, idx, val);
            Ok(())
        });
    }
    fn on_constraint_element_v4(&mut self, list: &[String], start_index: i32, index: String, value: i32) {
        guard!(self, {
            let array = self.scope(list)?;
            let val = self.constant(value);
            self.post_element(array, start_index, &index, val)
        });
    }
    fn on_constraint_element_v3(&mut self, list: &[String], start_index: i32, index: String, value: String) {
        guard!(self, {
            let array = self.scope(list)?;
            let val = self.var_id(&value)?;
            self.post_element(array, start_index, &index, val)
        });
    }
    fn on_constraint_element_v5(&mut self, list: &[String], start_index: i32, index: String, operator: ROp, operand: Operand) {
        guard!(self, {
            let array = self.scope(list)?;
            self.post_element_cond(array, start_index, &index, operator, operand)
        });
    }
    fn on_constraint_element_v8(&mut self, list: &[i32], start_index: i32, index: String, operator: ROp, operand: Operand) {
        guard!(self, {
            let array = self.consts(list);
            self.post_element_cond(array, start_index, &index, operator, operand)
        });
    }
    fn on_constraint_element_v6(&mut self, list: &[i32], start_index: i32, index: String, value: String) {
        guard!(self, {
            let array = self.consts(list);
            let val = self.var_id(&value)?;
            self.post_element(array, start_index, &index, val)
        });
    }
    fn on_constraint_element_v7(&mut self, list: &[i32], start_index: i32, index: String, value: i32) {
        guard!(self, {
            let array = self.consts(list);
            let val = self.constant(value);
            self.post_element(array, start_index, &index, val)
        });
    }

    fn on_constraint_element_matrix_v1(
        &mut self,
        matrix: &Vec<Vec<String>>,
        row_index: String,
        col_index: String,
        start_row_index: i32,
        start_col_index: i32,
        value: i32,
    ) {
        guard!(self, {
            let (rows, cols) = self.matrix_shape(matrix)?;
            let array = matrix.iter().flatten().map(|s| self.var_id(s)).collect::<Result<Vec<_>, _>>()?;
            let val = self.constant(value);
            let access = MatrixAccess { rows, cols, row_index: &row_index, col_index: &col_index, start_row_index, start_col_index };
            self.post_element_matrix(array, access, val)
        });
    }
    fn on_constraint_element_matrix_v2(
        &mut self,
        matrix: &Vec<Vec<String>>,
        row_index: String,
        col_index: String,
        start_row_index: i32,
        start_col_index: i32,
        value: String,
    ) {
        guard!(self, {
            let (rows, cols) = self.matrix_shape(matrix)?;
            let array = matrix.iter().flatten().map(|s| self.var_id(s)).collect::<Result<Vec<_>, _>>()?;
            let val = self.var_id(&value)?;
            let access = MatrixAccess { rows, cols, row_index: &row_index, col_index: &col_index, start_row_index, start_col_index };
            self.post_element_matrix(array, access, val)
        });
    }
    fn on_constraint_element_matrix_v5(
        &mut self,
        matrix: &Vec<Vec<i32>>,
        row_index: String,
        col_index: String,
        start_row_index: i32,
        start_col_index: i32,
        value: i32,
    ) {
        guard!(self, {
            let (rows, cols) = self.matrix_shape(matrix)?;
            let array = matrix.iter().flatten().map(|&v| self.constant(v)).collect();
            let val = self.constant(value);
            let access = MatrixAccess { rows, cols, row_index: &row_index, col_index: &col_index, start_row_index, start_col_index };
            self.post_element_matrix(array, access, val)
        });
    }
    fn on_constraint_element_matrix_v6(
        &mut self,
        matrix: &Vec<Vec<i32>>,
        row_index: String,
        col_index: String,
        start_row_index: i32,
        start_col_index: i32,
        value: String,
    ) {
        guard!(self, {
            let (rows, cols) = self.matrix_shape(matrix)?;
            let array = matrix.iter().flatten().map(|&v| self.constant(v)).collect();
            let val = self.var_id(&value)?;
            let access = MatrixAccess { rows, cols, row_index: &row_index, col_index: &col_index, start_row_index, start_col_index };
            self.post_element_matrix(array, access, val)
        });
    }

    fn on_constraint_cumulative_v1(&mut self, origins: &[String], lengths: &[i32], heights: &[i32], operator: ROp, operand: Operand) {
        guard!(self, {
            require(matches!(operator, ROp::Le), "cumulative condition must be <=")?;
            let starts = self.scope(origins)?;
            let d = i64s(lengths);
            match self.rhs(&operand)? {
                // Fixed heights + constant capacity: the strong edge-finder.
                Rhs::Const(k) => {
                    let dv = self.consts(lengths);
                    let h = self.consts(heights);
                    self.local.add_cumulative(starts.clone(), dv, h, LocalRhs::Const(k));
                    let h = i64s(heights);
                    cumulative(&mut self.solver, &starts, &d, &h, k);
                }
                // Variable capacity: the weaker variable-resource propagator.
                Rhs::Var(cap) => {
                    let dv = self.consts(lengths);
                    let h = self.consts(heights);
                    self.local.add_cumulative(starts.clone(), dv.clone(), h.clone(), LocalRhs::Var(cap));
                    cumulative_var(&mut self.solver, &starts, &dv, &h, cap);
                }
            }
            Ok(())
        });
    }
    fn on_constraint_cumulative_v2(&mut self, origins: &[String], lengths: &[i32], heights: &[String], operator: ROp, operand: Operand) {
        guard!(self, {
            let dv = self.consts(lengths);
            let h = self.scope(heights)?;
            self.post_cumulative_var(origins, dv, h, operator, &operand)
        });
    }
    fn on_constraint_cumulative_v3(&mut self, origins: &[String], lengths: &[String], heights: &[i32], operator: ROp, operand: Operand) {
        guard!(self, {
            let dv = self.scope(lengths)?;
            let h = self.consts(heights);
            self.post_cumulative_var(origins, dv, h, operator, &operand)
        });
    }
    fn on_constraint_cumulative_v4(&mut self, origins: &[String], lengths: &[String], heights: &[String], operator: ROp, operand: Operand) {
        guard!(self, {
            let dv = self.scope(lengths)?;
            let h = self.scope(heights)?;
            self.post_cumulative_var(origins, dv, h, operator, &operand)
        });
    }

    fn on_constraint_nvalues_v1(&mut self, list: &[String], operator: ROp, operand: Operand) {
        guard!(self, {
            let vars = self.scope(list)?;
            self.post_nvalues(vars, operator, operand)
        });
    }

    fn on_constraint_nvalues_v3(&mut self, list: &[ExpressionTree], operator: ROp, operand: Operand) {
        guard!(self, {
            let vars = self.tree_vars(list)?;
            self.post_nvalues(vars, operator, operand)
        });
    }

    fn on_constraint_channel_v1(&mut self, list: &[String], start_index: i32) {
        guard!(self, {
            let vars = self.scope(list)?;
            self.local.add_channel_inverse(vars.clone(), start_index, vars.clone(), start_index);
            if start_index == 0 {
                channel(&mut self.solver, &vars, &vars);
            } else {
                // The plain propagator is 0-based; shifted positions need the
                // offset-aware decomposition (mirrors the two-list variant).
                let vars2 = vars.clone();
                self.post_channel_inverse_kernel(&vars, start_index, &vars2, start_index);
            }
            Ok(())
        });
    }
    fn on_constraint_channel_v2(&mut self, list1: &[String], start_index1: i32, list2: &[String], start_index2: i32) {
        guard!(self, {
            let xs = self.scope(list1)?;
            let ys = self.scope(list2)?;
            self.local.add_channel_inverse(xs.clone(), start_index1, ys.clone(), start_index2);
            self.post_channel_inverse_kernel(&xs, start_index1, &ys, start_index2);
            Ok(())
        });
    }
    fn on_constraint_channel_v3(&mut self, list: &[String], start_index: i32, value: String) {
        guard!(self, {
            // 0/1 list; `value` is the index of the single 1:
            // x[i] == 1  ⟺  value == i + start_index.
            let xs = self.scope(list)?;
            let v = self.var_id(&value)?;
            self.local.add_channel_onehot(xs.clone(), v, start_index);
            flatten::post_channel_onehot_index(&mut self.solver, &xs, v, start_index);
            Ok(())
        });
    }

    fn on_constraint_precedence_v1(&mut self, list: &[String], covered: bool) {
        guard!(self, {
            // Default value order: the sorted distinct values across all domains.
            let vars = self.scope(list)?;
            let mut vals: Vec<i32> = Vec::new();
            for &v in &vars {
                vals.extend(self.solver.store.values(v));
            }
            vals.sort_unstable();
            vals.dedup();
            self.local.add_precedence(vars.clone(), vals.clone(), covered);
            crate::constraints::primitives::precedence_with_covered(&mut self.solver, &vars, &vals, covered);
            Ok(())
        });
    }
    fn on_constraint_precedence_v2(&mut self, list: &[String], values: &[i32], covered: bool) {
        guard!(self, {
            let vars = self.scope(list)?;
            self.local.add_precedence(vars.clone(), values.to_vec(), covered);
            crate::constraints::primitives::precedence_with_covered(&mut self.solver, &vars, values, covered);
            Ok(())
        });
    }

    fn on_constraint_circuit_v1(&mut self, list: &Vec<String>) {
        guard!(self, {
            let vars = self.scope(list)?;
            self.local.add_circuit(vars.clone());
            circuit(&mut self.solver, &vars);
            Ok(())
        });
    }

    fn on_constraint_lex(&mut self, lists: &Vec<Vec<String>>, operator: ROp) {
        guard!(self, { self.post_lex_rows(lists, operator) });
    }
    fn on_constraint_lex_matrix(&mut self, matrix: &Vec<Vec<String>>, operator: ROp) {
        guard!(self, { self.post_lex_rows(matrix, operator) });
    }

    fn on_constraint_cardinality_v1(&mut self, list: &[String], values: &[i32], occurs: &[i32], closed: bool) {
        guard!(self, {
            if occurs.len() != values.len() {
                return Err("cardinality: occurs/values length mismatch".to_string());
            }
            let vars = self.scope(list)?;
            let low = i64s(occurs);
            self.local.add_cardinality(vars.clone(), values.to_vec(), low.clone(), low.clone(), closed);
            cardinality(&mut self.solver, &vars, values, &low, &low, closed);
            Ok(())
        });
    }
    fn on_constraint_cardinality_v2(&mut self, list: &[String], values: &[i32], occurs: &[String], closed: bool) {
        guard!(self, {
            if occurs.len() != values.len() {
                return Err("cardinality: occurs/values length mismatch".to_string());
            }
            let vars = self.scope(list)?;
            let occ = self.scope(occurs)?;
            for (&value, &target) in values.iter().zip(&occ) {
                self.local.add_count(vars.clone(), vec![value], Relation::Eq, LocalRhs::Var(target));
                flatten::count_values_to_var(&mut self.solver, &vars, &[value], Relation::Eq, target);
            }
            if closed {
                let low = vec![0; values.len()];
                let high = vec![vars.len() as i64; values.len()];
                self.local.add_cardinality(vars.clone(), values.to_vec(), low.clone(), high.clone(), true);
                cardinality(&mut self.solver, &vars, values, &low, &high, true);
            }
            Ok(())
        });
    }
    fn on_constraint_cardinality_v3(&mut self, list: &[String], values: &[i32], occurs: &[(i32, i32)], closed: bool) {
        guard!(self, {
            if occurs.len() != values.len() {
                return Err("cardinality: occurs/values length mismatch".to_string());
            }
            let vars = self.scope(list)?;
            let low: Vec<i64> = occurs.iter().map(|&(l, _)| l as i64).collect();
            let high: Vec<i64> = occurs.iter().map(|&(_, h)| h as i64).collect();
            self.local.add_cardinality(vars.clone(), values.to_vec(), low.clone(), high.clone(), closed);
            cardinality(&mut self.solver, &vars, values, &low, &high, closed);
            Ok(())
        });
    }

    fn on_constraint_bin_packing_v1(&mut self, list: &[String], sizes: &[i32], operator: ROp, operand: Operand) {
        guard!(self, {
            let items = self.scope(list)?;
            let s = i64s(sizes);
            let cap = match (operator, self.rhs(&operand)?) {
                (ROp::Le, Rhs::Const(k)) => k,
                _ => return Err("binPacking needs <= constant capacity".to_string()),
            };
            // count bins from the item domains
            let nbins = (0..items.len()).map(|i| self.solver.store.max(items[i]) + 1).max().unwrap_or(0).max(1) as usize;
            self.local.add_bin_packing(items.clone(), s.clone(), vec![LocalRhs::Const(cap); nbins], false);
            bin_packing(&mut self.solver, &items, &s, &vec![cap; nbins]);
            Ok(())
        });
    }
    fn on_constraint_bin_packing_v2(&mut self, list: &[String], sizes: &[i32], limits: &[i32]) {
        guard!(self, {
            let items = self.scope(list)?;
            let s = i64s(sizes);
            let caps = i64s(limits);
            self.local.add_bin_packing(items.clone(), s.clone(), caps.iter().copied().map(LocalRhs::Const).collect(), false);
            bin_packing(&mut self.solver, &items, &s, &caps);
            Ok(())
        });
    }
    fn on_constraint_bin_packing_v3(&mut self, list: &[String], sizes: &[i32], limits: &[String]) {
        guard!(self, {
            // Variable capacities: load_b = Σ_i size_i·[item_i==b]  must be ≤ limit_b.
            let items = self.scope(list)?;
            let lim = self.scope(limits)?;
            let s = i64s(sizes);
            self.local.add_bin_packing(items.clone(), s.clone(), lim.iter().copied().map(LocalRhs::Var).collect(), false);
            let total: i64 = s.iter().sum();
            let loads: Vec<VarId> = (0..lim.len()).map(|_| self.solver.new_var_range(0, clamp(total))).collect();
            flatten::post_bin_loads(&mut self.solver, &items, &s, &loads);
            for (&load, &cap) in loads.iter().zip(&lim) {
                linear(&mut self.solver, &[1, -1], &[load, cap], Relation::Le, 0);
            }
            Ok(())
        });
    }
    fn on_constraint_bin_packing_v4(&mut self, list: &[String], sizes: &[i32], loads: &[i32]) {
        guard!(self, {
            // Each bin's load is fixed to the given constant loads[b].
            let items = self.scope(list)?;
            let s = i64s(sizes);
            self.local.add_bin_packing(items.clone(), s.clone(), loads.iter().map(|&k| LocalRhs::Const(k as i64)).collect(), true);
            let loadv: Vec<VarId> = loads.iter().map(|&k| self.solver.new_var_range(k, k)).collect();
            flatten::post_bin_loads(&mut self.solver, &items, &s, &loadv);
            Ok(())
        });
    }
    fn on_constraint_bin_packing_v5(&mut self, list: &[String], sizes: &[i32], loads: &[String]) {
        guard!(self, {
            // Each bin's load variable equals the total size assigned to it:
            // load_b = Σ_i size_i · [item_i == b].
            let items = self.scope(list)?;
            let loadv = self.scope(loads)?;
            let s = i64s(sizes);
            self.local.add_bin_packing(items.clone(), s.clone(), loadv.iter().copied().map(LocalRhs::Var).collect(), true);
            flatten::post_bin_loads(&mut self.solver, &items, &s, &loadv);
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
            let w = i64s(weights);
            let p = i64s(profits);
            // Σ w·x ≤ W and Σ p·x ≥ P (W, P may be variables).
            self.post_linear(w, vars.clone(), Relation::Le, self.rhs(&w_operand)?)?;
            self.post_linear(p, vars, Relation::Ge, self.rhs(&p_operand)?)?;
            Ok(())
        });
    }

    fn on_constraint_no_overlap_v1(&mut self, list: &[String], lengths: &[i32], zero_ignored: bool) {
        guard!(self, {
            let starts = self.scope(list)?;
            let d = i64s(lengths);
            let origins = starts.iter().copied().map(|start| vec![start]).collect::<Vec<_>>();
            let len = lengths.iter().map(|&length| vec![expr::int(length as i64)]).collect::<Vec<_>>();
            if zero_ignored || lengths.iter().any(|&length| length <= 0) {
                self.post_diffn(origins, len, zero_ignored)?;
            } else {
                self.local.add_no_overlap(origins, len, zero_ignored);
                no_overlap(&mut self.solver, &starts, &d);
            }
            Ok(())
        });
    }
    fn on_constraint_no_overlap_v2(&mut self, list: &[String], lengths: &[String], zero: bool) {
        guard!(self, {
            if list.len() != lengths.len() {
                return Err("noOverlap: list/lengths length mismatch".to_string());
            }
            let mut org = Vec::with_capacity(list.len());
            let mut len = Vec::with_capacity(list.len());
            for (s, l) in list.iter().zip(lengths) {
                org.push(vec![self.var_id(s)?]);
                len.push(vec![expr::var(self.var_id(l)?)]);
            }
            self.post_diffn(org, len, zero)
        });
    }
    fn on_constraint_no_overlap_k_dim_v1(&mut self, origins: &Vec<Vec<String>>, lengths: &Vec<Vec<i32>>, zero: bool) {
        guard!(self, {
            let org = self.origins_to_ids(origins)?;
            let len: Vec<Vec<Expr>> = lengths.iter().map(|b| b.iter().map(|&l| expr::int(l as i64)).collect()).collect();
            self.post_diffn(org, len, zero)
        });
    }
    fn on_constraint_no_overlap_k_dim_v2(&mut self, origins: &Vec<Vec<String>>, lengths: &Vec<Vec<String>>, zero: bool) {
        guard!(self, {
            let org = self.origins_to_ids(origins)?;
            let mut len = Vec::with_capacity(lengths.len());
            for b in lengths {
                let mut row = Vec::with_capacity(b.len());
                for s in b {
                    row.push(expr::var(self.var_id(s)?));
                }
                len.push(row);
            }
            self.post_diffn(org, len, zero)
        });
    }
    fn on_constraint_no_overlap_k_dim_v3(&mut self, origins: &Vec<Vec<String>>, lengths: &Vec<(String, i32)>, zero: bool) {
        guard!(self, {
            let org = self.origins_to_ids(origins)?;
            let mut len = Vec::with_capacity(lengths.len());
            for &(ref s, l) in lengths {
                len.push(vec![expr::var(self.var_id(s)?), expr::int(l as i64)]);
            }
            self.post_diffn(org, len, zero)
        });
    }

    fn on_constraint_regular(&mut self, list: &[String], start: String, finals: &[String], transitions: &[(String, i32, String)]) {
        guard!(self, {
            let vars = self.scope(list)?;
            let mut states: HashMap<String, usize> = HashMap::new();
            let state_id = |s: &str, m: &mut HashMap<String, usize>| {
                let n = m.len();
                *m.entry(s.to_string()).or_insert(n)
            };
            let trans: Vec<(usize, i32, usize)> =
                transitions.iter().map(|(a, v, b)| (state_id(a, &mut states), *v, state_id(b, &mut states))).collect();
            let start_s = state_id(&start, &mut states);
            let accept: Vec<usize> = finals.iter().map(|s| state_id(s, &mut states)).collect();
            let dfa = Dfa { n_states: states.len(), start: start_s, accept, transitions: trans };
            self.local.add_regular(vars.clone(), dfa.clone());
            regular(&mut self.solver, &vars, dfa);
            Ok(())
        });
    }

    fn on_constraint_mdd(&mut self, list: &[String], transitions: &Vec<(String, i32, String)>) {
        guard!(self, {
            let vars = self.scope(list)?;
            // Layer per variable, inferred from a root reachable BFS over labels.
            let m = build_mdd(vars.len(), transitions)?;
            self.local.add_mdd(vars.clone(), m.clone());
            mdd(&mut self.solver, &vars, m);
            Ok(())
        });
    }

    // --- objectives ---
    fn on_minimize_var(&mut self, var: String) {
        guard!(self, { self.set_var_objective(&var, true) });
    }
    fn on_maximize_var(&mut self, var: String) {
        guard!(self, { self.set_var_objective(&var, false) });
    }
    fn on_minimize_expression(&mut self, e: &ExpressionTree) {
        guard!(self, { self.set_expr_objective(e, true) });
    }
    fn on_maximize_expression(&mut self, e: &ExpressionTree) {
        guard!(self, { self.set_expr_objective(e, false) });
    }
    fn on_minimize_v1(&mut self, t: XElementOperator, list: &[String], coeffs: &[i32]) {
        guard!(self, { self.objective_sum(t, list, Some(coeffs), true) });
    }
    fn on_maximize_v1(&mut self, t: XElementOperator, list: &[String], coeffs: &[i32]) {
        guard!(self, { self.objective_sum(t, list, Some(coeffs), false) });
    }
    fn on_minimize_v2(&mut self, t: XElementOperator, list: &[String], coeffs: &[String]) {
        guard!(self, { self.objective_sum_varcoeffs(t, list, coeffs, true) });
    }
    fn on_maximize_v2(&mut self, t: XElementOperator, list: &[String], coeffs: &[String]) {
        guard!(self, { self.objective_sum_varcoeffs(t, list, coeffs, false) });
    }
    fn on_minimize_v5(&mut self, t: XElementOperator, list: &[String]) {
        guard!(self, { self.objective_sum(t, list, None, true) });
    }
    fn on_maximize_v5(&mut self, t: XElementOperator, list: &[String]) {
        guard!(self, { self.objective_sum(t, list, None, false) });
    }
    fn on_minimize_v3(&mut self, t: XElementOperator, list: &[ExpressionTree], coeffs: &[i32]) {
        guard!(self, { self.objective_exprs(t, list, Some(coeffs), true) });
    }
    fn on_maximize_v3(&mut self, t: XElementOperator, list: &[ExpressionTree], coeffs: &[i32]) {
        guard!(self, { self.objective_exprs(t, list, Some(coeffs), false) });
    }
    fn on_minimize_v6(&mut self, t: XElementOperator, list: &[ExpressionTree]) {
        guard!(self, { self.objective_exprs(t, list, None, true) });
    }
    fn on_maximize_v6(&mut self, t: XElementOperator, list: &[ExpressionTree]) {
        guard!(self, { self.objective_exprs(t, list, None, false) });
    }
}

impl Model {
    fn min_max(&mut self, list: &[String], operator: ROp, operand: Operand, is_min: bool) -> Result<(), String> {
        let xs = self.scope(list)?;
        self.min_max_vars(xs, operator, operand, is_min)
    }

    /// `minimum`/`maximum` over already-resolved variables.
    fn min_max_vars(&mut self, xs: Vec<VarId>, operator: ROp, operand: Operand, is_min: bool) -> Result<(), String> {
        let rel = Model::rel(operator)?;
        let y = self.rhs_var(&operand)?;
        self.local.add_extremum(xs.clone(), is_min, rel, LocalRhs::Var(y));
        if matches!(rel, Relation::Eq) {
            if is_min {
                minimum(&mut self.solver, y, &xs);
            } else {
                maximum(&mut self.solver, y, &xs);
            }
        } else {
            // aux = min/max(xs); then aux rel y.
            let (lo, hi) = (
                xs.iter().map(|&v| self.solver.store.min(v)).min().unwrap_or(0),
                xs.iter().map(|&v| self.solver.store.max(v)).max().unwrap_or(0),
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

    /// `nValues` over already-resolved variables: count distinct, then `rel rhs`.
    fn post_nvalues(&mut self, vars: Vec<VarId>, operator: ROp, operand: Operand) -> Result<(), String> {
        // Decode through the generic condition handler so interval/`notin`
        // conditions expand to the right set of relations.
        for (rel, rhs) in self.conditions(operator, operand)? {
            self.local.add_n_values(vars.clone(), rel, Self::local_rhs(&rhs));
            match rhs {
                Rhs::Const(k) => n_values(&mut self.solver, &vars, rel, k),
                Rhs::Var(y) => {
                    let aux = flatten::nvalues_var(&mut self.solver, &vars);
                    linear(&mut self.solver, &[1, -1], &[aux, y], rel, 0);
                }
            }
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
                flatten::count_values_to_var(&mut self.solver, vars, values, rel, y);
            }
            Rhs::Var(y) => flatten::count_values_to_var(&mut self.solver, vars, values, rel, y),
        }
    }

    fn origins_to_ids(&self, origins: &[Vec<String>]) -> Result<Vec<Vec<VarId>>, String> {
        origins.iter().map(|b| b.iter().map(|s| self.var_id(s)).collect()).collect()
    }

    /// Offset-aware kernel encoding of `channel(xs, ys)`: value bounds plus the
    /// pairwise iff decomposition. Used whenever a start index shifts positions
    /// out of the plain 0-based `channel` propagator's frame.
    fn post_channel_inverse_kernel(&mut self, xs: &[VarId], start_index1: i32, ys: &[VarId], start_index2: i32) {
        for &x in xs {
            linear(&mut self.solver, &[1], &[x], Relation::Ge, start_index2 as i64);
            linear(&mut self.solver, &[1], &[x], Relation::Le, start_index2 as i64 + ys.len() as i64 - 1);
        }
        for &y in ys {
            linear(&mut self.solver, &[1], &[y], Relation::Ge, start_index1 as i64);
            linear(&mut self.solver, &[1], &[y], Relation::Le, start_index1 as i64 + xs.len() as i64 - 1);
        }
        for (i, &x) in xs.iter().enumerate() {
            for (j, &y) in ys.iter().enumerate() {
                let xv = expr::eq(expr::var(x), expr::int(start_index2 as i64 + j as i64));
                let yv = expr::eq(expr::var(y), expr::int(start_index1 as i64 + i as i64));
                crate::constraints::intension::intension(&mut self.solver, expr::iff(xv, yv));
            }
        }
    }

    fn post_diffn(&mut self, origins: Vec<Vec<VarId>>, lengths: Vec<Vec<Expr>>, zero_ignored: bool) -> Result<(), String> {
        self.local.add_no_overlap(origins.clone(), lengths.clone(), zero_ignored);
        flatten::post_diffn(&mut self.solver, &origins, &lengths, zero_ignored)
    }

    /// Fold a `startIndex` offset into the index variable, returning a fresh
    /// 0-based index var constrained to `idx - start_index`, or `idx` unchanged
    /// when the list is already 0-based. Same technique as `matrix_index`.
    fn zero_based_index(&mut self, idx: VarId, start_index: i32, len: usize) -> VarId {
        if start_index == 0 {
            return idx;
        }
        let idx0 = self.solver.new_var_range(0, len as i32 - 1);
        let offset = i64::from(start_index);
        linear(&mut self.solver, &[1, -1], &[idx0, idx], Relation::Eq, -offset);
        self.local.add_expr(expr::eq(expr::var(idx0), expr::add(vec![expr::var(idx), expr::int(-offset)])));
        idx0
    }

    fn post_element(&mut self, array: Vec<VarId>, start_index: i32, index: &str, value: VarId) -> Result<(), String> {
        let idx = self.var_id(index)?;
        let idx = self.zero_based_index(idx, start_index, array.len());
        self.local.add_element(array.clone(), idx, value, 0);
        element(&mut self.solver, &array, idx, value);
        Ok(())
    }

    fn post_element_cond(
        &mut self,
        array: Vec<VarId>,
        start_index: i32,
        index: &str,
        operator: ROp,
        operand: Operand,
    ) -> Result<(), String> {
        let idx = self.var_id(index)?;
        let idx = self.zero_based_index(idx, start_index, array.len());
        self.element_cond(array, idx, operator, operand)
    }

    fn matrix_shape<T>(&self, matrix: &[Vec<T>]) -> Result<(usize, usize), String> {
        require(!matrix.is_empty(), "element matrix: empty matrix")?;
        let cols = matrix[0].len();
        require(cols > 0, "element matrix: empty row")?;
        for row in matrix {
            require(row.len() == cols, "element matrix: ragged matrix")?;
        }
        Ok((matrix.len(), cols))
    }

    fn matrix_index(&mut self, access: MatrixAccess<'_>) -> Result<VarId, String> {
        let len = access.rows.checked_mul(access.cols).ok_or_else(|| "element matrix: too large".to_string())?;
        require(len <= i32::MAX as usize, "element matrix: too large")?;
        let row = self.var_or_constant(access.row_index)?;
        let col = self.var_or_constant(access.col_index)?;
        let idx = self.solver.new_var_range(0, len as i32 - 1);
        let offset = access.cols as i64 * i64::from(access.start_row_index) + i64::from(access.start_col_index);
        linear(&mut self.solver, &[1, -(access.cols as i64), -1], &[idx, row, col], Relation::Eq, -offset);
        let idx_expr = expr::add(vec![expr::mul(vec![expr::int(access.cols as i64), expr::var(row)]), expr::var(col), expr::int(-offset)]);
        self.local.add_expr(expr::eq(expr::var(idx), idx_expr));
        Ok(idx)
    }

    fn post_element_matrix(&mut self, array: Vec<VarId>, access: MatrixAccess<'_>, value: VarId) -> Result<(), String> {
        let idx = self.matrix_index(access)?;
        self.local.add_element(array.clone(), idx, value, 0);
        element(&mut self.solver, &array, idx, value);
        Ok(())
    }

    fn post_cumulative_var(
        &mut self,
        origins: &[String],
        durations: Vec<VarId>,
        heights: Vec<VarId>,
        operator: ROp,
        operand: &Operand,
    ) -> Result<(), String> {
        require(matches!(operator, ROp::Le), "cumulative condition must be <=")?;
        let starts = self.scope(origins)?;
        let cap = self.rhs_var(operand)?;
        self.local.add_cumulative(starts.clone(), durations.clone(), heights.clone(), LocalRhs::Var(cap));
        cumulative_var(&mut self.solver, &starts, &durations, &heights, cap);
        Ok(())
    }

    /// `aux = array[index]` then `aux  rel  operand`.
    fn element_cond(&mut self, array: Vec<VarId>, idx: VarId, operator: ROp, operand: Operand) -> Result<(), String> {
        let lo = array.iter().map(|&v| self.solver.store.min(v)).min().unwrap_or(0);
        let hi = array.iter().map(|&v| self.solver.store.max(v)).max().unwrap_or(0);
        let aux = self.solver.new_var_range(lo, hi);
        element(&mut self.solver, &array, idx, aux);
        let rel = Model::rel(operator)?;
        let y = self.rhs_var(&operand)?;
        self.local.add_element(array.clone(), idx, aux, 0);
        self.local.add_linear(vec![1, -1], vec![aux, y], rel, 0);
        linear(&mut self.solver, &[1, -1], &[aux, y], rel, 0);
        Ok(())
    }

    fn set_var_objective(&mut self, var: &str, minimize: bool) -> Result<(), String> {
        self.objective = Some(Objective::Var(minimize, self.var_id(var)?));
        Ok(())
    }

    fn set_expr_objective(&mut self, tree: &ExpressionTree, minimize: bool) -> Result<(), String> {
        let objective_expr = self.tree(tree)?;
        let aux = self.aux_for(objective_expr);
        self.objective = Some(Objective::Var(minimize, aux));
        Ok(())
    }

    /// Objective over named variables.
    fn objective_sum(&mut self, t: XElementOperator, list: &[String], coeffs: Option<&[i32]>, minimize: bool) -> Result<(), String> {
        let vars = self.scope(list)?;
        let coeffs = coeffs.map(i64s).unwrap_or_else(|| ones(vars.len()));
        self.objective_agg(t, vars, coeffs, minimize)
    }

    /// Materialise variable-coefficient products `c_i · x_i`, each an aux var, from
    /// the named term/coeff lists. `what` tags the length-mismatch error.
    fn var_coeff_products(&mut self, list: &[String], coeffs: &[String], what: &str) -> Result<Vec<VarId>, String> {
        let xs = self.scope(list)?;
        let cs = self.scope(coeffs)?;
        if xs.len() != cs.len() {
            return Err(format!("{what}: list/coeffs length mismatch"));
        }
        Ok(xs.iter().zip(&cs).map(|(&x, &c)| self.aux_for(expr::mul(vec![expr::var(c), expr::var(x)]))).collect())
    }

    /// Objective with variable coefficients: each term is a var*var product,
    /// materialised to an aux var, then aggregated with unit coefficients.
    fn objective_sum_varcoeffs(&mut self, t: XElementOperator, list: &[String], coeffs: &[String], minimize: bool) -> Result<(), String> {
        let prods = self.var_coeff_products(list, coeffs, "objective")?;
        let coeffs = ones(prods.len());
        self.objective_agg(t, prods, coeffs, minimize)
    }

    /// Objective over expression trees (each becomes an aux variable).
    fn objective_exprs(
        &mut self,
        t: XElementOperator,
        list: &[ExpressionTree],
        coeffs: Option<&[i32]>,
        minimize: bool,
    ) -> Result<(), String> {
        use XElementOperator::*;
        let coeffs = coeffs.map(i64s).unwrap_or_else(|| ones(list.len()));
        if matches!(t, Sum) {
            if coeffs.len() != list.len() {
                return Err("objective: coeffs/terms length mismatch".to_string());
            }
            let terms = list
                .iter()
                .zip(&coeffs)
                .map(|(tree, &coeff)| {
                    let term = self.tree(tree)?;
                    Ok(if coeff == 1 { term } else { expr::mul(vec![expr::int(coeff), term]) })
                })
                .collect::<Result<_, String>>()?;
            self.objective = Some(Objective::Expr(minimize, expr::add(terms)));
            return Ok(());
        }
        let vars = self.tree_vars(list)?;
        self.objective_agg(t, vars, coeffs, minimize)
    }

    /// Build the objective variable as `aggregate_t(coeffs·vars)` and record it.
    fn objective_agg(&mut self, t: XElementOperator, vars: Vec<VarId>, coeffs: Vec<i64>, minimize: bool) -> Result<(), String> {
        use XElementOperator::*;
        let aligned = coeffs.len() == vars.len();
        let obj = match t {
            Sum => {
                if !aligned {
                    return Err("objective: coeffs/terms length mismatch".to_string());
                }
                // Saturating throughout: extreme coeffs (near i32::MAX) over wide
                // domains can exceed i64, and a wrapped span would slip past the
                // guard below. Saturating fails safe into the Linear fallback.
                let (mut lo, mut hi) = (0i64, 0i64);
                for (&c, &v) in coeffs.iter().zip(&vars) {
                    let (vmin, vmax) = (self.solver.store.min(v) as i64, self.solver.store.max(v) as i64);
                    let (a, b) = if c >= 0 { (c.saturating_mul(vmin), c.saturating_mul(vmax)) } else { (c.saturating_mul(vmax), c.saturating_mul(vmin)) };
                    lo = lo.saturating_add(a);
                    hi = hi.saturating_add(b);
                }
                if hi.saturating_sub(lo) > MAX_MATERIALIZED_OBJECTIVE_SPAN {
                    self.objective = Some(Objective::Linear(minimize, coeffs, vars));
                    return Ok(());
                }
                let obj = self.solver.new_var_range(clamp(lo), clamp(hi));
                let mut cc = coeffs;
                cc.push(-1);
                let mut vv = vars;
                vv.push(obj);
                linear(&mut self.solver, &cc, &vv, Relation::Eq, 0);
                self.local.add_linear(cc, vv, Relation::Eq, 0);
                obj
            }
            Minimum | Maximum => {
                if !aligned {
                    return Err("objective: coeffs/terms length mismatch".to_string());
                }
                // Fold coefficients into aux terms when they aren't all 1.
                let terms: Vec<VarId> = if coeffs.iter().all(|&c| c == 1) {
                    vars
                } else {
                    vars.iter().zip(&coeffs).map(|(&v, &c)| self.aux_for(expr::mul(vec![expr::int(c), expr::var(v)]))).collect()
                };
                let lo = terms.iter().map(|&v| self.solver.store.min(v)).min().unwrap_or(0);
                let hi = terms.iter().map(|&v| self.solver.store.max(v)).max().unwrap_or(0);
                let obj = self.solver.new_var_range(lo, hi);
                if matches!(t, Minimum) {
                    minimum(&mut self.solver, obj, &terms);
                } else {
                    maximum(&mut self.solver, obj, &terms);
                }
                self.local.add_extremum(terms, matches!(t, Minimum), Relation::Eq, LocalRhs::Var(obj));
                obj
            }
            NValues => {
                let obj = flatten::nvalues_var(&mut self.solver, &vars);
                self.local.add_n_values(vars, Relation::Eq, LocalRhs::Var(obj));
                obj
            }
            _ => return Err("unsupported objective type (product/lex)".to_string()),
        };
        self.objective = Some(Objective::Var(minimize, obj));
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
    let root = *all.iter().find(|s| !has_in.contains(*s)).ok_or("mdd: no root")?;
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
        layers[la].push(MddArc { from, value: *v, to });
    }
    Ok(Mdd { layers, nodes_per_layer })
}
