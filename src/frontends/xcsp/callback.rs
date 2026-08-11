//! Bridge from the `xcsp3-rust-parser` callback interface to the semantic model.
//!
//! Each callback emits frontend-neutral declarations. Unsupported forms set
//! `error`; backend compilation happens only after parsing has completed.

use std::collections::{BTreeMap, HashMap};

use xcsp3_rust_parser::data_structs::expression_tree::xcsp3_utils::{ExpressionTree, Operator as EOp, TreeNode};
use xcsp3_rust_parser::data_structs::xrelational_operand::xcsp3_core::Operand;
use xcsp3_rust_parser::data_structs::xrelational_operator::xcsp3_core::Operator as ROp;
use xcsp3_rust_parser::objectives::xobjective_element::xcsp3_core::XElementOperator;
use xcsp3_rust_parser::xcsp_callback::XcspCallback;
use xcsp3_rust_parser::xcsp_xml::xcsp_xml_model::xcsp3_xml::InstanceType;

use crate::model::{
    Automaton, Constraint, IntDomain, IntExpr, IntGlobalConstraint, IntVarRef, Mdd, MddArc, Model as SemanticModel, ModelObject,
    ModelPackage, Objective as SemanticObjective, Relation,
};

const MAX_MATERIALIZED_OBJECTIVE_SPAN: i64 = 1_000_000;
const STAR: i32 = i32::MIN;

type Expr = IntExpr;

mod expr {
    use super::{Expr, IntExpr, IntVarRef};

    pub(super) fn int(value: i64) -> Expr {
        IntExpr::Constant(value)
    }

    pub(super) fn var(variable: IntVarRef) -> Expr {
        IntExpr::Variable(variable)
    }

    pub(super) fn neg(value: Expr) -> Expr {
        IntExpr::Neg(Box::new(value))
    }

    pub(super) fn abs(value: Expr) -> Expr {
        IntExpr::Abs(Box::new(value))
    }

    pub(super) fn add(values: Vec<Expr>) -> Expr {
        IntExpr::Add(values)
    }

    pub(super) fn sub(left: Expr, right: Expr) -> Expr {
        IntExpr::Sub(Box::new(left), Box::new(right))
    }

    pub(super) fn mul(values: Vec<Expr>) -> Expr {
        IntExpr::Mul(values)
    }

    pub(super) fn div(left: Expr, right: Expr) -> Expr {
        IntExpr::Div(Box::new(left), Box::new(right))
    }

    pub(super) fn rem(left: Expr, right: Expr) -> Expr {
        IntExpr::Mod(Box::new(left), Box::new(right))
    }

    pub(super) fn min_of(values: Vec<Expr>) -> Expr {
        IntExpr::Min(values)
    }

    pub(super) fn max_of(values: Vec<Expr>) -> Expr {
        IntExpr::Max(values)
    }

    pub(super) fn eq(left: Expr, right: Expr) -> Expr {
        IntExpr::Eq(Box::new(left), Box::new(right))
    }

    pub(super) fn ne(left: Expr, right: Expr) -> Expr {
        IntExpr::Ne(Box::new(left), Box::new(right))
    }

    pub(super) fn lt(left: Expr, right: Expr) -> Expr {
        IntExpr::Lt(Box::new(left), Box::new(right))
    }

    pub(super) fn le(left: Expr, right: Expr) -> Expr {
        IntExpr::Le(Box::new(left), Box::new(right))
    }

    pub(super) fn gt(left: Expr, right: Expr) -> Expr {
        IntExpr::Gt(Box::new(left), Box::new(right))
    }

    pub(super) fn ge(left: Expr, right: Expr) -> Expr {
        IntExpr::Ge(Box::new(left), Box::new(right))
    }

    pub(super) fn not(value: Expr) -> Expr {
        IntExpr::Not(Box::new(value))
    }

    pub(super) fn and(values: Vec<Expr>) -> Expr {
        IntExpr::And(values)
    }

    pub(super) fn or(values: Vec<Expr>) -> Expr {
        IntExpr::Or(values)
    }

    pub(super) fn imp(left: Expr, right: Expr) -> Expr {
        IntExpr::Imp(Box::new(left), Box::new(right))
    }

    pub(super) fn iff(left: Expr, right: Expr) -> Expr {
        IntExpr::Iff(Box::new(left), Box::new(right))
    }

    pub(super) fn ite(condition: Expr, then_value: Expr, else_value: Expr) -> Expr {
        IntExpr::IfThenElse(Box::new(condition), Box::new(then_value), Box::new(else_value))
    }
}

#[derive(Clone)]
struct Dfa {
    n_states: usize,
    start: usize,
    accept: Vec<usize>,
    transitions: Vec<(usize, i32, usize)>,
}

fn intension(model: &mut SemanticModel, expression: Expr) {
    model.add_constraint(Constraint::Intension(expression));
}

fn linear(model: &mut SemanticModel, coefficients: &[i64], variables: &[IntVarRef], relation: Relation, rhs: i64) {
    debug_assert_eq!(coefficients.len(), variables.len());
    model.add_constraint(Constraint::Linear {
        terms: coefficients.iter().copied().zip(variables.iter().copied()).collect(),
        relation,
        rhs,
    });
}

fn global(model: &mut SemanticModel, constraint: IntGlobalConstraint) {
    model.add_constraint(Constraint::IntegerGlobal(constraint));
}

fn all_different(model: &mut SemanticModel, variables: &[IntVarRef]) {
    global(model, IntGlobalConstraint::AllDifferent { variables: variables.to_vec(), except: Vec::new() });
}

fn all_equal(model: &mut SemanticModel, variables: &[IntVarRef]) {
    global(model, IntGlobalConstraint::AllEqual(variables.to_vec()));
}

fn element(model: &mut SemanticModel, array: &[IntVarRef], index: IntVarRef, value: IntVarRef) {
    global(model, IntGlobalConstraint::Element { array: array.to_vec(), index, value });
}

fn instantiation(model: &mut SemanticModel, variables: &[IntVarRef], values: &[i32]) {
    global(model, IntGlobalConstraint::Instantiation { variables: variables.to_vec(), values: values.to_vec() });
}

fn minimum(model: &mut SemanticModel, target: IntVarRef, variables: &[IntVarRef]) {
    global(model, IntGlobalConstraint::Minimum { target, variables: variables.to_vec() });
}

fn maximum(model: &mut SemanticModel, target: IntVarRef, variables: &[IntVarRef]) {
    global(model, IntGlobalConstraint::Maximum { target, variables: variables.to_vec() });
}

fn ordered(model: &mut SemanticModel, variables: &[IntVarRef], relation: Relation) {
    global(model, IntGlobalConstraint::Ordered { variables: variables.to_vec(), relation });
}

fn count(model: &mut SemanticModel, variables: &[IntVarRef], value: i32, relation: Relation, count: i64) {
    global(model, IntGlobalConstraint::Count { variables: variables.to_vec(), value, relation, count });
}

fn n_values(model: &mut SemanticModel, variables: &[IntVarRef], relation: Relation, count: i64) {
    global(model, IntGlobalConstraint::NValues { variables: variables.to_vec(), relation, count });
}

fn cardinality(model: &mut SemanticModel, variables: &[IntVarRef], values: &[i32], lower: &[i64], upper: &[i64], closed: bool) {
    global(
        model,
        IntGlobalConstraint::Cardinality {
            variables: variables.to_vec(),
            values: values.to_vec(),
            lower: lower.to_vec(),
            upper: upper.to_vec(),
            closed,
        },
    );
}

fn lex_chain(model: &mut SemanticModel, rows: &[Vec<IntVarRef>], strict: bool) {
    global(model, IntGlobalConstraint::LexChain { rows: rows.to_vec(), strict });
}

fn channel(model: &mut SemanticModel, left: &[IntVarRef], right: &[IntVarRef]) {
    global(model, IntGlobalConstraint::Channel { left: left.to_vec(), right: right.to_vec() });
}

fn circuit(model: &mut SemanticModel, successors: &[IntVarRef]) {
    global(model, IntGlobalConstraint::Circuit { successors: successors.to_vec(), cutset: false });
}

fn bin_packing(model: &mut SemanticModel, items: &[IntVarRef], sizes: &[i64], capacities: &[i64]) {
    global(model, IntGlobalConstraint::BinPacking { items: items.to_vec(), sizes: sizes.to_vec(), capacities: capacities.to_vec() });
}

fn cumulative(model: &mut SemanticModel, starts: &[IntVarRef], durations: &[i64], demands: &[i64], capacity: i64) {
    global(
        model,
        IntGlobalConstraint::Cumulative { starts: starts.to_vec(), durations: durations.to_vec(), demands: demands.to_vec(), capacity },
    );
}

fn cumulative_var(model: &mut SemanticModel, starts: &[IntVarRef], durations: &[IntVarRef], demands: &[IntVarRef], capacity: IntVarRef) {
    global(
        model,
        IntGlobalConstraint::CumulativeVar { starts: starts.to_vec(), durations: durations.to_vec(), demands: demands.to_vec(), capacity },
    );
}

fn no_overlap(model: &mut SemanticModel, starts: &[IntVarRef], durations: &[i64]) {
    global(model, IntGlobalConstraint::NoOverlap { starts: starts.to_vec(), durations: durations.to_vec() });
}

fn extension(model: &mut SemanticModel, variables: &[IntVarRef], tuples: &[Vec<i32>], positive: bool) {
    global(model, IntGlobalConstraint::Table { variables: variables.to_vec(), tuples: tuples.to_vec(), positive });
}

fn regular(model: &mut SemanticModel, variables: &[IntVarRef], dfa: Dfa) {
    global(
        model,
        IntGlobalConstraint::Regular {
            variables: variables.to_vec(),
            automaton: Automaton { states: dfa.n_states, start: dfa.start, accepting: dfa.accept, transitions: dfa.transitions },
        },
    );
}

fn mdd(model: &mut SemanticModel, variables: &[IntVarRef], mdd: Mdd) {
    global(model, IntGlobalConstraint::Mdd { variables: variables.to_vec(), mdd });
}

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

fn weighted_sum_expr(coeffs: &[i64], vars: &[IntVarRef]) -> Expr {
    let mut combined = BTreeMap::<IntVarRef, i128>::new();
    let mut normalized = true;
    for (&coefficient, &variable) in coeffs.iter().zip(vars) {
        let Some(sum) = combined.get(&variable).copied().unwrap_or(0).checked_add(i128::from(coefficient)) else {
            normalized = false;
            break;
        };
        if sum == 0 {
            combined.remove(&variable);
        } else {
            combined.insert(variable, sum);
        }
    }
    let normalized_terms = normalized.then(|| {
        combined
            .into_iter()
            .map(|(variable, coefficient)| i64::try_from(coefficient).ok().map(|coefficient| (coefficient, variable)))
            .collect::<Option<Vec<_>>>()
    });
    let terms_and_variables = normalized_terms.flatten().unwrap_or_else(|| coeffs.iter().copied().zip(vars.iter().copied()).collect());

    let mut terms = Vec::with_capacity(terms_and_variables.len());
    for (coeff, var) in terms_and_variables {
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
    cells: HashMap<Vec<usize>, IntVarRef>,
}

/// Accumulates the model as the parser walks the instance.
pub struct Model {
    pub(super) package: ModelPackage,
    pub declared: Vec<(String, IntVarRef)>,
    pub(super) objective: Option<SemanticObjective>,
    pub error: Option<String>,
    /// Scalar variables by name. Array cells use compact row-major storage.
    ids: HashMap<String, IntVarRef>,
    arrays: HashMap<String, ArrayDecl>,
    constants: HashMap<i32, IntVarRef>,
}

/// Right-hand side of a condition.
enum Rhs {
    Const(i64),
    Var(IntVarRef),
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
            package: ModelPackage::new(SemanticModel::new()),
            declared: Vec::new(),
            objective: None,
            error: None,
            ids: HashMap::new(),
            arrays: HashMap::new(),
            constants: HashMap::new(),
        }
    }

    pub(super) fn into_package(mut self) -> ModelPackage {
        if let Some(objective) = self.objective.take() {
            self.package.model.add_objective(objective);
        }
        self.package
    }

    pub(super) fn num_variables(&self) -> usize {
        self.package.model.int_vars().len()
    }

    pub(super) fn num_sparse_domains(&self) -> usize {
        self.package.model.int_vars().iter().filter(|domain| matches!(domain, IntDomain::Set(_))).count()
    }

    pub(super) fn num_bounds_domains(&self) -> usize {
        self.package.model.int_vars().iter().filter(|domain| !matches!(domain, IntDomain::Set(_))).count()
    }

    pub(super) fn num_constraints(&self) -> usize {
        self.package.model.constraints().len()
    }

    pub(super) fn has_objective(&self) -> bool {
        self.objective.is_some()
    }

    pub(super) fn relevant_variables(&self) -> Vec<bool> {
        let mut relevant = vec![false; self.num_variables()];
        let mut variables = Vec::new();
        for constraint in self.package.model.constraints() {
            match constraint {
                Constraint::Intension(expression) => expression.variables(&mut variables),
                Constraint::Linear { terms, .. } => variables.extend(terms.iter().map(|(_, variable)| *variable)),
                Constraint::Clause(literals) => variables.extend(literals.iter().map(|literal| literal.variable)),
                Constraint::IntegerGlobal(global) => global.variables(&mut variables),
                _ => {}
            }
        }
        if let Some(SemanticObjective::IntExpr { expr, .. }) = &self.objective {
            expr.variables(&mut variables);
        }
        for variable in variables {
            relevant[variable.0] = true;
        }
        relevant
    }

    fn domain(&self, variable: IntVarRef) -> &IntDomain {
        &self.package.model.int_vars()[variable.0]
    }

    fn min(&self, variable: IntVarRef) -> i32 {
        match self.domain(variable) {
            IntDomain::Bool => 0,
            IntDomain::Range { lo, .. } => *lo,
            IntDomain::Set(values) => values.iter().copied().min().expect("validated non-empty integer domain"),
        }
    }

    fn max(&self, variable: IntVarRef) -> i32 {
        match self.domain(variable) {
            IntDomain::Bool => 1,
            IntDomain::Range { hi, .. } => *hi,
            IntDomain::Set(values) => values.iter().copied().max().expect("validated non-empty integer domain"),
        }
    }

    fn values(&self, variable: IntVarRef) -> Vec<i32> {
        match self.domain(variable) {
            IntDomain::Bool => vec![0, 1],
            IntDomain::Range { lo, hi } => (*lo..=*hi).collect(),
            IntDomain::Set(values) => values.clone(),
        }
    }

    fn fail(&mut self, msg: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(msg.into());
        }
    }

    fn var_id(&self, name: &str) -> Result<IntVarRef, String> {
        if let Some((base, indices)) = cell_ref(name) {
            let array = self.arrays.get(base).ok_or_else(|| format!("unknown variable `{name}`"))?;
            return array.cells.get(&indices).copied().ok_or_else(|| format!("unknown variable `{name}`"));
        }
        self.ids.get(name).copied().ok_or_else(|| format!("unknown variable `{name}`"))
    }

    fn scope(&self, list: &[String]) -> Result<Vec<IntVarRef>, String> {
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
    fn expand_array_ref(&self, s: &str) -> Option<Vec<IntVarRef>> {
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
    fn expand_array_slice(&self, name: &str) -> Option<Vec<IntVarRef>> {
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

    fn remember_var(&mut self, id: String, var: IntVarRef) {
        let object = ModelObject::IntVar(var);
        self.package.metadata.names.insert(object, id.clone());
        self.package.metadata.frontend_ids.insert(("xcsp".to_string(), id.clone()), object);
        self.package.metadata.outputs.push(object);
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
    fn constant(&mut self, value: i32) -> IntVarRef {
        if let Some(&variable) = self.constants.get(&value) {
            return variable;
        }
        let variable = self.package.model.int_set(vec![value]);
        self.constants.insert(value, variable);
        variable
    }

    /// Fresh fixed variables, one per integer (for slots that take variables).
    fn consts(&mut self, values: &[i32]) -> Vec<IntVarRef> {
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

    fn rhs_var(&mut self, operand: &Operand) -> Result<IntVarRef, String> {
        Ok(match self.rhs(operand)? {
            Rhs::Const(k) => self.constant(clamp(k)),
            Rhs::Var(v) => v,
        })
    }

    fn var_or_constant(&mut self, s: &str) -> Result<IntVarRef, String> {
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
    fn post_sum(&mut self, coeffs: Vec<i64>, vars: Vec<IntVarRef>, operator: ROp, operand: Operand) -> Result<(), String> {
        match (operator, operand) {
            (ROp::In, Operand::SetInteger(set)) => {
                let allowed = sorted_values(set)?;
                let y = self.package.model.int_set(allowed);
                self.post_linear(coeffs, vars, Relation::Eq, Rhs::Var(y))?;
            }
            (ROp::Notin, Operand::Interval(lo, hi)) => {
                // Σ ∉ [lo,hi] is a disjunction. Keep it symbolic instead of
                // materialising the complement, which can be empty or huge.
                let sum = weighted_sum_expr(&coeffs, &vars);
                let e = expr::or(vec![expr::lt(sum.clone(), expr::int(lo as i64)), expr::gt(sum, expr::int(hi as i64))]);
                intension(&mut self.package.model, e);
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

    fn post_count_condition(&mut self, vars: &[IntVarRef], values: &[i32], operator: ROp, operand: Operand) -> Result<(), String> {
        match (operator, operand) {
            (ROp::In, Operand::SetInteger(set)) => {
                let allowed = sorted_values(set)?;
                let y = self.package.model.int_set(allowed);
                self.post_count(vars, values, Relation::Eq, Rhs::Var(y));
            }
            (operator, operand) => {
                for (rel, rhs) in self.conditions(operator, operand)? {
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
        lex_chain(&mut self.package.model, &rows, strict);
        Ok(())
    }

    /// Post `Σ coeffs·vars  rel  rhs` (moving a variable rhs to the lhs).
    fn post_linear(&mut self, mut coeffs: Vec<i64>, mut vars: Vec<IntVarRef>, rel: Relation, rhs: Rhs) -> Result<(), String> {
        if coeffs.len() != vars.len() {
            return Err("linear: coeffs/vars length mismatch".to_string());
        }
        match rhs {
            Rhs::Const(k) => {
                linear(&mut self.package.model, &coeffs, &vars, rel, k);
            }
            Rhs::Var(y) => {
                coeffs.push(-1);
                vars.push(y);
                linear(&mut self.package.model, &coeffs, &vars, rel, 0);
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
    fn aux_for(&mut self, e: Expr) -> IntVarRef {
        let (lo, hi) = self.expr_bounds(&e);
        let aux = self.package.model.int_range(clamp(lo), clamp(hi));
        intension(&mut self.package.model, expr::eq(expr::var(aux), e));
        aux
    }

    fn expr_bounds(&self, expression: &Expr) -> (i64, i64) {
        let add = |left: (i64, i64), right: (i64, i64)| (left.0.saturating_add(right.0), left.1.saturating_add(right.1));
        match expression {
            Expr::Constant(value) => (*value, *value),
            Expr::Variable(variable) => (i64::from(self.min(*variable)), i64::from(self.max(*variable))),
            Expr::Neg(value) => {
                let (lo, hi) = self.expr_bounds(value);
                (hi.saturating_neg(), lo.saturating_neg())
            }
            Expr::Abs(value) => {
                let (lo, hi) = self.expr_bounds(value);
                let upper = lo.saturating_abs().max(hi.saturating_abs());
                (if lo <= 0 && hi >= 0 { 0 } else { lo.saturating_abs().min(hi.saturating_abs()) }, upper)
            }
            Expr::Add(values) => values.iter().map(|value| self.expr_bounds(value)).fold((0, 0), add),
            Expr::Sub(left, right) => {
                let (llo, lhi) = self.expr_bounds(left);
                let (rlo, rhi) = self.expr_bounds(right);
                (llo.saturating_sub(rhi), lhi.saturating_sub(rlo))
            }
            Expr::Mul(values) => values.iter().map(|value| self.expr_bounds(value)).fold((1, 1), |(alo, ahi), (blo, bhi)| {
                let endpoints = [alo.saturating_mul(blo), alo.saturating_mul(bhi), ahi.saturating_mul(blo), ahi.saturating_mul(bhi)];
                (*endpoints.iter().min().unwrap(), *endpoints.iter().max().unwrap())
            }),
            Expr::Min(values) => {
                let bounds = values.iter().map(|value| self.expr_bounds(value)).collect::<Vec<_>>();
                (
                    bounds.iter().map(|bound| bound.0).min().unwrap_or(i32::MIN as i64),
                    bounds.iter().map(|bound| bound.1).min().unwrap_or(i32::MAX as i64),
                )
            }
            Expr::Max(values) => {
                let bounds = values.iter().map(|value| self.expr_bounds(value)).collect::<Vec<_>>();
                (
                    bounds.iter().map(|bound| bound.0).max().unwrap_or(i32::MIN as i64),
                    bounds.iter().map(|bound| bound.1).max().unwrap_or(i32::MAX as i64),
                )
            }
            Expr::Div(_, _) | Expr::Mod(_, _) => (i32::MIN as i64, i32::MAX as i64),
            Expr::IfThenElse(_, then_value, else_value) => {
                let then_bounds = self.expr_bounds(then_value);
                let else_bounds = self.expr_bounds(else_value);
                (then_bounds.0.min(else_bounds.0), then_bounds.1.max(else_bounds.1))
            }
            Expr::Eq(_, _)
            | Expr::Ne(_, _)
            | Expr::Lt(_, _)
            | Expr::Le(_, _)
            | Expr::Gt(_, _)
            | Expr::Ge(_, _)
            | Expr::Not(_)
            | Expr::And(_)
            | Expr::Or(_)
            | Expr::Imp(_, _)
            | Expr::Iff(_, _) => (0, 1),
        }
    }

    /// Turn a list of parser expression trees into solver variables (aux vars
    /// for non-trivial expressions).
    fn tree_vars(&mut self, list: &[ExpressionTree]) -> Result<Vec<IntVarRef>, String> {
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
                Expr::Variable(v) => out.push(v),
                other => out.push(self.aux_for(other)),
            }
        }
        Ok(out)
    }
}

fn clamp(x: i64) -> i32 {
    x.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
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
                Expr::Constant(0) => expr::int(1),
                Expr::Constant(k) if (1..=8).contains(&k) => expr::mul(vec![a; k as usize]),
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

    fn begin_group(&mut self) {}

    fn end_group(&mut self) {}

    fn begin_slide(&mut self) {
        self.begin_group();
    }

    fn end_slide(&mut self) {
        self.end_group();
    }

    // --- variables ---
    fn on_variable_interval(&mut self, id: String, min: i32, max: i32) {
        let v = self.package.model.int_range(min, max);
        self.declared.push((id.clone(), v));
        self.remember_var(id, v);
    }
    fn on_variable_values(&mut self, id: String, values: &[i32]) {
        let v = self.package.model.int_set(values.to_vec());
        self.declared.push((id.clone(), v));
        self.remember_var(id, v);
    }

    // --- constraints ---
    fn on_constraint_extension(&mut self, list: &[String], tuples: &Vec<Vec<i32>>, is_support: bool, _has_star: bool) {
        guard!(self, {
            let vars = self.scope(list)?;
            let tuples = tuples.iter().map(|t| t.iter().map(|&v| if v == i32::MAX { STAR } else { v }).collect()).collect::<Vec<Vec<_>>>();
            extension(&mut self.package.model, &vars, &tuples, is_support);
            Ok(())
        });
    }

    fn on_constraint_unary(&mut self, list: &String, values: &[i32], is_support: bool) {
        guard!(self, {
            let v = self.var_id(list)?;
            let tuples = values.iter().map(|&x| vec![x]).collect::<Vec<_>>();
            extension(&mut self.package.model, &[v], &tuples, is_support);
            Ok(())
        });
    }

    fn on_constraint_intention(&mut self, _list: &[String], tree: &ExpressionTree) {
        guard!(self, {
            let e = self.tree(tree)?;
            intension(&mut self.package.model, e);
            Ok(())
        });
    }

    fn on_constraint_all_different_v1(&mut self, list: &[String]) {
        guard!(self, {
            let vars = self.scope(list)?;
            all_different(&mut self.package.model, &vars);
            Ok(())
        });
    }

    fn on_constraint_all_different_v2(&mut self, list: &[ExpressionTree]) {
        guard!(self, {
            let vars = self.tree_vars(list)?;
            all_different(&mut self.package.model, &vars);
            Ok(())
        });
    }

    fn on_constraint_all_different_list(&mut self, lists: &[Vec<String>]) {
        guard!(self, {
            if lists.len() == 1 {
                let vars = self.scope(&lists[0])?;
                all_different(&mut self.package.model, &vars);
            } else {
                // allDifferent on lists: the tuples must be pairwise distinct,
                // i.e. each pair differs in at least one position.
                let tuples = lists.iter().map(|l| self.scope(l)).collect::<Result<Vec<_>, _>>()?;
                for i in 0..tuples.len() {
                    for j in (i + 1)..tuples.len() {
                        require(tuples[i].len() == tuples[j].len(), "allDifferent lists: tuple length mismatch")?;
                        let differences =
                            tuples[i].iter().zip(&tuples[j]).map(|(&left, &right)| expr::ne(expr::var(left), expr::var(right))).collect();
                        intension(&mut self.package.model, expr::or(differences));
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
                all_different(&mut self.package.model, &vars);
                matrix.push(vars);
            }
            let columns = (0..cols).map(|c| (0..rows).map(|r| matrix[r][c]).collect::<Vec<_>>()).collect::<Vec<_>>();
            for col in columns {
                all_different(&mut self.package.model, &col);
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
            global(&mut self.package.model, IntGlobalConstraint::AllDifferent { variables: vars, except: except.to_vec() });
            Ok(())
        });
    }

    fn on_constraint_all_equal_v1(&mut self, list: &[String]) {
        guard!(self, {
            let vars = self.scope(list)?;
            all_equal(&mut self.package.model, &vars);
            Ok(())
        });
    }

    fn on_constraint_ordered_v1(&mut self, list: &[String], operator: ROp) {
        guard!(self, {
            let vars = self.scope(list)?;
            let rel = Model::rel(operator)?;
            ordered(&mut self.package.model, &vars, rel);
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
                linear(&mut self.package.model, &[1, 1, -1], &[vars[i], lens[i], vars[i + 1]], rel, 0);
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
                linear(&mut self.package.model, &[1, -1], &[vars[i], vars[i + 1]], rel, rhs);
            }
            Ok(())
        });
    }

    fn on_constraint_instantiation(&mut self, list: &[String], values: &[i32]) {
        guard!(self, {
            let vars = self.scope(list)?;
            instantiation(&mut self.package.model, &vars, values);
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
            let inds: Vec<IntVarRef> = vars
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
            let n = array.len() as i32;
            let idx = self.package.model.int_range(0, n - 1);
            let val = self.constant(value);
            element(&mut self.package.model, &array, idx, val);
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
                    let h = i64s(heights);
                    cumulative(&mut self.package.model, &starts, &d, &h, k);
                }
                // Variable capacity: the weaker variable-resource propagator.
                Rhs::Var(cap) => {
                    let dv = self.consts(lengths);
                    let h = self.consts(heights);
                    cumulative_var(&mut self.package.model, &starts, &dv, &h, cap);
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
            if start_index == 0 {
                channel(&mut self.package.model, &vars, &vars);
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
            linear(&mut self.package.model, &[1], &[v], Relation::Ge, i64::from(start_index));
            linear(&mut self.package.model, &[1], &[v], Relation::Le, i64::from(start_index) + xs.len() as i64 - 1);
            for (index, &indicator) in xs.iter().enumerate() {
                let active = expr::eq(expr::var(indicator), expr::int(1));
                let selected = expr::eq(expr::var(v), expr::int(i64::from(start_index) + index as i64));
                intension(&mut self.package.model, expr::iff(active, selected));
            }
            Ok(())
        });
    }

    fn on_constraint_precedence_v1(&mut self, list: &[String], covered: bool) {
        guard!(self, {
            // Default value order: the sorted distinct values across all domains.
            let vars = self.scope(list)?;
            let mut vals: Vec<i32> = Vec::new();
            for &v in &vars {
                vals.extend(self.values(v));
            }
            vals.sort_unstable();
            vals.dedup();
            global(&mut self.package.model, IntGlobalConstraint::ValuePrecedence { variables: vars, values: vals, covered });
            Ok(())
        });
    }
    fn on_constraint_precedence_v2(&mut self, list: &[String], values: &[i32], covered: bool) {
        guard!(self, {
            let vars = self.scope(list)?;
            global(&mut self.package.model, IntGlobalConstraint::ValuePrecedence { variables: vars, values: values.to_vec(), covered });
            Ok(())
        });
    }

    fn on_constraint_circuit_v1(&mut self, list: &Vec<String>) {
        guard!(self, {
            let vars = self.scope(list)?;
            circuit(&mut self.package.model, &vars);
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
            cardinality(&mut self.package.model, &vars, values, &low, &low, closed);
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
                self.count_values_to_var(&vars, &[value], Relation::Eq, target);
            }
            if closed {
                let low = vec![0; values.len()];
                let high = vec![vars.len() as i64; values.len()];
                cardinality(&mut self.package.model, &vars, values, &low, &high, true);
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
            cardinality(&mut self.package.model, &vars, values, &low, &high, closed);
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
            let nbins = items.iter().map(|&item| self.max(item) + 1).max().unwrap_or(0).max(1) as usize;
            bin_packing(&mut self.package.model, &items, &s, &vec![cap; nbins]);
            Ok(())
        });
    }
    fn on_constraint_bin_packing_v2(&mut self, list: &[String], sizes: &[i32], limits: &[i32]) {
        guard!(self, {
            let items = self.scope(list)?;
            let s = i64s(sizes);
            let caps = i64s(limits);
            bin_packing(&mut self.package.model, &items, &s, &caps);
            Ok(())
        });
    }
    fn on_constraint_bin_packing_v3(&mut self, list: &[String], sizes: &[i32], limits: &[String]) {
        guard!(self, {
            // Variable capacities: load_b = Σ_i size_i·[item_i==b]  must be ≤ limit_b.
            let items = self.scope(list)?;
            let lim = self.scope(limits)?;
            let s = i64s(sizes);
            let total: i64 = s.iter().sum();
            let loads: Vec<IntVarRef> = (0..lim.len()).map(|_| self.package.model.int_range(0, clamp(total))).collect();
            global(&mut self.package.model, IntGlobalConstraint::BinLoads { items, sizes: s, loads: loads.clone() });
            for (&load, &cap) in loads.iter().zip(&lim) {
                linear(&mut self.package.model, &[1, -1], &[load, cap], Relation::Le, 0);
            }
            Ok(())
        });
    }
    fn on_constraint_bin_packing_v4(&mut self, list: &[String], sizes: &[i32], loads: &[i32]) {
        guard!(self, {
            // Each bin's load is fixed to the given constant loads[b].
            let items = self.scope(list)?;
            let s = i64s(sizes);
            let loadv: Vec<IntVarRef> = loads.iter().map(|&k| self.package.model.int_range(k, k)).collect();
            global(&mut self.package.model, IntGlobalConstraint::BinLoads { items, sizes: s, loads: loadv });
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
            global(&mut self.package.model, IntGlobalConstraint::BinLoads { items, sizes: s, loads: loadv });
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
            // With strictly positive fixed durations the zeroIgnored flag is
            // semantically inert. Keep the compact one-dimensional global in
            // that common case instead of expanding every pair into a generic
            // intension constraint.
            if lengths.iter().all(|&length| length > 0) {
                no_overlap(&mut self.package.model, &starts, &d);
            } else {
                self.post_diffn(origins, len, zero_ignored)?;
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
            regular(&mut self.package.model, &vars, dfa);
            Ok(())
        });
    }

    fn on_constraint_mdd(&mut self, list: &[String], transitions: &Vec<(String, i32, String)>) {
        guard!(self, {
            let vars = self.scope(list)?;
            // Layer per variable, inferred from a root reachable BFS over labels.
            let m = build_mdd(vars.len(), transitions)?;
            mdd(&mut self.package.model, &vars, m);
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
    fn min_max_vars(&mut self, xs: Vec<IntVarRef>, operator: ROp, operand: Operand, is_min: bool) -> Result<(), String> {
        let rel = Model::rel(operator)?;
        let y = self.rhs_var(&operand)?;
        if matches!(rel, Relation::Eq) {
            if is_min {
                minimum(&mut self.package.model, y, &xs);
            } else {
                maximum(&mut self.package.model, y, &xs);
            }
        } else {
            // aux = min/max(xs); then aux rel y.
            let (lo, hi) = (xs.iter().map(|&v| self.min(v)).min().unwrap_or(0), xs.iter().map(|&v| self.max(v)).max().unwrap_or(0));
            let aux = self.package.model.int_range(lo, hi);
            if is_min {
                minimum(&mut self.package.model, aux, &xs);
            } else {
                maximum(&mut self.package.model, aux, &xs);
            }
            linear(&mut self.package.model, &[1, -1], &[aux, y], rel, 0);
        }
        Ok(())
    }

    /// `nValues` over already-resolved variables: count distinct, then `rel rhs`.
    fn post_nvalues(&mut self, vars: Vec<IntVarRef>, operator: ROp, operand: Operand) -> Result<(), String> {
        // Decode through the generic condition handler so interval/`notin`
        // conditions expand to the right set of relations.
        for (rel, rhs) in self.conditions(operator, operand)? {
            match rhs {
                Rhs::Const(k) => n_values(&mut self.package.model, &vars, rel, k),
                Rhs::Var(y) => {
                    let aux = self.nvalues_var(&vars);
                    linear(&mut self.package.model, &[1, -1], &[aux, y], rel, 0);
                }
            }
        }
        Ok(())
    }

    /// Post a count condition `#{i : vars[i] ∈ values}  rel  rhs`, choosing the
    /// dedicated bounds propagator for the single-value/constant case.
    fn post_count(&mut self, vars: &[IntVarRef], values: &[i32], rel: Relation, rhs: Rhs) {
        match rhs {
            Rhs::Const(k) if values.len() == 1 => count(&mut self.package.model, vars, values[0], rel, k),
            Rhs::Const(k) => {
                let y = self.constant(clamp(k));
                self.count_values_to_var(vars, values, rel, y);
            }
            Rhs::Var(y) => self.count_values_to_var(vars, values, rel, y),
        }
    }

    fn count_values_to_var(&mut self, vars: &[IntVarRef], values: &[i32], rel: Relation, target: IntVarRef) {
        let mut terms = Vec::with_capacity(vars.len() + 1);
        for &variable in vars {
            let membership = match values {
                [] => expr::int(0),
                [value] => expr::eq(expr::var(variable), expr::int(i64::from(*value))),
                _ => expr::or(values.iter().map(|&value| expr::eq(expr::var(variable), expr::int(i64::from(value)))).collect()),
            };
            terms.push(self.aux_for(membership));
        }
        let mut coefficients = vec![1; terms.len()];
        coefficients.push(-1);
        terms.push(target);
        linear(&mut self.package.model, &coefficients, &terms, rel, 0);
    }

    fn nvalues_var(&mut self, vars: &[IntVarRef]) -> IntVarRef {
        let mut values = vars.iter().flat_map(|&variable| self.values(variable)).collect::<Vec<_>>();
        values.sort_unstable();
        values.dedup();
        let mut present = Vec::with_capacity(values.len() + 1);
        for value in values {
            let expression = expr::or(vars.iter().map(|&variable| expr::eq(expr::var(variable), expr::int(i64::from(value)))).collect());
            present.push(self.aux_for(expression));
        }
        let count = present.len();
        let aux = self.package.model.int_range(if vars.is_empty() { 0 } else { 1 }, count as i32);
        let mut coefficients = vec![1; count];
        coefficients.push(-1);
        present.push(aux);
        linear(&mut self.package.model, &coefficients, &present, Relation::Eq, 0);
        aux
    }

    fn origins_to_ids(&self, origins: &[Vec<String>]) -> Result<Vec<Vec<IntVarRef>>, String> {
        origins.iter().map(|b| b.iter().map(|s| self.var_id(s)).collect()).collect()
    }

    /// Offset-aware kernel encoding of `channel(xs, ys)`: value bounds plus the
    /// pairwise iff decomposition. Used whenever a start index shifts positions
    /// out of the plain 0-based `channel` propagator's frame.
    fn post_channel_inverse_kernel(&mut self, xs: &[IntVarRef], start_index1: i32, ys: &[IntVarRef], start_index2: i32) {
        for &x in xs {
            linear(&mut self.package.model, &[1], &[x], Relation::Ge, start_index2 as i64);
            linear(&mut self.package.model, &[1], &[x], Relation::Le, start_index2 as i64 + ys.len() as i64 - 1);
        }
        for &y in ys {
            linear(&mut self.package.model, &[1], &[y], Relation::Ge, start_index1 as i64);
            linear(&mut self.package.model, &[1], &[y], Relation::Le, start_index1 as i64 + xs.len() as i64 - 1);
        }
        for (i, &x) in xs.iter().enumerate() {
            for (j, &y) in ys.iter().enumerate() {
                let xv = expr::eq(expr::var(x), expr::int(start_index2 as i64 + j as i64));
                let yv = expr::eq(expr::var(y), expr::int(start_index1 as i64 + i as i64));
                intension(&mut self.package.model, expr::iff(xv, yv));
            }
        }
    }

    fn post_diffn(&mut self, origins: Vec<Vec<IntVarRef>>, lengths: Vec<Vec<Expr>>, zero_ignored: bool) -> Result<(), String> {
        require(origins.len() == lengths.len(), "noOverlap: origins/lengths length mismatch")?;
        for (origin, length) in origins.iter().zip(&lengths) {
            require(origin.len() == length.len(), "noOverlap: box origin/length arity mismatch")?;
        }
        for i in 0..origins.len() {
            for j in (i + 1)..origins.len() {
                let mut separated = Vec::new();
                for dimension in 0..origins[i].len() {
                    separated.push(expr::le(
                        expr::add(vec![expr::var(origins[i][dimension]), lengths[i][dimension].clone()]),
                        expr::var(origins[j][dimension]),
                    ));
                    separated.push(expr::le(
                        expr::add(vec![expr::var(origins[j][dimension]), lengths[j][dimension].clone()]),
                        expr::var(origins[i][dimension]),
                    ));
                }
                if zero_ignored {
                    separated.extend(lengths[i].iter().chain(&lengths[j]).map(|length| expr::eq(length.clone(), expr::int(0))));
                }
                intension(&mut self.package.model, expr::or(separated));
            }
        }
        Ok(())
    }

    /// Fold a `startIndex` offset into the index variable, returning a fresh
    /// 0-based index var constrained to `idx - start_index`, or `idx` unchanged
    /// when the list is already 0-based. Same technique as `matrix_index`.
    fn zero_based_index(&mut self, idx: IntVarRef, start_index: i32, len: usize) -> IntVarRef {
        if start_index == 0 {
            return idx;
        }
        let idx0 = self.package.model.int_range(0, len as i32 - 1);
        let offset = i64::from(start_index);
        linear(&mut self.package.model, &[1, -1], &[idx0, idx], Relation::Eq, -offset);
        idx0
    }

    fn post_element(&mut self, array: Vec<IntVarRef>, start_index: i32, index: &str, value: IntVarRef) -> Result<(), String> {
        let idx = self.var_id(index)?;
        let idx = self.zero_based_index(idx, start_index, array.len());
        element(&mut self.package.model, &array, idx, value);
        Ok(())
    }

    fn post_element_cond(
        &mut self,
        array: Vec<IntVarRef>,
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

    fn matrix_index(&mut self, access: MatrixAccess<'_>) -> Result<IntVarRef, String> {
        let len = access.rows.checked_mul(access.cols).ok_or_else(|| "element matrix: too large".to_string())?;
        require(len <= i32::MAX as usize, "element matrix: too large")?;
        let row = self.var_or_constant(access.row_index)?;
        let col = self.var_or_constant(access.col_index)?;
        let idx = self.package.model.int_range(0, len as i32 - 1);
        let offset = access.cols as i64 * i64::from(access.start_row_index) + i64::from(access.start_col_index);
        linear(&mut self.package.model, &[1, -(access.cols as i64), -1], &[idx, row, col], Relation::Eq, -offset);
        Ok(idx)
    }

    fn post_element_matrix(&mut self, array: Vec<IntVarRef>, access: MatrixAccess<'_>, value: IntVarRef) -> Result<(), String> {
        let idx = self.matrix_index(access)?;
        element(&mut self.package.model, &array, idx, value);
        Ok(())
    }

    fn post_cumulative_var(
        &mut self,
        origins: &[String],
        durations: Vec<IntVarRef>,
        heights: Vec<IntVarRef>,
        operator: ROp,
        operand: &Operand,
    ) -> Result<(), String> {
        require(matches!(operator, ROp::Le), "cumulative condition must be <=")?;
        let starts = self.scope(origins)?;
        let cap = self.rhs_var(operand)?;
        cumulative_var(&mut self.package.model, &starts, &durations, &heights, cap);
        Ok(())
    }

    /// `aux = array[index]` then `aux  rel  operand`.
    fn element_cond(&mut self, array: Vec<IntVarRef>, idx: IntVarRef, operator: ROp, operand: Operand) -> Result<(), String> {
        let lo = array.iter().map(|&v| self.min(v)).min().unwrap_or(0);
        let hi = array.iter().map(|&v| self.max(v)).max().unwrap_or(0);
        let aux = self.package.model.int_range(lo, hi);
        element(&mut self.package.model, &array, idx, aux);
        let rel = Model::rel(operator)?;
        let y = self.rhs_var(&operand)?;
        linear(&mut self.package.model, &[1, -1], &[aux, y], rel, 0);
        Ok(())
    }

    fn set_var_objective(&mut self, var: &str, minimize: bool) -> Result<(), String> {
        self.objective = Some(SemanticObjective::IntExpr { minimize, expr: expr::var(self.var_id(var)?) });
        Ok(())
    }

    fn set_expr_objective(&mut self, tree: &ExpressionTree, minimize: bool) -> Result<(), String> {
        let expression = self.tree(tree)?;
        let expression = match expression {
            expression @ Expr::Variable(_) => expression,
            expression => {
                let (lo, hi) = self.expr_bounds(&expression);
                if i32::try_from(lo).is_ok() && i32::try_from(hi).is_ok() {
                    expr::var(self.aux_for(expression))
                } else {
                    expression
                }
            }
        };
        self.objective = Some(SemanticObjective::IntExpr { minimize, expr: expression });
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
    fn var_coeff_products(&mut self, list: &[String], coeffs: &[String], what: &str) -> Result<Vec<IntVarRef>, String> {
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
            self.objective = Some(SemanticObjective::IntExpr { minimize, expr: expr::add(terms) });
            return Ok(());
        }
        let vars = self.tree_vars(list)?;
        self.objective_agg(t, vars, coeffs, minimize)
    }

    /// Build the objective variable as `aggregate_t(coeffs·vars)` and record it.
    fn objective_agg(&mut self, t: XElementOperator, vars: Vec<IntVarRef>, coeffs: Vec<i64>, minimize: bool) -> Result<(), String> {
        use XElementOperator::*;
        let aligned = coeffs.len() == vars.len();
        let obj = match t {
            Sum => {
                if !aligned {
                    return Err("objective: coeffs/terms length mismatch".to_string());
                }
                // Materialize every affine objective whose complete range fits
                // comfortably in one integer variable. The CP compiler can
                // recover the defining equality as an affine search view, while
                // bounds continue to use the strongly propagated objective
                // variable.
                let mut bounds = Some((0i128, 0i128));
                for (&coefficient, &variable) in coeffs.iter().zip(&vars) {
                    let (minimum, maximum) = (i128::from(self.min(variable)), i128::from(self.max(variable)));
                    let coefficient = i128::from(coefficient);
                    let (lower, upper) = if coefficient >= 0 {
                        (coefficient * minimum, coefficient * maximum)
                    } else {
                        (coefficient * maximum, coefficient * minimum)
                    };
                    bounds = bounds.and_then(|(lo, hi)| Some((lo.checked_add(lower)?, hi.checked_add(upper)?)));
                }
                let materialized_bounds = bounds.and_then(|(lo, hi)| {
                    let span = hi.checked_sub(lo)?;
                    (span <= i128::from(MAX_MATERIALIZED_OBJECTIVE_SPAN))
                        .then(|| i32::try_from(lo).ok().zip(i32::try_from(hi).ok()))
                        .flatten()
                });
                if materialized_bounds.is_none() {
                    self.objective = Some(SemanticObjective::IntExpr { minimize, expr: weighted_sum_expr(&coeffs, &vars) });
                    return Ok(());
                }
                let (lo, hi) = materialized_bounds.expect("checked above");
                let obj = self.package.model.int_range(lo, hi);
                let mut coefficients = coeffs;
                coefficients.push(-1);
                let mut variables = vars;
                variables.push(obj);
                linear(&mut self.package.model, &coefficients, &variables, Relation::Eq, 0);
                obj
            }
            Minimum | Maximum => {
                if !aligned {
                    return Err("objective: coeffs/terms length mismatch".to_string());
                }
                // Fold coefficients into aux terms when they aren't all 1.
                let terms: Vec<IntVarRef> = if coeffs.iter().all(|&c| c == 1) {
                    vars
                } else {
                    vars.iter().zip(&coeffs).map(|(&v, &c)| self.aux_for(expr::mul(vec![expr::int(c), expr::var(v)]))).collect()
                };
                let lo = terms.iter().map(|&v| self.min(v)).min().unwrap_or(0);
                let hi = terms.iter().map(|&v| self.max(v)).max().unwrap_or(0);
                let obj = self.package.model.int_range(lo, hi);
                if matches!(t, Minimum) {
                    minimum(&mut self.package.model, obj, &terms);
                } else {
                    maximum(&mut self.package.model, obj, &terms);
                }
                obj
            }
            NValues => self.nvalues_var(&vars),
            _ => return Err("unsupported objective type (product/lex)".to_string()),
        };
        self.objective = Some(SemanticObjective::IntExpr { minimize, expr: expr::var(obj) });
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
