use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use pyo3::class::basic::CompareOp;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyModule};

use crate::constraints::count;
use crate::constraints::graph;
use crate::constraints::intension;
use crate::constraints::lex;
use crate::constraints::linear::{self, Relation};
use crate::constraints::primitives;
use crate::constraints::scheduling;
use crate::constraints::table;
use crate::expr::Expr;
use crate::ids::VarId;
use crate::search::{self, Objective as SearchObjective, SearchControl, SolveStats};
use crate::Solver;

static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct ExprLike {
    model_id: Option<u64>,
    expr: Expr,
    text: String,
}

impl ExprLike {
    fn int(value: i64) -> Self {
        Self { model_id: None, expr: Expr::Const(value), text: value.to_string() }
    }

    fn var(var: &PyIntVar) -> Self {
        Self { model_id: Some(var.model_id), expr: Expr::Var(VarId(var.index)), text: var.display_name() }
    }
}

#[pyclass(name = "IntVar", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyIntVar {
    model_id: u64,
    index: u32,
    name: Option<String>,
}

#[pyclass(name = "Expr", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyExpr {
    inner: ExprLike,
}

#[pyclass(name = "Constraint", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyConstraint {
    inner: ExprLike,
}

#[pyclass(name = "SolveStats", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PySolveStats {
    #[pyo3(get)]
    solutions: u64,
    #[pyo3(get)]
    nodes: u64,
    #[pyo3(get)]
    failures: u64,
    #[pyo3(get)]
    learned_lits: u64,
    #[pyo3(get)]
    vivified_clauses: u64,
    #[pyo3(get)]
    vivified_lits: u64,
}

#[pyclass(name = "Solution", module = "qayd")]
struct PySolution {
    status: String,
    objective: Option<i64>,
    objective_sense: Option<String>,
    objective_expr: Option<String>,
    values: Vec<Option<i32>>,
    stats: PySolveStats,
}

#[derive(Clone)]
struct ObjectiveSpec {
    minimizing: bool,
    expr: ExprLike,
}

#[pyclass(name = "Model", module = "qayd", unsendable)]
struct PyModel {
    id: u64,
    solver: Solver,
    names: Vec<Option<String>>,
    objective: Option<ObjectiveSpec>,
}

impl PyIntVar {
    fn display_name(&self) -> String {
        self.name.clone().unwrap_or_else(|| format!("x{}", self.index))
    }
}

impl From<SolveStats> for PySolveStats {
    fn from(stats: SolveStats) -> Self {
        Self {
            solutions: stats.solutions,
            nodes: stats.nodes,
            failures: stats.failures,
            learned_lits: stats.learned_lits,
            vivified_clauses: stats.vivified_clauses,
            vivified_lits: stats.vivified_lits,
        }
    }
}

fn merge_model_ids(a: Option<u64>, b: Option<u64>) -> PyResult<Option<u64>> {
    match (a, b) {
        (Some(x), Some(y)) if x != y => Err(PyValueError::new_err("cannot combine expressions from different models")),
        (Some(x), _) => Ok(Some(x)),
        (_, Some(y)) => Ok(Some(y)),
        _ => Ok(None),
    }
}

fn expr_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ExprLike> {
    if let Ok(var) = obj.extract::<PyRef<'_, PyIntVar>>() {
        return Ok(ExprLike::var(&var));
    }
    if let Ok(expr) = obj.extract::<PyRef<'_, PyExpr>>() {
        return Ok(expr.inner.clone());
    }
    if let Ok(constraint) = obj.extract::<PyRef<'_, PyConstraint>>() {
        return Ok(constraint.inner.clone());
    }
    if let Ok(value) = obj.extract::<i64>() {
        return Ok(ExprLike::int(value));
    }
    Err(PyTypeError::new_err("expected an IntVar, Expr, Constraint, or integer"))
}

fn constraint_from_py(obj: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
    if let Ok(constraint) = obj.extract::<PyRef<'_, PyConstraint>>() {
        return Ok(constraint.clone());
    }
    Err(PyTypeError::new_err("expected a Constraint"))
}

fn expr_list_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Vec<ExprLike>> {
    let iter = PyIterator::from_object(obj)?;
    iter.map(|item| expr_from_py(&item?)).collect()
}

fn var_from_py(obj: &Bound<'_, PyAny>) -> PyResult<PyIntVar> {
    obj.extract::<PyRef<'_, PyIntVar>>().map(|var| var.clone()).map_err(|_| PyTypeError::new_err("expected an IntVar"))
}

fn var_list_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Vec<PyIntVar>> {
    let iter = PyIterator::from_object(obj)?;
    iter.map(|item| var_from_py(&item?)).collect()
}

fn var_matrix_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<PyIntVar>>> {
    let iter = PyIterator::from_object(obj)?;
    iter.map(|row| var_list_from_py(&row?)).collect()
}

fn ids_for(model_id: u64, vars: &[PyIntVar]) -> PyResult<Vec<VarId>> {
    let mut out = Vec::with_capacity(vars.len());
    for var in vars {
        if var.model_id != model_id {
            return Err(PyValueError::new_err("variable belongs to a different model"));
        }
        out.push(VarId(var.index));
    }
    Ok(out)
}

fn one_id_for(model_id: u64, var: &PyIntVar) -> PyResult<VarId> {
    if var.model_id != model_id {
        return Err(PyValueError::new_err("variable belongs to a different model"));
    }
    Ok(VarId(var.index))
}

fn search_ids(model: &PyModel, search: Option<&Bound<'_, PyAny>>, extra: Option<VarId>) -> PyResult<Vec<VarId>> {
    let mut vars = match search {
        Some(obj) if !obj.is_none() => ids_for(model.id, &var_list_from_py(obj)?)?,
        _ => (0..model.names.len()).map(|i| VarId(i as u32)).collect(),
    };
    if let Some(var) = extra {
        if !vars.contains(&var) {
            vars.push(var);
        }
    }
    Ok(vars)
}

fn append_expr_vars(vars: &mut Vec<VarId>, expr: &Expr) {
    let mut objective_vars = Vec::new();
    expr.collect_vars(&mut objective_vars);
    for var in objective_vars {
        if !vars.contains(&var) {
            vars.push(var);
        }
    }
}

fn make_solution(
    status: &str,
    vars: &[VarId],
    assignment: Option<&[i32]>,
    objective: Option<i64>,
    objective_sense: Option<&str>,
    objective_expr: Option<&str>,
    stats: SolveStats,
    num_vars: usize,
) -> PySolution {
    let mut values = vec![None; num_vars];
    if let Some(assignment) = assignment {
        for (&var, &value) in vars.iter().zip(assignment) {
            values[var.index()] = Some(value);
        }
    }
    PySolution {
        status: status.to_string(),
        objective,
        objective_sense: objective_sense.map(str::to_string),
        objective_expr: objective_expr.map(str::to_string),
        values,
        stats: stats.into(),
    }
}

fn parse_objective_sense(sense: &str) -> PyResult<bool> {
    match sense {
        "min" | "minimize" | "minimum" => Ok(true),
        "max" | "maximize" | "maximum" => Ok(false),
        _ => Err(PyValueError::new_err("objective sense must be 'min' or 'max'")),
    }
}

fn verbose_start(num_vars: usize, num_constraints: usize, has_objective: bool) {
    println!("qayd solve");
    println!("  variables: {num_vars}");
    println!("  constraints: {num_constraints}");
    println!("  objective: {}", if has_objective { "yes" } else { "no" });
}

fn verbose_finish(solution: &PySolution) {
    println!("qayd result");
    println!("  status: {}", solution.status);
    if let Some(objective) = solution.objective {
        println!("  objective: {objective}");
    }
    println!("  solutions: {}", solution.stats.solutions);
    println!("  nodes: {}", solution.stats.nodes);
    println!("  failures: {}", solution.stats.failures);
    println!("  learned_lits: {}", solution.stats.learned_lits);
}

fn binary_expr(lhs: ExprLike, rhs: &Bound<'_, PyAny>, op: &str, f: impl FnOnce(Expr, Expr) -> Expr) -> PyResult<PyExpr> {
    let rhs = expr_from_py(rhs)?;
    let model_id = merge_model_ids(lhs.model_id, rhs.model_id)?;
    Ok(PyExpr { inner: ExprLike { model_id, expr: f(lhs.expr, rhs.expr), text: format!("({} {} {})", lhs.text, op, rhs.text) } })
}

fn rbinary_expr(lhs: &Bound<'_, PyAny>, rhs: ExprLike, op: &str, f: impl FnOnce(Expr, Expr) -> Expr) -> PyResult<PyExpr> {
    let lhs = expr_from_py(lhs)?;
    let model_id = merge_model_ids(lhs.model_id, rhs.model_id)?;
    Ok(PyExpr { inner: ExprLike { model_id, expr: f(lhs.expr, rhs.expr), text: format!("({} {} {})", lhs.text, op, rhs.text) } })
}

fn unary_expr(arg: ExprLike, text: String, f: impl FnOnce(Expr) -> Expr) -> PyExpr {
    PyExpr { inner: ExprLike { model_id: arg.model_id, expr: f(arg.expr), text } }
}

fn compare_expr(lhs: ExprLike, rhs: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<PyConstraint> {
    let rhs = expr_from_py(rhs)?;
    let model_id = merge_model_ids(lhs.model_id, rhs.model_id)?;
    let op_text = match op {
        CompareOp::Lt => "<",
        CompareOp::Le => "<=",
        CompareOp::Eq => "==",
        CompareOp::Ne => "!=",
        CompareOp::Gt => ">",
        CompareOp::Ge => ">=",
    };
    let expr = match op {
        CompareOp::Lt => Expr::Lt(Box::new(lhs.expr), Box::new(rhs.expr)),
        CompareOp::Le => Expr::Le(Box::new(lhs.expr), Box::new(rhs.expr)),
        CompareOp::Eq => Expr::Eq(Box::new(lhs.expr), Box::new(rhs.expr)),
        CompareOp::Ne => Expr::Ne(Box::new(lhs.expr), Box::new(rhs.expr)),
        CompareOp::Gt => Expr::Gt(Box::new(lhs.expr), Box::new(rhs.expr)),
        CompareOp::Ge => Expr::Ge(Box::new(lhs.expr), Box::new(rhs.expr)),
    };
    Ok(PyConstraint { inner: ExprLike { model_id, expr, text: format!("({} {} {})", lhs.text, op_text, rhs.text) } })
}

fn constraint_binary(lhs: ExprLike, rhs: &Bound<'_, PyAny>, op: &str, f: impl FnOnce(Vec<Expr>) -> Expr) -> PyResult<PyConstraint> {
    let rhs = expr_from_py(rhs)?;
    let model_id = merge_model_ids(lhs.model_id, rhs.model_id)?;
    Ok(PyConstraint {
        inner: ExprLike { model_id, expr: f(vec![lhs.expr, rhs.expr]), text: format!("({} {} {})", lhs.text, op, rhs.text) },
    })
}

fn parse_relation(relation: &str) -> PyResult<Relation> {
    match relation {
        "==" | "=" | "eq" | "Eq" => Ok(Relation::Eq),
        "!=" | "<>" | "ne" | "Ne" => Ok(Relation::Ne),
        "<=" | "le" | "Le" => Ok(Relation::Le),
        "<" | "lt" | "Lt" => Ok(Relation::Lt),
        ">=" | "ge" | "Ge" => Ok(Relation::Ge),
        ">" | "gt" | "Gt" => Ok(Relation::Gt),
        _ => Err(PyValueError::new_err("unknown relation, expected one of == != <= < >= >")),
    }
}

fn checked_i32(value: i64, name: &str) -> PyResult<i32> {
    i32::try_from(value).map_err(|_| PyValueError::new_err(format!("{name} is outside the i32 domain range")))
}

#[pymethods]
impl PyIntVar {
    #[getter]
    fn index(&self) -> u32 {
        self.index
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.name.clone()
    }

    fn expr(&self) -> PyExpr {
        PyExpr { inner: ExprLike::var(self) }
    }

    fn __repr__(&self) -> String {
        self.display_name()
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err("IntVar cannot be used as a Python bool, build a Constraint instead"))
    }

    fn __neg__(&self) -> PyExpr {
        unary_expr(ExprLike::var(self), format!("-{}", self.display_name()), |a| Expr::Neg(Box::new(a)))
    }

    fn __abs__(&self) -> PyExpr {
        unary_expr(ExprLike::var(self), format!("abs({})", self.display_name()), |a| Expr::Abs(Box::new(a)))
    }

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        binary_expr(ExprLike::var(self), rhs, "+", |a, b| Expr::Add(vec![a, b]))
    }

    fn __radd__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        rbinary_expr(lhs, ExprLike::var(self), "+", |a, b| Expr::Add(vec![a, b]))
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        binary_expr(ExprLike::var(self), rhs, "-", |a, b| Expr::Sub(Box::new(a), Box::new(b)))
    }

    fn __rsub__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        rbinary_expr(lhs, ExprLike::var(self), "-", |a, b| Expr::Sub(Box::new(a), Box::new(b)))
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        binary_expr(ExprLike::var(self), rhs, "*", |a, b| Expr::Mul(vec![a, b]))
    }

    fn __rmul__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        rbinary_expr(lhs, ExprLike::var(self), "*", |a, b| Expr::Mul(vec![a, b]))
    }

    fn __floordiv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        binary_expr(ExprLike::var(self), rhs, "//", |a, b| Expr::Div(Box::new(a), Box::new(b)))
    }

    fn __rfloordiv__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        rbinary_expr(lhs, ExprLike::var(self), "//", |a, b| Expr::Div(Box::new(a), Box::new(b)))
    }

    fn __mod__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        binary_expr(ExprLike::var(self), rhs, "%", |a, b| Expr::Mod(Box::new(a), Box::new(b)))
    }

    fn __rmod__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        rbinary_expr(lhs, ExprLike::var(self), "%", |a, b| Expr::Mod(Box::new(a), Box::new(b)))
    }

    fn __richcmp__(&self, rhs: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<PyConstraint> {
        compare_expr(ExprLike::var(self), rhs, op)
    }
}

#[pymethods]
impl PyExpr {
    fn __repr__(&self) -> String {
        self.inner.text.clone()
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err("Expr cannot be used as a Python bool, compare it to build a Constraint"))
    }

    fn __neg__(&self) -> PyExpr {
        unary_expr(self.inner.clone(), format!("-{}", self.inner.text), |a| Expr::Neg(Box::new(a)))
    }

    fn __abs__(&self) -> PyExpr {
        unary_expr(self.inner.clone(), format!("abs({})", self.inner.text), |a| Expr::Abs(Box::new(a)))
    }

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        binary_expr(self.inner.clone(), rhs, "+", |a, b| Expr::Add(vec![a, b]))
    }

    fn __radd__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        rbinary_expr(lhs, self.inner.clone(), "+", |a, b| Expr::Add(vec![a, b]))
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        binary_expr(self.inner.clone(), rhs, "-", |a, b| Expr::Sub(Box::new(a), Box::new(b)))
    }

    fn __rsub__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        rbinary_expr(lhs, self.inner.clone(), "-", |a, b| Expr::Sub(Box::new(a), Box::new(b)))
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        binary_expr(self.inner.clone(), rhs, "*", |a, b| Expr::Mul(vec![a, b]))
    }

    fn __rmul__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        rbinary_expr(lhs, self.inner.clone(), "*", |a, b| Expr::Mul(vec![a, b]))
    }

    fn __floordiv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        binary_expr(self.inner.clone(), rhs, "//", |a, b| Expr::Div(Box::new(a), Box::new(b)))
    }

    fn __rfloordiv__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        rbinary_expr(lhs, self.inner.clone(), "//", |a, b| Expr::Div(Box::new(a), Box::new(b)))
    }

    fn __mod__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        binary_expr(self.inner.clone(), rhs, "%", |a, b| Expr::Mod(Box::new(a), Box::new(b)))
    }

    fn __rmod__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        rbinary_expr(lhs, self.inner.clone(), "%", |a, b| Expr::Mod(Box::new(a), Box::new(b)))
    }

    fn __richcmp__(&self, rhs: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<PyConstraint> {
        compare_expr(self.inner.clone(), rhs, op)
    }
}

#[pymethods]
impl PyConstraint {
    fn __repr__(&self) -> String {
        self.inner.text.clone()
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err("Constraint cannot be used as a Python bool, pass it to Model.add"))
    }

    fn __invert__(&self) -> PyConstraint {
        PyConstraint {
            inner: ExprLike {
                model_id: self.inner.model_id,
                expr: Expr::Not(Box::new(self.inner.expr.clone())),
                text: format!("~{}", self.inner.text),
            },
        }
    }

    fn __and__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
        constraint_binary(self.inner.clone(), rhs, "&", Expr::And)
    }

    fn __or__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
        constraint_binary(self.inner.clone(), rhs, "|", Expr::Or)
    }

    fn implies(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
        let rhs = expr_from_py(rhs)?;
        let model_id = merge_model_ids(self.inner.model_id, rhs.model_id)?;
        Ok(PyConstraint {
            inner: ExprLike {
                model_id,
                expr: Expr::Imp(Box::new(self.inner.expr.clone()), Box::new(rhs.expr)),
                text: format!("({}) -> ({})", self.inner.text, rhs.text),
            },
        })
    }

    fn iff(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
        let rhs = expr_from_py(rhs)?;
        let model_id = merge_model_ids(self.inner.model_id, rhs.model_id)?;
        Ok(PyConstraint {
            inner: ExprLike {
                model_id,
                expr: Expr::Iff(Box::new(self.inner.expr.clone()), Box::new(rhs.expr)),
                text: format!("({}) <-> ({})", self.inner.text, rhs.text),
            },
        })
    }
}

#[pymethods]
impl PySolveStats {
    fn __repr__(&self) -> String {
        format!("SolveStats(solutions={}, nodes={}, failures={})", self.solutions, self.nodes, self.failures)
    }
}

#[pymethods]
impl PySolution {
    #[getter]
    fn status(&self) -> String {
        self.status.clone()
    }

    #[getter]
    fn objective(&self) -> Option<i64> {
        self.objective
    }

    #[getter]
    fn objective_sense(&self) -> Option<String> {
        self.objective_sense.clone()
    }

    #[getter]
    fn objective_expr(&self) -> Option<String> {
        self.objective_expr.clone()
    }

    #[getter]
    fn stats(&self) -> PySolveStats {
        self.stats.clone()
    }

    fn is_sat(&self) -> bool {
        self.status != "UNSATISFIABLE"
    }

    fn value(&self, var: &PyIntVar) -> PyResult<i32> {
        self.values
            .get(var.index as usize)
            .copied()
            .flatten()
            .ok_or_else(|| PyRuntimeError::new_err("no value is available for this variable in the solution"))
    }

    fn assignment(&self) -> HashMap<u32, i32> {
        self.values.iter().enumerate().filter_map(|(index, value)| value.map(|value| (index as u32, value))).collect()
    }

    fn __bool__(&self) -> bool {
        self.is_sat()
    }

    fn __repr__(&self) -> String {
        match self.objective {
            Some(value) => format!("Solution(status='{}', objective={}, sense={:?})", self.status, value, self.objective_sense),
            None => format!("Solution(status='{}')", self.status),
        }
    }
}

#[pymethods]
impl PyModel {
    #[new]
    fn new() -> Self {
        Self { id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed), solver: Solver::new(), names: Vec::new(), objective: None }
    }

    #[getter]
    fn num_vars(&self) -> usize {
        self.names.len()
    }

    #[getter]
    fn num_constraints(&self) -> usize {
        self.solver.num_propagators()
    }

    #[getter]
    fn objective_sense(&self) -> Option<String> {
        self.objective.as_ref().map(|objective| if objective.minimizing { "min".to_string() } else { "max".to_string() })
    }

    #[getter]
    fn objective_expr(&self) -> Option<String> {
        self.objective.as_ref().map(|objective| objective.expr.text.clone())
    }

    #[pyo3(signature = (lo=None, hi=None, *, values=None, name=None))]
    fn int_var(&mut self, lo: Option<i32>, hi: Option<i32>, values: Option<Vec<i32>>, name: Option<String>) -> PyResult<PyIntVar> {
        let id = match (lo, hi, values) {
            (None, None, Some(values)) => {
                if values.is_empty() {
                    return Err(PyValueError::new_err("domain values cannot be empty"));
                }
                self.solver.new_var_set(&values)
            }
            (Some(lo), Some(hi), None) => {
                if lo > hi {
                    return Err(PyValueError::new_err("lower bound must be <= upper bound"));
                }
                self.solver.new_var_range(lo, hi)
            }
            _ => return Err(PyValueError::new_err("use int_var(lo, hi, name=...) or int_var(values=[...], name=...)")),
        };
        self.names.push(name.clone());
        Ok(PyIntVar { model_id: self.id, index: id.0, name })
    }

    #[pyo3(signature = (name=None))]
    fn bool_var(&mut self, name: Option<String>) -> PyResult<PyIntVar> {
        self.int_var(Some(0), Some(1), None, name)
    }

    #[pyo3(signature = (n, lo, hi, *, name=None))]
    fn int_vars(&mut self, n: usize, lo: i32, hi: i32, name: Option<String>) -> PyResult<Vec<PyIntVar>> {
        let mut vars = Vec::with_capacity(n);
        for i in 0..n {
            let var_name = name.as_ref().map(|prefix| format!("{prefix}[{i}]"));
            vars.push(self.int_var(Some(lo), Some(hi), None, var_name)?);
        }
        Ok(vars)
    }

    fn variables(&self) -> Vec<PyIntVar> {
        self.names.iter().enumerate().map(|(index, name)| PyIntVar { model_id: self.id, index: index as u32, name: name.clone() }).collect()
    }

    fn add(&mut self, constraint: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(constraint) = constraint_from_py(constraint) {
            if let Some(model_id) = constraint.inner.model_id {
                if model_id != self.id {
                    return Err(PyValueError::new_err("constraint belongs to a different model"));
                }
            }
            intension::intension(&mut self.solver, constraint.inner.expr);
            return Ok(());
        }
        let iter = PyIterator::from_object(constraint)
            .map_err(|_| PyTypeError::new_err("expected a Constraint or an iterable of Constraint objects"))?;
        for item in iter {
            self.add(&item?)?;
        }
        Ok(())
    }

    #[pyo3(signature = (coeffs, vars, relation, rhs))]
    fn linear(&mut self, coeffs: Vec<i64>, vars: &Bound<'_, PyAny>, relation: &str, rhs: i64) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if coeffs.len() != vars.len() {
            return Err(PyValueError::new_err("coeffs and vars must have the same length"));
        }
        linear::linear(&mut self.solver, &coeffs, &vars, parse_relation(relation)?, rhs);
        Ok(())
    }

    #[pyo3(signature = (vars, relation, rhs))]
    fn sum(&mut self, vars: &Bound<'_, PyAny>, relation: &str, rhs: i64) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        linear::sum(&mut self.solver, &vars, parse_relation(relation)?, rhs);
        Ok(())
    }

    fn all_different(&mut self, vars: &Bound<'_, PyAny>) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        primitives::all_different(&mut self.solver, &vars);
        Ok(())
    }

    fn all_equal(&mut self, vars: &Bound<'_, PyAny>) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        primitives::all_equal(&mut self.solver, &vars);
        Ok(())
    }

    fn not_equal(&mut self, x: &PyIntVar, y: &PyIntVar) -> PyResult<()> {
        let x = one_id_for(self.id, x)?;
        let y = one_id_for(self.id, y)?;
        primitives::not_equal(&mut self.solver, x, y);
        Ok(())
    }

    #[pyo3(signature = (x, y, offset=0))]
    fn not_equal_offset(&mut self, x: &PyIntVar, y: &PyIntVar, offset: i32) -> PyResult<()> {
        let x = one_id_for(self.id, x)?;
        let y = one_id_for(self.id, y)?;
        primitives::not_equal_offset(&mut self.solver, x, y, offset);
        Ok(())
    }

    fn ordered(&mut self, vars: &Bound<'_, PyAny>, relation: &str) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        primitives::ordered(&mut self.solver, &vars, parse_relation(relation)?);
        Ok(())
    }

    fn instantiate(&mut self, vars: &Bound<'_, PyAny>, values: Vec<i32>) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if vars.len() != values.len() {
            return Err(PyValueError::new_err("vars and values must have the same length"));
        }
        primitives::instantiation(&mut self.solver, &vars, &values);
        Ok(())
    }

    fn minimum(&mut self, target: &PyIntVar, vars: &Bound<'_, PyAny>) -> PyResult<()> {
        let target = one_id_for(self.id, target)?;
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if vars.is_empty() {
            return Err(PyValueError::new_err("minimum requires at least one variable"));
        }
        primitives::minimum(&mut self.solver, target, &vars);
        Ok(())
    }

    fn maximum(&mut self, target: &PyIntVar, vars: &Bound<'_, PyAny>) -> PyResult<()> {
        let target = one_id_for(self.id, target)?;
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if vars.is_empty() {
            return Err(PyValueError::new_err("maximum requires at least one variable"));
        }
        primitives::maximum(&mut self.solver, target, &vars);
        Ok(())
    }

    fn element(&mut self, array: &Bound<'_, PyAny>, index: &PyIntVar, value: &PyIntVar) -> PyResult<()> {
        let array = ids_for(self.id, &var_list_from_py(array)?)?;
        if array.is_empty() {
            return Err(PyValueError::new_err("element requires a non-empty array"));
        }
        let index = one_id_for(self.id, index)?;
        let value = one_id_for(self.id, value)?;
        primitives::element(&mut self.solver, &array, index, value);
        Ok(())
    }

    fn element_const(&mut self, array: Vec<i32>, index: &PyIntVar, value: &PyIntVar) -> PyResult<()> {
        if array.is_empty() {
            return Err(PyValueError::new_err("element_const requires a non-empty array"));
        }
        let index = one_id_for(self.id, index)?;
        let value = one_id_for(self.id, value)?;
        primitives::element_const(&mut self.solver, &array, index, value);
        Ok(())
    }

    fn count(&mut self, vars: &Bound<'_, PyAny>, value: i32, relation: &str, k: i64) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        count::count(&mut self.solver, &vars, value, parse_relation(relation)?, k);
        Ok(())
    }

    #[pyo3(signature = (vars, values, low, high, *, closed=false))]
    fn cardinality(&mut self, vars: &Bound<'_, PyAny>, values: Vec<i32>, low: Vec<i64>, high: Vec<i64>, closed: bool) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if values.len() != low.len() || values.len() != high.len() {
            return Err(PyValueError::new_err("values, low, and high must have the same length"));
        }
        count::cardinality(&mut self.solver, &vars, &values, &low, &high, closed);
        Ok(())
    }

    fn n_values(&mut self, vars: &Bound<'_, PyAny>, relation: &str, k: i64) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        count::n_values(&mut self.solver, &vars, parse_relation(relation)?, k);
        Ok(())
    }

    #[pyo3(signature = (vars, tuples, *, positive=true))]
    fn table(&mut self, vars: &Bound<'_, PyAny>, tuples: Vec<Vec<i32>>, positive: bool) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if tuples.iter().any(|tuple| tuple.len() != vars.len()) {
            return Err(PyValueError::new_err("every tuple must match the variable arity"));
        }
        table::extension(&mut self.solver, &vars, &tuples, positive);
        Ok(())
    }

    #[pyo3(signature = (x, y, *, strict=false))]
    fn lex(&mut self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, strict: bool) -> PyResult<()> {
        let x = ids_for(self.id, &var_list_from_py(x)?)?;
        let y = ids_for(self.id, &var_list_from_py(y)?)?;
        if x.len() != y.len() {
            return Err(PyValueError::new_err("lex vectors must have the same length"));
        }
        lex::lex(&mut self.solver, &x, &y, strict);
        Ok(())
    }

    #[pyo3(signature = (rows, *, strict=false))]
    fn lex_chain(&mut self, rows: &Bound<'_, PyAny>, strict: bool) -> PyResult<()> {
        let rows = var_matrix_from_py(rows)?;
        let rows: Vec<Vec<VarId>> = rows.iter().map(|row| ids_for(self.id, row)).collect::<PyResult<_>>()?;
        lex::lex_chain(&mut self.solver, &rows, strict);
        Ok(())
    }

    fn channel(&mut self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let x = ids_for(self.id, &var_list_from_py(x)?)?;
        let y = ids_for(self.id, &var_list_from_py(y)?)?;
        if x.len() != y.len() {
            return Err(PyValueError::new_err("channel vectors must have the same length"));
        }
        lex::channel(&mut self.solver, &x, &y);
        Ok(())
    }

    fn no_overlap(&mut self, starts: &Bound<'_, PyAny>, durations: Vec<i64>) -> PyResult<()> {
        let starts = ids_for(self.id, &var_list_from_py(starts)?)?;
        if starts.len() != durations.len() {
            return Err(PyValueError::new_err("starts and durations must have the same length"));
        }
        scheduling::no_overlap(&mut self.solver, &starts, &durations);
        Ok(())
    }

    fn cumulative(&mut self, starts: &Bound<'_, PyAny>, durations: Vec<i64>, heights: Vec<i64>, capacity: i64) -> PyResult<()> {
        let starts = ids_for(self.id, &var_list_from_py(starts)?)?;
        if starts.len() != durations.len() || starts.len() != heights.len() {
            return Err(PyValueError::new_err("starts, durations, and heights must have the same length"));
        }
        scheduling::cumulative(&mut self.solver, &starts, &durations, &heights, capacity);
        Ok(())
    }

    fn cumulative_var(
        &mut self,
        starts: &Bound<'_, PyAny>,
        durations: &Bound<'_, PyAny>,
        heights: &Bound<'_, PyAny>,
        capacity: &PyIntVar,
    ) -> PyResult<()> {
        let starts = ids_for(self.id, &var_list_from_py(starts)?)?;
        let durations = ids_for(self.id, &var_list_from_py(durations)?)?;
        let heights = ids_for(self.id, &var_list_from_py(heights)?)?;
        if starts.len() != durations.len() || starts.len() != heights.len() {
            return Err(PyValueError::new_err("starts, durations, and heights must have the same length"));
        }
        let capacity = one_id_for(self.id, capacity)?;
        scheduling::cumulative_var(&mut self.solver, &starts, &durations, &heights, capacity);
        Ok(())
    }

    fn bin_packing(&mut self, items: &Bound<'_, PyAny>, sizes: Vec<i64>, capacities: Vec<i64>) -> PyResult<()> {
        let items = ids_for(self.id, &var_list_from_py(items)?)?;
        if items.len() != sizes.len() {
            return Err(PyValueError::new_err("items and sizes must have the same length"));
        }
        scheduling::bin_packing(&mut self.solver, &items, &sizes, &capacities);
        Ok(())
    }

    fn knapsack(
        &mut self,
        vars: &Bound<'_, PyAny>,
        weights: Vec<i64>,
        profits: Vec<i64>,
        weight_relation: &str,
        weight_limit: i64,
        profit_relation: &str,
        profit_limit: i64,
    ) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if vars.len() != weights.len() || vars.len() != profits.len() {
            return Err(PyValueError::new_err("vars, weights, and profits must have the same length"));
        }
        scheduling::knapsack(
            &mut self.solver,
            &vars,
            &weights,
            &profits,
            parse_relation(weight_relation)?,
            weight_limit,
            parse_relation(profit_relation)?,
            profit_limit,
        );
        Ok(())
    }

    fn circuit(&mut self, successors: &Bound<'_, PyAny>) -> PyResult<()> {
        let successors = ids_for(self.id, &var_list_from_py(successors)?)?;
        graph::circuit(&mut self.solver, &successors);
        Ok(())
    }

    #[pyo3(signature = (objective, *, sense="min"))]
    fn objective(&mut self, objective: &Bound<'_, PyAny>, sense: &str) -> PyResult<()> {
        let expr = expr_from_py(objective)?;
        if let Some(model_id) = expr.model_id {
            if model_id != self.id {
                return Err(PyValueError::new_err("objective belongs to a different model"));
            }
        }
        let mut objective_vars = Vec::new();
        expr.expr.collect_vars(&mut objective_vars);
        if objective_vars.is_empty() {
            return Err(PyValueError::new_err("objective must reference at least one model variable"));
        }
        self.objective = Some(ObjectiveSpec { minimizing: parse_objective_sense(sense)?, expr });
        Ok(())
    }

    fn clear_objective(&mut self) {
        self.objective = None;
    }

    #[pyo3(signature = (*, search=None, verbose=false))]
    fn solve(&self, search: Option<&Bound<'_, PyAny>>, verbose: bool) -> PyResult<PySolution> {
        if let Some(objective) = &self.objective {
            return self.solve_optimization(objective, search, verbose);
        }
        if verbose {
            verbose_start(self.names.len(), self.solver.num_propagators(), false);
        }
        let vars = search_ids(self, search, None)?;
        let mut solver = self.solver.clone();
        let mut assignment = None;
        let stats = search::solve(&mut solver, &vars, |solver| {
            assignment = Some(vars.iter().map(|&var| solver.store.value(var)).collect::<Vec<_>>());
            SearchControl::Stop
        });
        let status = if assignment.is_some() { "SATISFIABLE" } else { "UNSATISFIABLE" };
        let solution = make_solution(status, &vars, assignment.as_deref(), None, None, None, stats, self.names.len());
        if verbose {
            verbose_finish(&solution);
        }
        Ok(solution)
    }

    #[pyo3(signature = (search=None))]
    fn count_solutions(&self, search: Option<&Bound<'_, PyAny>>) -> PyResult<u64> {
        let vars = search_ids(self, search, None)?;
        let mut solver = self.solver.clone();
        Ok(search::count_solutions(&mut solver, &vars))
    }

    #[pyo3(signature = (objective, *, search=None, verbose=false))]
    fn minimize(&self, objective: &Bound<'_, PyAny>, search: Option<&Bound<'_, PyAny>>, verbose: bool) -> PyResult<PySolution> {
        let objective = ObjectiveSpec { minimizing: true, expr: expr_from_py(objective)? };
        self.solve_optimization(&objective, search, verbose)
    }

    #[pyo3(signature = (objective, *, search=None, verbose=false))]
    fn maximize(&self, objective: &Bound<'_, PyAny>, search: Option<&Bound<'_, PyAny>>, verbose: bool) -> PyResult<PySolution> {
        let objective = ObjectiveSpec { minimizing: false, expr: expr_from_py(objective)? };
        self.solve_optimization(&objective, search, verbose)
    }

    fn __repr__(&self) -> String {
        format!("Model(num_vars={}, num_constraints={})", self.names.len(), self.solver.num_propagators())
    }
}

impl PyModel {
    fn solve_optimization(&self, objective: &ObjectiveSpec, search: Option<&Bound<'_, PyAny>>, verbose: bool) -> PyResult<PySolution> {
        if let Some(model_id) = objective.expr.model_id {
            if model_id != self.id {
                return Err(PyValueError::new_err("objective belongs to a different model"));
            }
        }
        if verbose {
            verbose_start(self.names.len(), self.solver.num_propagators(), true);
            println!("  direction: {}", if objective.minimizing { "min" } else { "max" });
            println!("  expression: {}", objective.expr.text);
        }
        let mut vars = search_ids(self, search, None)?;
        append_expr_vars(&mut vars, &objective.expr.expr);
        let mut solver = self.solver.clone();
        let stop = AtomicBool::new(false);
        let search_objective = match &objective.expr.expr {
            Expr::Var(var) => SearchObjective::Var(*var),
            expr => SearchObjective::Expr(expr),
        };
        let (best, stats, _) = search::optimize_seeded(
            &mut solver,
            &vars,
            search_objective,
            objective.minimizing,
            &stop,
            0,
            None,
            None,
            &[],
            None,
            |value, _| {
                if verbose {
                    println!("  incumbent: {value}");
                }
            },
        );
        let Some((assignment, objective_value)) = best else {
            let solution = make_solution(
                "UNSATISFIABLE",
                &vars,
                None,
                None,
                Some(if objective.minimizing { "min" } else { "max" }),
                Some(&objective.expr.text),
                stats,
                self.names.len(),
            );
            if verbose {
                verbose_finish(&solution);
            }
            return Ok(solution);
        };
        let solution = make_solution(
            "OPTIMAL",
            &vars,
            Some(&assignment),
            Some(objective_value),
            Some(if objective.minimizing { "min" } else { "max" }),
            Some(&objective.expr.text),
            stats,
            self.names.len(),
        );
        if verbose {
            verbose_finish(&solution);
        }
        Ok(solution)
    }
}

#[pyfunction(name = "expr")]
fn expr_fn(value: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    Ok(PyExpr { inner: expr_from_py(value)? })
}

#[pyfunction(name = "all")]
fn all_fn(items: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
    let items = expr_list_from_py(items)?;
    let mut model_id = None;
    let mut exprs = Vec::with_capacity(items.len());
    let mut texts = Vec::with_capacity(items.len());
    for item in items {
        model_id = merge_model_ids(model_id, item.model_id)?;
        exprs.push(item.expr);
        texts.push(item.text);
    }
    Ok(PyConstraint { inner: ExprLike { model_id, expr: Expr::And(exprs), text: format!("all({})", texts.join(", ")) } })
}

#[pyfunction(name = "any")]
fn any_fn(items: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
    let items = expr_list_from_py(items)?;
    let mut model_id = None;
    let mut exprs = Vec::with_capacity(items.len());
    let mut texts = Vec::with_capacity(items.len());
    for item in items {
        model_id = merge_model_ids(model_id, item.model_id)?;
        exprs.push(item.expr);
        texts.push(item.text);
    }
    Ok(PyConstraint { inner: ExprLike { model_id, expr: Expr::Or(exprs), text: format!("any({})", texts.join(", ")) } })
}

#[pyfunction]
fn implies(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
    let a = expr_from_py(a)?;
    let b = expr_from_py(b)?;
    let model_id = merge_model_ids(a.model_id, b.model_id)?;
    Ok(PyConstraint {
        inner: ExprLike { model_id, expr: Expr::Imp(Box::new(a.expr), Box::new(b.expr)), text: format!("({}) -> ({})", a.text, b.text) },
    })
}

#[pyfunction]
fn iff(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
    let a = expr_from_py(a)?;
    let b = expr_from_py(b)?;
    let model_id = merge_model_ids(a.model_id, b.model_id)?;
    Ok(PyConstraint {
        inner: ExprLike { model_id, expr: Expr::Iff(Box::new(a.expr), Box::new(b.expr)), text: format!("({}) <-> ({})", a.text, b.text) },
    })
}

#[pyfunction]
fn min_of(items: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    let items = expr_list_from_py(items)?;
    if items.is_empty() {
        return Err(PyValueError::new_err("min_of requires at least one expression"));
    }
    let mut model_id = None;
    let mut exprs = Vec::with_capacity(items.len());
    let mut texts = Vec::with_capacity(items.len());
    for item in items {
        model_id = merge_model_ids(model_id, item.model_id)?;
        exprs.push(item.expr);
        texts.push(item.text);
    }
    Ok(PyExpr { inner: ExprLike { model_id, expr: Expr::Min(exprs), text: format!("min({})", texts.join(", ")) } })
}

#[pyfunction]
fn max_of(items: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    let items = expr_list_from_py(items)?;
    if items.is_empty() {
        return Err(PyValueError::new_err("max_of requires at least one expression"));
    }
    let mut model_id = None;
    let mut exprs = Vec::with_capacity(items.len());
    let mut texts = Vec::with_capacity(items.len());
    for item in items {
        model_id = merge_model_ids(model_id, item.model_id)?;
        exprs.push(item.expr);
        texts.push(item.text);
    }
    Ok(PyExpr { inner: ExprLike { model_id, expr: Expr::Max(exprs), text: format!("max({})", texts.join(", ")) } })
}

#[pyfunction]
fn if_then_else(condition: &Bound<'_, PyAny>, then_expr: &Bound<'_, PyAny>, else_expr: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    let condition = expr_from_py(condition)?;
    let then_expr = expr_from_py(then_expr)?;
    let else_expr = expr_from_py(else_expr)?;
    let model_id = merge_model_ids(merge_model_ids(condition.model_id, then_expr.model_id)?, else_expr.model_id)?;
    Ok(PyExpr {
        inner: ExprLike {
            model_id,
            expr: Expr::IfThenElse(Box::new(condition.expr), Box::new(then_expr.expr), Box::new(else_expr.expr)),
            text: format!("if_then_else({}, {}, {})", condition.text, then_expr.text, else_expr.text),
        },
    })
}

#[pyfunction]
fn domain(values: Vec<i64>) -> PyResult<Vec<i32>> {
    values.into_iter().map(|value| checked_i32(value, "domain value")).collect()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyIntVar>()?;
    m.add_class::<PyExpr>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<PySolution>()?;
    m.add_class::<PySolveStats>()?;
    m.add_function(wrap_pyfunction!(expr_fn, m)?)?;
    m.add_function(wrap_pyfunction!(all_fn, m)?)?;
    m.add_function(wrap_pyfunction!(any_fn, m)?)?;
    m.add_function(wrap_pyfunction!(implies, m)?)?;
    m.add_function(wrap_pyfunction!(iff, m)?)?;
    m.add_function(wrap_pyfunction!(min_of, m)?)?;
    m.add_function(wrap_pyfunction!(max_of, m)?)?;
    m.add_function(wrap_pyfunction!(if_then_else, m)?)?;
    m.add_function(wrap_pyfunction!(domain, m)?)?;
    m.add("STAR", table::STAR)?;
    Ok(())
}
