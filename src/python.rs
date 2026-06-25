use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
use crate::collection;
use crate::expr::{self, Expr};
use crate::ids::VarId;
use crate::ls::{solve_fast_cop, LocalRhs, LocalSearchSpec, LsConfig};
use crate::problem::{Objective as ProblemObjective, Problem};
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

/// A list (set) decision variable handle: one of the lists the universe is
/// partitioned among. Carries its owning model and the generation of the
/// `list_vars` call that created it, so stale or cross-model use is rejected.
#[pyclass(name = "ListVar", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyListVar {
    model_id: u64,
    gen: u64,
    index: u32,
}

#[pymethods]
impl PyListVar {
    #[getter]
    fn index(&self) -> usize {
        self.index as usize
    }

    fn __repr__(&self) -> String {
        format!("ListVar({})", self.index)
    }
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
    /// List variable contents, set only for collection (list) models.
    routes: Option<Vec<Vec<i32>>>,
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
    /// Local-search model built in parallel with the CP posts, so `--turbo`-style
    /// LS (`solve(local_search=True)`) can run on the same model.
    local: LocalSearchSpec,
    /// Set once `list_vars` is called: the universe of items partitioned among
    /// the list variables. Its presence makes `solve()` dispatch to the
    /// collection engine instead of the integer CP/LS path.
    col_universe: Option<Vec<i32>>,
    col_lists: usize,
    /// Bumped on each `list_vars` call so route handles from an earlier call (or
    /// another model) are rejected instead of silently aliasing the wrong list.
    col_gen: u64,
    col_minimize: bool,
    col_objective: Vec<collection::Reduction>,
    col_constraints: Vec<collection::Constraint>,
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

#[allow(clippy::too_many_arguments)]
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
        routes: None,
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

    /// List variable contents (one list per list variable), or `None` for an
    /// integer model.
    #[getter]
    fn routes(&self) -> Option<Vec<Vec<i32>>> {
        self.routes.clone()
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
        Self {
            id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            solver: Solver::new(),
            names: Vec::new(),
            objective: None,
            local: LocalSearchSpec::default(),
            col_universe: None,
            col_lists: 0,
            col_gen: 0,
            col_minimize: true,
            col_objective: Vec::new(),
            col_constraints: Vec::new(),
        }
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
        self.local.add_var(id);
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
        // A term comparison is a constraint on the list it references.
        if let Ok(lc) = constraint.extract::<PyRef<'_, PyListConstraint>>() {
            self.check_term_scope(lc.model_id, lc.gen)?;
            self.col_constraints.push(collection::Constraint { reduction: lc.reduction.clone(), op: lc.op, rhs: lc.rhs });
            return Ok(());
        }
        if let Ok(constraint) = constraint_from_py(constraint) {
            if let Some(model_id) = constraint.inner.model_id {
                if model_id != self.id {
                    return Err(PyValueError::new_err("constraint belongs to a different model"));
                }
            }
            self.local.add_expr(constraint.inner.expr.clone());
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
        let rel = parse_relation(relation)?;
        self.local.add_linear(coeffs.clone(), vars.clone(), rel, rhs);
        linear::linear(&mut self.solver, &coeffs, &vars, rel, rhs);
        Ok(())
    }

    #[pyo3(signature = (vars, relation, rhs))]
    fn sum(&mut self, vars: &Bound<'_, PyAny>, relation: &str, rhs: i64) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        let rel = parse_relation(relation)?;
        self.local.add_linear(vec![1; vars.len()], vars.clone(), rel, rhs);
        linear::sum(&mut self.solver, &vars, rel, rhs);
        Ok(())
    }

    fn all_different(&mut self, vars: &Bound<'_, PyAny>) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        self.local.add_all_different(vars.clone());
        primitives::all_different(&mut self.solver, &vars);
        Ok(())
    }

    fn all_equal(&mut self, vars: &Bound<'_, PyAny>) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        self.local.add_all_equal(vars.clone());
        primitives::all_equal(&mut self.solver, &vars);
        Ok(())
    }

    fn not_equal(&mut self, x: &PyIntVar, y: &PyIntVar) -> PyResult<()> {
        let x = one_id_for(self.id, x)?;
        let y = one_id_for(self.id, y)?;
        self.local.add_expr(expr::ne(expr::var(x), expr::var(y)));
        primitives::not_equal(&mut self.solver, x, y);
        Ok(())
    }

    #[pyo3(signature = (x, y, offset=0))]
    fn not_equal_offset(&mut self, x: &PyIntVar, y: &PyIntVar, offset: i32) -> PyResult<()> {
        let x = one_id_for(self.id, x)?;
        let y = one_id_for(self.id, y)?;
        self.local.add_expr(expr::ne(expr::var(x), expr::add(vec![expr::var(y), expr::int(offset as i64)])));
        primitives::not_equal_offset(&mut self.solver, x, y, offset);
        Ok(())
    }

    fn ordered(&mut self, vars: &Bound<'_, PyAny>, relation: &str) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        let rel = parse_relation(relation)?;
        for pair in vars.windows(2) {
            self.local.add_linear(vec![1, -1], vec![pair[0], pair[1]], rel, 0);
        }
        primitives::ordered(&mut self.solver, &vars, rel);
        Ok(())
    }

    fn instantiate(&mut self, vars: &Bound<'_, PyAny>, values: Vec<i32>) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if vars.len() != values.len() {
            return Err(PyValueError::new_err("vars and values must have the same length"));
        }
        for (&var, &value) in vars.iter().zip(&values) {
            self.local.add_linear(vec![1], vec![var], Relation::Eq, value as i64);
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
        self.local.add_extremum(vars.clone(), true, Relation::Eq, LocalRhs::Var(target));
        primitives::minimum(&mut self.solver, target, &vars);
        Ok(())
    }

    fn maximum(&mut self, target: &PyIntVar, vars: &Bound<'_, PyAny>) -> PyResult<()> {
        let target = one_id_for(self.id, target)?;
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if vars.is_empty() {
            return Err(PyValueError::new_err("maximum requires at least one variable"));
        }
        self.local.add_extremum(vars.clone(), false, Relation::Eq, LocalRhs::Var(target));
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
        self.local.add_element(array.clone(), index, value, 0);
        primitives::element(&mut self.solver, &array, index, value);
        Ok(())
    }

    fn element_const(&mut self, array: Vec<i32>, index: &PyIntVar, value: &PyIntVar) -> PyResult<()> {
        if array.is_empty() {
            return Err(PyValueError::new_err("element_const requires a non-empty array"));
        }
        let index = one_id_for(self.id, index)?;
        let value = one_id_for(self.id, value)?;
        // LS has no constant-array element; model it as element over fixed vars.
        let const_vars: Vec<VarId> = array.iter().map(|&v| self.const_var(v)).collect();
        self.local.add_element(const_vars, index, value, 0);
        primitives::element_const(&mut self.solver, &array, index, value);
        Ok(())
    }

    fn count(&mut self, vars: &Bound<'_, PyAny>, value: i32, relation: &str, k: i64) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        let rel = parse_relation(relation)?;
        self.local.add_count(vars.clone(), vec![value], rel, LocalRhs::Const(k));
        count::count(&mut self.solver, &vars, value, rel, k);
        Ok(())
    }

    #[pyo3(signature = (vars, values, low, high, *, closed=false))]
    fn cardinality(&mut self, vars: &Bound<'_, PyAny>, values: Vec<i32>, low: Vec<i64>, high: Vec<i64>, closed: bool) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if values.len() != low.len() || values.len() != high.len() {
            return Err(PyValueError::new_err("values, low, and high must have the same length"));
        }
        self.local.add_cardinality(vars.clone(), values.clone(), low.clone(), high.clone(), closed);
        count::cardinality(&mut self.solver, &vars, &values, &low, &high, closed);
        Ok(())
    }

    fn n_values(&mut self, vars: &Bound<'_, PyAny>, relation: &str, k: i64) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        let rel = parse_relation(relation)?;
        self.local.add_n_values(vars.clone(), rel, LocalRhs::Const(k));
        count::n_values(&mut self.solver, &vars, rel, k);
        Ok(())
    }

    #[pyo3(signature = (vars, tuples, *, positive=true))]
    fn table(&mut self, vars: &Bound<'_, PyAny>, tuples: Vec<Vec<i32>>, positive: bool) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if tuples.iter().any(|tuple| tuple.len() != vars.len()) {
            return Err(PyValueError::new_err("every tuple must match the variable arity"));
        }
        self.local.add_extension(vars.clone(), tuples.clone(), positive);
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
        self.local.add_lex_chain(vec![x.clone(), y.clone()], strict);
        lex::lex(&mut self.solver, &x, &y, strict);
        Ok(())
    }

    #[pyo3(signature = (rows, *, strict=false))]
    fn lex_chain(&mut self, rows: &Bound<'_, PyAny>, strict: bool) -> PyResult<()> {
        let rows = var_matrix_from_py(rows)?;
        let rows: Vec<Vec<VarId>> = rows.iter().map(|row| ids_for(self.id, row)).collect::<PyResult<_>>()?;
        self.local.add_lex_chain(rows.clone(), strict);
        lex::lex_chain(&mut self.solver, &rows, strict);
        Ok(())
    }

    fn channel(&mut self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let x = ids_for(self.id, &var_list_from_py(x)?)?;
        let y = ids_for(self.id, &var_list_from_py(y)?)?;
        if x.len() != y.len() {
            return Err(PyValueError::new_err("channel vectors must have the same length"));
        }
        self.local.mark_unsupported();
        lex::channel(&mut self.solver, &x, &y);
        Ok(())
    }

    fn no_overlap(&mut self, starts: &Bound<'_, PyAny>, durations: Vec<i64>) -> PyResult<()> {
        let starts = ids_for(self.id, &var_list_from_py(starts)?)?;
        if starts.len() != durations.len() {
            return Err(PyValueError::new_err("starts and durations must have the same length"));
        }
        let origins: Vec<Vec<VarId>> = starts.iter().map(|&s| vec![s]).collect();
        let lengths: Vec<Vec<Expr>> = durations.iter().map(|&d| vec![expr::int(d)]).collect();
        self.local.add_no_overlap(origins, lengths, false);
        scheduling::no_overlap(&mut self.solver, &starts, &durations);
        Ok(())
    }

    fn cumulative(&mut self, starts: &Bound<'_, PyAny>, durations: Vec<i64>, heights: Vec<i64>, capacity: i64) -> PyResult<()> {
        let starts = ids_for(self.id, &var_list_from_py(starts)?)?;
        if starts.len() != durations.len() || starts.len() != heights.len() {
            return Err(PyValueError::new_err("starts, durations, and heights must have the same length"));
        }
        // LS cumulative wants per-task duration/height vars; pin constants as fixed vars.
        let dur_vars: Vec<VarId> = durations.iter().map(|&d| self.const_var(d as i32)).collect();
        let height_vars: Vec<VarId> = heights.iter().map(|&h| self.const_var(h as i32)).collect();
        self.local.add_cumulative(starts.clone(), dur_vars, height_vars, LocalRhs::Const(capacity));
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
        self.local.add_cumulative(starts.clone(), durations.clone(), heights.clone(), LocalRhs::Var(capacity));
        scheduling::cumulative_var(&mut self.solver, &starts, &durations, &heights, capacity);
        Ok(())
    }

    fn bin_packing(&mut self, items: &Bound<'_, PyAny>, sizes: Vec<i64>, capacities: Vec<i64>) -> PyResult<()> {
        let items = ids_for(self.id, &var_list_from_py(items)?)?;
        if items.len() != sizes.len() {
            return Err(PyValueError::new_err("items and sizes must have the same length"));
        }
        let limits: Vec<LocalRhs> = capacities.iter().map(|&c| LocalRhs::Const(c)).collect();
        self.local.add_bin_packing(items.clone(), sizes.clone(), limits, false);
        scheduling::bin_packing(&mut self.solver, &items, &sizes, &capacities);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
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
        self.local.mark_unsupported();
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
        self.local.add_circuit(successors.clone());
        graph::circuit(&mut self.solver, &successors);
        Ok(())
    }

    /// Declare `k` list variables that partition `universe` among them. Their
    /// presence switches `solve()` to the collection engine. Objective and
    /// constraints are added as reductions (e.g. `route_cost`, `list_sum`).
    #[pyo3(signature = (k, universe))]
    fn list_vars(&mut self, k: usize, universe: Vec<i32>) -> PyResult<Vec<PyListVar>> {
        if universe.is_empty() {
            return Err(PyValueError::new_err("list universe cannot be empty"));
        }
        if k == 0 {
            return Err(PyValueError::new_err("need at least one list"));
        }
        if !self.names.is_empty() || self.objective.is_some() {
            return Err(PyValueError::new_err("cannot mix integer variables with list_vars; use one modeling style per model"));
        }
        // The universe is a set partitioned among the lists; duplicate ids would
        // be independent positions that share a value, which silently diverges
        // from set semantics (and from how reductions aggregate by value).
        let mut seen = std::collections::HashSet::with_capacity(universe.len());
        if let Some(dup) = universe.iter().find(|v| !seen.insert(**v)) {
            return Err(PyValueError::new_err(format!("list universe has a duplicate item {dup}; items must be distinct")));
        }
        self.col_universe = Some(universe);
        self.col_lists = k;
        self.col_gen += 1;
        // Reset any reductions recorded against a previous `list_vars` generation.
        self.col_objective.clear();
        self.col_constraints.clear();
        let gen = self.col_gen;
        Ok((0..k).map(|i| PyListVar { model_id: self.id, gen, index: i as u32 }).collect())
    }

    #[pyo3(signature = (objective, *, sense="min"))]
    fn objective(&mut self, objective: &Bound<'_, PyAny>, sense: &str) -> PyResult<()> {
        // A term objective targets the list (collection) model.
        if let Ok(term) = objective.extract::<PyRef<'_, PyTerm>>() {
            self.check_term_scope(term.model_id, term.gen)?;
            self.col_minimize = parse_objective_sense(sense)?;
            // Replace, matching the integer path: a second objective() call
            // restates the objective rather than silently summing onto the first.
            self.col_objective.clear();
            self.col_objective.extend(term.reductions.iter().cloned());
            return Ok(());
        }
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

    #[pyo3(signature = (*, search=None, verbose=false, local_search=false, time_limit=None, seed=0))]
    fn solve(
        &self,
        search: Option<&Bound<'_, PyAny>>,
        verbose: bool,
        local_search: bool,
        time_limit: Option<u64>,
        seed: u64,
    ) -> PyResult<PySolution> {
        // List variables present: the model is a collection model, solved by the
        // collection engine regardless of the integer-path options.
        if self.col_universe.is_some() {
            if !self.names.is_empty() || self.objective.is_some() {
                return Err(PyValueError::new_err("model mixes integer variables and list variables; use one modeling style per model"));
            }
            return self.solve_collection(time_limit, seed, verbose);
        }
        if local_search {
            let Some(objective) = &self.objective else {
                return Err(PyValueError::new_err("local_search=True requires an objective (call model.objective(...))"));
            };
            return self.solve_local_search(objective, search, verbose, time_limit, seed);
        }
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
    /// Create a fixed (single-value) variable and register it with both the CP
    /// solver and the LS spec. Used to express constant arrays/durations in the LS
    /// model, which only takes variable operands.
    fn const_var(&mut self, value: i32) -> VarId {
        let id = self.solver.new_var_set(&[value]);
        self.names.push(None);
        self.local.add_var(id);
        id
    }

    /// Solve the recorded collection model (list variables + reductions) with the
    /// collection local-search engine, time-limited (default 5s).
    /// Reject a term that belongs to a different model or to a superseded
    /// `list_vars` generation.
    fn check_term_scope(&self, model_id: u64, gen: u64) -> PyResult<()> {
        if model_id != self.id {
            return Err(PyValueError::new_err("this term/route belongs to a different model"));
        }
        if gen != self.col_gen {
            return Err(PyValueError::new_err("this term/route is stale; rebuild it from the current list_vars()"));
        }
        Ok(())
    }

    fn solve_collection(&self, time_limit: Option<u64>, seed: u64, verbose: bool) -> PyResult<PySolution> {
        let model = collection::CollectionModel {
            items: self.col_universe.clone().unwrap_or_default(),
            lists: self.col_lists,
            minimize: self.col_minimize,
            objective: self.col_objective.clone(),
            constraints: self.col_constraints.clone(),
        };
        model.validate().map_err(PyValueError::new_err)?;
        let limit = time_limit.unwrap_or(5);
        if verbose {
            println!("qayd solve (collection)");
            println!("  items: {}", model.items.len());
            println!("  lists: {}", model.lists);
            println!("  constraints: {}", model.constraints.len());
            println!("  objective: {}", if model.objective.is_empty() { "no" } else { "yes" });
            println!("  time limit: {limit}s");
        }
        let stop = Arc::new(AtomicBool::new(false));
        {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(limit));
                stop.store(true, Ordering::SeqCst);
            });
        }
        let sense = if self.col_minimize { "min" } else { "max" };
        let start = Instant::now();
        let mut improvements = 0u64;
        let mut report = |objective: i64| {
            if verbose {
                improvements += 1;
                println!("  o {objective}  ({sense}, {:.2}s)", start.elapsed().as_secs_f64());
            }
        };
        let sol = collection::solve_collection(&model, seed, &stop, &mut report);
        if verbose {
            println!("qayd result (collection)");
            println!("  status: {}", if sol.feasible { "SATISFIABLE" } else { "UNKNOWN" });
            if sol.feasible {
                println!("  objective: {}", sol.objective);
            }
            println!("  improvements: {improvements}");
        }
        Ok(PySolution {
            status: if sol.feasible { "SATISFIABLE".to_string() } else { "UNKNOWN".to_string() },
            // Only expose the objective and routes for a feasible incumbent. An
            // infeasible best-effort partition may violate constraints, so hiding
            // it stops a caller from accidentally consuming an invalid solution.
            objective: sol.feasible.then_some(sol.objective),
            objective_sense: Some(sense.to_string()),
            objective_expr: None,
            values: Vec::new(),
            stats: SolveStats::default().into(),
            routes: sol.feasible.then_some(sol.lists),
        })
    }

    /// Solve a COP with the local-search engine (`solve_fast_cop`) - the same
    /// incumbent-only LS that powers `--turbo`. Requires an objective and a time
    /// limit (defaults to 10s, since LS never terminates on its own).
    fn solve_local_search(
        &self,
        objective: &ObjectiveSpec,
        search: Option<&Bound<'_, PyAny>>,
        verbose: bool,
        time_limit: Option<u64>,
        seed: u64,
    ) -> PyResult<PySolution> {
        if let Some(model_id) = objective.expr.model_id {
            if model_id != self.id {
                return Err(PyValueError::new_err("objective belongs to a different model"));
            }
        }
        let mut vars = search_ids(self, search, None)?;
        append_expr_vars(&mut vars, &objective.expr.expr);
        if verbose {
            verbose_start(self.names.len(), self.solver.num_propagators(), true);
            println!("  direction: {}", if objective.minimizing { "min" } else { "max" });
            println!("  expression: {}", objective.expr.text);
            println!("  engine: local-search");
        }
        let problem = Problem {
            solver: self.solver.clone(),
            search: vars.clone(),
            objective: Some(ProblemObjective::Expr(objective.minimizing, objective.expr.expr.clone())),
        };
        let stop = Arc::new(AtomicBool::new(false));
        let limit = time_limit.unwrap_or(10);
        {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(limit));
                stop.store(true, Ordering::SeqCst);
            });
        }
        let config = LsConfig { gls: true, min_conflicts: true, kick_bandit: false };
        let outcome = solve_fast_cop(problem, self.local.clone(), &stop, seed, config, |value, _solution, _source| {
            if verbose {
                println!("  incumbent: {value}");
            }
        });
        let sense = if objective.minimizing { "min" } else { "max" };
        let solution = match outcome.best {
            Some((assignment, objective_value)) => make_solution(
                "SATISFIABLE",
                &vars,
                Some(&assignment),
                Some(objective_value),
                Some(sense),
                Some(&objective.expr.text),
                SolveStats::default(),
                self.names.len(),
            ),
            None => make_solution("UNKNOWN", &vars, None, None, Some(sense), Some(&objective.expr.text), SolveStats::default(), self.names.len()),
        };
        if verbose {
            verbose_finish(&solution);
        }
        Ok(solution)
    }

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

/// A node of a lambda body, built by the Python lambda at model-construction
/// time. Held as an `Arc` tree so subexpressions and constant tables are shared
/// rather than copied. Never executed as Python during solving; it is lowered
/// to a [`collection::ExprArena`] when the term joins the model.
enum PyNode {
    Const(i64),
    Arg(u8),
    Array(Arc<Vec<i64>>, Arc<PyNode>),
    Matrix(Arc<Vec<Vec<i64>>>, Arc<PyNode>, Arc<PyNode>),
    Add(Arc<PyNode>, Arc<PyNode>),
    Sub(Arc<PyNode>, Arc<PyNode>),
    Mul(Arc<PyNode>, Arc<PyNode>),
}

/// A symbolic lambda-body expression. Arithmetic operators build a bigger tree;
/// indexing an `Array`/`Matrix` with one of these builds a table lookup.
#[pyclass(name = "LambdaExpr", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyLambdaExpr {
    node: Arc<PyNode>,
}

fn node(n: PyNode) -> PyLambdaExpr {
    PyLambdaExpr { node: Arc::new(n) }
}

/// Coerce a Python value used in a lambda body to a node: a lambda expression
/// stays, an integer becomes a constant.
fn coerce_node(obj: &Bound<'_, PyAny>) -> PyResult<Arc<PyNode>> {
    if let Ok(e) = obj.extract::<PyRef<'_, PyLambdaExpr>>() {
        return Ok(e.node.clone());
    }
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(Arc::new(PyNode::Const(v)));
    }
    Err(PyTypeError::new_err("a lambda body may only combine lambda expressions and integers"))
}

#[pymethods]
impl PyLambdaExpr {
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Add(self.node.clone(), coerce_node(other)?)))
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Add(coerce_node(other)?, self.node.clone())))
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Sub(self.node.clone(), coerce_node(other)?)))
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Sub(coerce_node(other)?, self.node.clone())))
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Mul(self.node.clone(), coerce_node(other)?)))
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Mul(coerce_node(other)?, self.node.clone())))
    }
    fn __neg__(&self) -> PyLambdaExpr {
        node(PyNode::Sub(Arc::new(PyNode::Const(0)), self.node.clone()))
    }
}

/// A constant integer array; index it with a lambda arg to read `array[i]`.
#[pyclass(name = "Array", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyArray {
    data: Arc<Vec<i64>>,
}

#[pymethods]
impl PyArray {
    fn __getitem__(&self, idx: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Array(self.data.clone(), coerce_node(idx)?)))
    }
    fn __len__(&self) -> usize {
        self.data.len()
    }
}

/// A constant integer matrix; `matrix[i][j]` reads a cell.
#[pyclass(name = "Matrix", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyMatrix {
    data: Arc<Vec<Vec<i64>>>,
}

#[pymethods]
impl PyMatrix {
    fn __getitem__(&self, row: &Bound<'_, PyAny>) -> PyResult<PyMatrixRow> {
        Ok(PyMatrixRow { data: self.data.clone(), row: coerce_node(row)? })
    }
}

/// The partially indexed `matrix[i]`; index again to get `matrix[i][j]`.
#[pyclass(name = "MatrixRow", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyMatrixRow {
    data: Arc<Vec<Vec<i64>>>,
    row: Arc<PyNode>,
}

#[pymethods]
impl PyMatrixRow {
    fn __getitem__(&self, col: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Matrix(self.data.clone(), self.row.clone(), coerce_node(col)?)))
    }
}

/// A term over one or more list variables: a sum of reductions. Built by the
/// reduction operators (`sum`, `sum_edges`, ...), added together for an
/// objective, and compared to an integer for a constraint.
#[pyclass(name = "Term", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyTerm {
    model_id: u64,
    gen: u64,
    reductions: Vec<collection::Reduction>,
}

/// A constraint `term <op> rhs` over a single list reduction.
#[pyclass(name = "ListConstraint", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyListConstraint {
    model_id: u64,
    gen: u64,
    reduction: collection::Reduction,
    op: collection::Op,
    rhs: i64,
}

/// Sum two terms, rejecting a mix of models or `list_vars` generations.
fn combine_terms(a: &PyTerm, b: &PyTerm) -> PyResult<PyTerm> {
    if a.model_id != b.model_id || a.gen != b.gen {
        return Err(PyValueError::new_err("cannot combine terms from different models or list_vars generations"));
    }
    let mut reductions = a.reductions.clone();
    reductions.extend(b.reductions.iter().cloned());
    Ok(PyTerm { model_id: a.model_id, gen: a.gen, reductions })
}

#[pymethods]
impl PyTerm {
    fn __add__(&self, other: PyRef<'_, PyTerm>) -> PyResult<PyTerm> {
        combine_terms(self, &other)
    }

    /// Support Python's `sum(...)`, which starts the accumulation at integer 0.
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
        if other.extract::<i64>().is_ok_and(|v| v == 0) {
            return Ok(self.clone());
        }
        Err(PyTypeError::new_err("a term can only be added to another term"))
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<PyListConstraint> {
        let rhs = other.extract::<i64>().map_err(|_| PyTypeError::new_err("a term can only be compared to an integer bound"))?;
        let op = match op {
            CompareOp::Le => collection::Op::Le,
            CompareOp::Ge => collection::Op::Ge,
            CompareOp::Eq => collection::Op::Eq,
            _ => return Err(PyValueError::new_err("a term supports only <=, >=, ==")),
        };
        if self.reductions.len() != 1 {
            return Err(PyValueError::new_err("a constraint must be a single reduction over one list, not a sum of terms"));
        }
        Ok(PyListConstraint { model_id: self.model_id, gen: self.gen, reduction: self.reductions[0].clone(), op, rhs })
    }
}

/// Lower a Python lambda-body tree into a reduction's flat expression arena.
fn lower(n: &PyNode, arena: &mut collection::ExprArena) -> collection::ExprId {
    match n {
        PyNode::Const(c) => arena.constant(*c),
        PyNode::Arg(k) => arena.arg(*k),
        PyNode::Array(a, i) => {
            let ie = lower(i, arena);
            arena.array(a.clone(), ie)
        }
        PyNode::Matrix(m, i, j) => {
            let ie = lower(i, arena);
            let je = lower(j, arena);
            arena.matrix(m.clone(), ie, je)
        }
        PyNode::Add(a, b) => {
            let x = lower(a, arena);
            let y = lower(b, arena);
            arena.add(x, y)
        }
        PyNode::Sub(a, b) => {
            let x = lower(a, arena);
            let y = lower(b, arena);
            arena.sub(x, y)
        }
        PyNode::Mul(a, b) => {
            let x = lower(a, arena);
            let y = lower(b, arena);
            arena.mul(x, y)
        }
    }
}

fn single_term(route: &PyListVar, reduction: collection::Reduction) -> PyTerm {
    PyTerm { model_id: route.model_id, gen: route.gen, reductions: vec![reduction] }
}

/// Build a per-item reduction `op(route, i => body)` from a Python lambda.
fn build_items_reduction(route: &PyListVar, op: collection::ReduceOp, func: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
    let body = coerce_node(&func.call1((node(PyNode::Arg(0)),))?)?;
    let mut arena = collection::ExprArena::default();
    let body_id = lower(&body, &mut arena);
    Ok(single_term(route, collection::Reduction { op, iterable: collection::Iterable::Items(route.index as usize), arena, body: body_id }))
}

/// `sum(route, i => body)`, or `sum(terms)` to add a collection of terms.
#[pyfunction]
#[pyo3(signature = (arg, func=None))]
fn sum(arg: &Bound<'_, PyAny>, func: Option<&Bound<'_, PyAny>>) -> PyResult<PyTerm> {
    if let Some(f) = func {
        let route = arg
            .extract::<PyRef<'_, PyListVar>>()
            .map_err(|_| PyTypeError::new_err("sum(route, lambda): the first argument must be a list variable"))?;
        return build_items_reduction(&route, collection::ReduceOp::Sum, f);
    }
    let mut acc: Option<PyTerm> = None;
    for item in arg.try_iter()? {
        let t = item?
            .extract::<PyRef<'_, PyTerm>>()
            .map_err(|_| PyTypeError::new_err("sum(iterable) expects an iterable of terms"))?;
        acc = Some(match acc {
            None => t.clone(),
            Some(a) => combine_terms(&a, &t)?,
        });
    }
    acc.ok_or_else(|| PyValueError::new_err("sum got no terms to add"))
}

/// `min(route, i => body)` over a route's items (undefined, hence infeasible,
/// for an empty route).
#[pyfunction]
fn minimum(route: &PyListVar, func: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
    build_items_reduction(route, collection::ReduceOp::Min, func)
}

/// `max(route, i => body)` over a route's items.
#[pyfunction]
fn maximum(route: &PyListVar, func: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
    build_items_reduction(route, collection::ReduceOp::Max, func)
}

/// `count(route, i => predicate)`: items whose body is non-zero. With no lambda,
/// the route's length.
#[pyfunction(name = "count")]
#[pyo3(signature = (route, func=None))]
fn count_reduction(route: &PyListVar, func: Option<&Bound<'_, PyAny>>) -> PyResult<PyTerm> {
    match func {
        Some(f) => build_items_reduction(route, collection::ReduceOp::Count, f),
        None => {
            let mut arena = collection::ExprArena::default();
            let body = arena.constant(1);
            Ok(single_term(route, collection::Reduction { op: collection::ReduceOp::Count, iterable: collection::Iterable::Items(route.index as usize), arena, body }))
        }
    }
}

/// `sum_edges(route, (i, j) => body, start=, end=)`: sum the body over the edges
/// of the closed tour `[start, items.., end]`.
#[pyfunction]
#[pyo3(signature = (route, func, *, start=0, end=0))]
fn sum_edges(route: &PyListVar, func: &Bound<'_, PyAny>, start: i32, end: i32) -> PyResult<PyTerm> {
    let body = coerce_node(&func.call1((node(PyNode::Arg(0)), node(PyNode::Arg(1))))?)?;
    let mut arena = collection::ExprArena::default();
    let body_id = lower(&body, &mut arena);
    let iterable = collection::Iterable::Edges { list: route.index as usize, start, end };
    Ok(single_term(route, collection::Reduction { op: collection::ReduceOp::Sum, iterable, arena, body: body_id }))
}

/// Wrap a constant integer array / matrix for use inside lambdas.
#[pyfunction]
fn array(data: Vec<i64>) -> PyArray {
    PyArray { data: Arc::new(data) }
}

#[pyfunction]
fn matrix(data: Vec<Vec<i64>>) -> PyMatrix {
    PyMatrix { data: Arc::new(data) }
}

// --- convenience builders for the common raw-array cases ---

/// Closed-tour cost of `route`: `sum_edges(route, (i, j) => matrix[i][j])` with
/// `depot` at both ends.
#[pyfunction]
#[pyo3(signature = (route, matrix, *, depot=0))]
fn route_cost(route: &PyListVar, matrix: Vec<Vec<i64>>, depot: i32) -> PyTerm {
    let mut arena = collection::ExprArena::default();
    let i = arena.arg(0);
    let j = arena.arg(1);
    let body = arena.matrix(Arc::new(matrix), i, j);
    let iterable = collection::Iterable::Edges { list: route.index as usize, start: depot, end: depot };
    single_term(route, collection::Reduction { op: collection::ReduceOp::Sum, iterable, arena, body })
}

fn over_items_raw(route: &PyListVar, op: collection::ReduceOp, values: Vec<i64>) -> PyTerm {
    let mut arena = collection::ExprArena::default();
    let i = arena.arg(0);
    let body = arena.array(Arc::new(values), i);
    single_term(route, collection::Reduction { op, iterable: collection::Iterable::Items(route.index as usize), arena, body })
}

/// Sum of `values[item]` over `route`'s items.
#[pyfunction]
fn list_sum(route: &PyListVar, values: Vec<i64>) -> PyTerm {
    over_items_raw(route, collection::ReduceOp::Sum, values)
}

/// Minimum / maximum of `values[item]` over `route`'s items.
#[pyfunction]
fn list_min(route: &PyListVar, values: Vec<i64>) -> PyTerm {
    over_items_raw(route, collection::ReduceOp::Min, values)
}

#[pyfunction]
fn list_max(route: &PyListVar, values: Vec<i64>) -> PyTerm {
    over_items_raw(route, collection::ReduceOp::Max, values)
}

/// Number of items in `route`.
#[pyfunction]
fn list_count(route: &PyListVar) -> PyTerm {
    let mut arena = collection::ExprArena::default();
    let body = arena.constant(1);
    single_term(route, collection::Reduction { op: collection::ReduceOp::Count, iterable: collection::Iterable::Items(route.index as usize), arena, body })
}


#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyIntVar>()?;
    m.add_class::<PyListVar>()?;
    m.add_class::<PyTerm>()?;
    m.add_class::<PyListConstraint>()?;
    m.add_class::<PyExpr>()?;
    m.add_class::<PyLambdaExpr>()?;
    m.add_class::<PyArray>()?;
    m.add_class::<PyMatrix>()?;
    m.add_class::<PyMatrixRow>()?;
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
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(matrix, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(minimum, m)?)?;
    m.add_function(wrap_pyfunction!(maximum, m)?)?;
    m.add_function(wrap_pyfunction!(count_reduction, m)?)?;
    m.add_function(wrap_pyfunction!(sum_edges, m)?)?;
    m.add_function(wrap_pyfunction!(route_cost, m)?)?;
    m.add_function(wrap_pyfunction!(list_sum, m)?)?;
    m.add_function(wrap_pyfunction!(list_min, m)?)?;
    m.add_function(wrap_pyfunction!(list_max, m)?)?;
    m.add_function(wrap_pyfunction!(list_count, m)?)?;
    m.add("STAR", table::STAR)?;
    Ok(())
}
