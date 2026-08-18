use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, PyTuple};

use crate::model::list;

/// Symbolic syntax built by calling a Python lambda during model construction.
/// It is lowered once into the canonical list expression arena and never calls
/// Python while the solver is searching.
enum Node {
    Const(i64),
    Arg(u8),
    Array(Arc<Vec<i64>>, Arc<Node>),
    Matrix(Arc<Vec<Vec<i64>>>, Arc<Node>, Arc<Node>),
    Add(Arc<Node>, Arc<Node>),
    Sub(Arc<Node>, Arc<Node>),
    Mul(Arc<Node>, Arc<Node>),
    Mod(Arc<Node>, Arc<Node>),
    Pow(Arc<Node>, u32),
    MulScaled(Arc<Node>, Arc<Node>, i64),
    DivScaled(Arc<Node>, Arc<Node>, i64),
    Min(Arc<Node>, Arc<Node>),
    Max(Arc<Node>, Arc<Node>),
    Div(Arc<Node>, Arc<Node>),
    Abs(Arc<Node>),
    Lt(Arc<Node>, Arc<Node>),
    Le(Arc<Node>, Arc<Node>),
    Eq(Arc<Node>, Arc<Node>),
    Ne(Arc<Node>, Arc<Node>),
    IfThenElse(Arc<Node>, Arc<Node>, Arc<Node>),
    PiecewiseLinear(Arc<Node>, Arc<Vec<(i64, i64)>>),
    External(Arc<str>, Arc<Vec<Arc<Node>>>),
}

#[pyclass(name = "LambdaExpr", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyLambdaExpr {
    node: Arc<Node>,
}

fn expression(node: Node) -> PyLambdaExpr {
    PyLambdaExpr { node: Arc::new(node) }
}

fn coerce(obj: &Bound<'_, PyAny>) -> PyResult<Arc<Node>> {
    if let Ok(expression) = obj.extract::<PyRef<'_, PyLambdaExpr>>() {
        return Ok(Arc::clone(&expression.node));
    }
    if let Ok(value) = obj.extract::<i64>() {
        return Ok(Arc::new(Node::Const(value)));
    }
    Err(PyTypeError::new_err("a lambda body may only combine lambda expressions and integers"))
}

#[pymethods]
impl PyLambdaExpr {
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Add(Arc::clone(&self.node), coerce(other)?)))
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Add(coerce(other)?, Arc::clone(&self.node))))
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Sub(Arc::clone(&self.node), coerce(other)?)))
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Sub(coerce(other)?, Arc::clone(&self.node))))
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Mul(Arc::clone(&self.node), coerce(other)?)))
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Mul(coerce(other)?, Arc::clone(&self.node))))
    }

    fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Mod(Arc::clone(&self.node), coerce(other)?)))
    }

    fn __rmod__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Mod(coerce(other)?, Arc::clone(&self.node))))
    }

    fn __pow__(&self, exponent: u32, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<PyLambdaExpr> {
        if modulo.is_some() {
            return Err(PyValueError::new_err("modular power is not supported; use (x ** n) % m"));
        }
        Ok(expression(Node::Pow(Arc::clone(&self.node), exponent)))
    }

    fn __neg__(&self) -> PyLambdaExpr {
        expression(Node::Sub(Arc::new(Node::Const(0)), Arc::clone(&self.node)))
    }

    fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Div(Arc::clone(&self.node), coerce(other)?)))
    }

    fn __rfloordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Div(coerce(other)?, Arc::clone(&self.node))))
    }

    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Lt(Arc::clone(&self.node), coerce(other)?)))
    }

    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Le(Arc::clone(&self.node), coerce(other)?)))
    }

    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Lt(coerce(other)?, Arc::clone(&self.node))))
    }

    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Le(coerce(other)?, Arc::clone(&self.node))))
    }
}

#[pyclass(name = "Array", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyArray {
    data: Arc<Vec<i64>>,
}

#[pymethods]
impl PyArray {
    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Array(Arc::clone(&self.data), coerce(index)?)))
    }

    fn __len__(&self) -> usize {
        self.data.len()
    }
}

#[pyclass(name = "Matrix", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyMatrix {
    data: Arc<Vec<Vec<i64>>>,
}

#[pymethods]
impl PyMatrix {
    fn __getitem__(&self, row: &Bound<'_, PyAny>) -> PyResult<PyMatrixRow> {
        Ok(PyMatrixRow { data: Arc::clone(&self.data), row: coerce(row)? })
    }
}

#[pyclass(name = "MatrixRow", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyMatrixRow {
    data: Arc<Vec<Vec<i64>>>,
    row: Arc<Node>,
}

#[pymethods]
impl PyMatrixRow {
    fn __getitem__(&self, column: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(expression(Node::Matrix(Arc::clone(&self.data), Arc::clone(&self.row), coerce(column)?)))
    }
}

/// Invoke a Python modeling lambda with the requested semantic argument slots,
/// then append its lowered body to `arena`. Callers can compile related bodies
/// into one arena by invoking this function repeatedly on the same value.
pub(super) fn compile_callable(callable: &Bound<'_, PyAny>, slots: &[u8], arena: &mut list::ExprArena) -> PyResult<list::ExprId> {
    let py = callable.py();
    let arguments = slots.iter().map(|&slot| Py::new(py, expression(Node::Arg(slot)))).collect::<PyResult<Vec<_>>>()?;
    let arguments = PyTuple::new(py, arguments)?;
    let body = callable.call1(arguments)?;
    let body = coerce(&body)?;
    Ok(lower(&body, arena))
}

fn lower(node: &Node, arena: &mut list::ExprArena) -> list::ExprId {
    match node {
        Node::Const(value) => arena.constant(*value),
        Node::Arg(index) => arena.arg(*index),
        Node::Array(values, index) => {
            let index = lower(index, arena);
            arena.array(Arc::clone(values), index)
        }
        Node::Matrix(values, row, column) => {
            let row = lower(row, arena);
            let column = lower(column, arena);
            arena.matrix(Arc::clone(values), row, column)
        }
        Node::Add(left, right) => {
            let left = lower(left, arena);
            let right = lower(right, arena);
            arena.add(left, right)
        }
        Node::Sub(left, right) => {
            let left = lower(left, arena);
            let right = lower(right, arena);
            arena.sub(left, right)
        }
        Node::Mul(left, right) => {
            let left = lower(left, arena);
            let right = lower(right, arena);
            arena.mul(left, right)
        }
        Node::Mod(left, right) => {
            let left = lower(left, arena);
            let right = lower(right, arena);
            arena.modulo(left, right)
        }
        Node::Pow(base, exponent) => {
            let base = lower(base, arena);
            arena.pow(base, *exponent)
        }
        Node::MulScaled(left, right, scale) => {
            let left = lower(left, arena);
            let right = lower(right, arena);
            arena.mul_scaled(left, right, *scale)
        }
        Node::DivScaled(left, right, scale) => {
            let left = lower(left, arena);
            let right = lower(right, arena);
            arena.div_scaled(left, right, *scale)
        }
        Node::Min(left, right) => {
            let left = lower(left, arena);
            let right = lower(right, arena);
            arena.min(left, right)
        }
        Node::Max(left, right) => {
            let left = lower(left, arena);
            let right = lower(right, arena);
            arena.max(left, right)
        }
        Node::Div(left, right) => {
            let left = lower(left, arena);
            let right = lower(right, arena);
            arena.div(left, right)
        }
        Node::Abs(value) => {
            let value = lower(value, arena);
            arena.abs(value)
        }
        Node::Lt(left, right) => {
            let left = lower(left, arena);
            let right = lower(right, arena);
            arena.lt(left, right)
        }
        Node::Le(left, right) => {
            let left = lower(left, arena);
            let right = lower(right, arena);
            arena.le(left, right)
        }
        Node::Eq(left, right) => {
            let left = lower(left, arena);
            let right = lower(right, arena);
            arena.eq(left, right)
        }
        Node::Ne(left, right) => {
            let left = lower(left, arena);
            let right = lower(right, arena);
            arena.ne(left, right)
        }
        Node::IfThenElse(condition, then_value, else_value) => {
            let condition = lower(condition, arena);
            let then_value = lower(then_value, arena);
            let else_value = lower(else_value, arena);
            arena.if_then_else(condition, then_value, else_value)
        }
        Node::PiecewiseLinear(input, points) => {
            let input = lower(input, arena);
            arena.piecewise_linear(input, Arc::clone(points))
        }
        Node::External(name, arguments) => {
            let arguments = arguments.iter().map(|argument| lower(argument, arena)).collect();
            arena.external(Arc::clone(name), arguments)
        }
    }
}

#[pyfunction(name = "min")]
fn min_expr(left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
    Ok(expression(Node::Min(coerce(left)?, coerce(right)?)))
}

#[pyfunction(name = "max")]
fn max_expr(left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
    Ok(expression(Node::Max(coerce(left)?, coerce(right)?)))
}

#[pyfunction(name = "abs")]
fn abs_expr(value: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
    Ok(expression(Node::Abs(coerce(value)?)))
}

#[pyfunction(name = "if_")]
fn if_expr(condition: &Bound<'_, PyAny>, then_value: &Bound<'_, PyAny>, else_value: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
    Ok(expression(Node::IfThenElse(coerce(condition)?, coerce(then_value)?, coerce(else_value)?)))
}

#[pyfunction(name = "eq")]
fn eq_expr(left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
    Ok(expression(Node::Eq(coerce(left)?, coerce(right)?)))
}

#[pyfunction(name = "ne")]
fn ne_expr(left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
    Ok(expression(Node::Ne(coerce(left)?, coerce(right)?)))
}

#[pyfunction]
#[pyo3(signature = (value, *, scale=1_000_000))]
fn fixed(value: f64, scale: i64) -> PyResult<i64> {
    list::FixedPoint::from_f64(value, scale).map(|value| value.raw).map_err(PyValueError::new_err)
}

#[pyfunction]
fn mul_scaled(left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>, scale: i64) -> PyResult<PyLambdaExpr> {
    require_positive_scale(scale)?;
    Ok(expression(Node::MulScaled(coerce(left)?, coerce(right)?, scale)))
}

#[pyfunction]
fn div_scaled(left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>, scale: i64) -> PyResult<PyLambdaExpr> {
    require_positive_scale(scale)?;
    Ok(expression(Node::DivScaled(coerce(left)?, coerce(right)?, scale)))
}

fn require_positive_scale(scale: i64) -> PyResult<()> {
    if scale <= 0 {
        Err(PyValueError::new_err("fixed-point scale must be positive"))
    } else {
        Ok(())
    }
}

#[pyfunction]
fn piecewise(input: &Bound<'_, PyAny>, points: Vec<(i64, i64)>) -> PyResult<PyLambdaExpr> {
    if points.is_empty() || points.windows(2).any(|window| window[0].0 >= window[1].0) {
        return Err(PyValueError::new_err("piecewise points need strictly increasing x coordinates"));
    }
    Ok(expression(Node::PiecewiseLinear(coerce(input)?, Arc::new(points))))
}

#[pyfunction]
fn register_external(name: String, function: Py<PyAny>) -> PyResult<()> {
    list::register_external_function(name, move |arguments| {
        Python::attach(|py| {
            let arguments = PyTuple::new(py, arguments).map_err(|error| error.to_string())?;
            function.bind(py).call1(arguments).and_then(|value| value.extract::<i64>()).map_err(|error| error.to_string())
        })
    })
    .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (name, *args))]
fn external(name: String, args: &Bound<'_, PyTuple>) -> PyResult<PyLambdaExpr> {
    if !list::external_function_registered(&name) {
        return Err(PyValueError::new_err(format!("external function '{name}' is not registered")));
    }
    let arguments = args.iter().map(|argument| coerce(&argument)).collect::<PyResult<Vec<_>>>()?;
    Ok(expression(Node::External(name.into(), Arc::new(arguments))))
}

#[pyfunction]
fn array(data: Vec<i64>) -> PyArray {
    PyArray { data: Arc::new(data) }
}

#[pyfunction]
fn matrix(data: Vec<Vec<i64>>) -> PyMatrix {
    PyMatrix { data: Arc::new(data) }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLambdaExpr>()?;
    module.add_class::<PyArray>()?;
    module.add_class::<PyMatrix>()?;
    module.add_class::<PyMatrixRow>()?;
    module.add_function(wrap_pyfunction!(array, module)?)?;
    module.add_function(wrap_pyfunction!(matrix, module)?)?;
    module.add_function(wrap_pyfunction!(min_expr, module)?)?;
    module.add_function(wrap_pyfunction!(max_expr, module)?)?;
    module.add_function(wrap_pyfunction!(abs_expr, module)?)?;
    module.add_function(wrap_pyfunction!(if_expr, module)?)?;
    module.add_function(wrap_pyfunction!(eq_expr, module)?)?;
    module.add_function(wrap_pyfunction!(ne_expr, module)?)?;
    module.add_function(wrap_pyfunction!(fixed, module)?)?;
    module.add_function(wrap_pyfunction!(mul_scaled, module)?)?;
    module.add_function(wrap_pyfunction!(div_scaled, module)?)?;
    module.add_function(wrap_pyfunction!(piecewise, module)?)?;
    module.add_function(wrap_pyfunction!(register_external, module)?)?;
    module.add_function(wrap_pyfunction!(external, module)?)?;
    Ok(())
}
