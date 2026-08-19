use std::str::FromStr;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator};

use crate::orchestrator::{SearchPhase as SemanticSearchPhase, SearchPolicy as SemanticSearchPolicy, ValueSelector, VariableSelector};

use super::PyIntVar;

#[pyclass(name = "SearchPhase", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
pub(super) struct PySearchPhase {
    variables: Vec<PyIntVar>,
    variable_selector: VariableSelector,
    value_selector: ValueSelector,
}

#[pymethods]
impl PySearchPhase {
    #[new]
    #[pyo3(signature = (variables, variable_selector="auto", value_selector="auto"))]
    fn new(variables: &Bound<'_, PyAny>, variable_selector: &str, value_selector: &str) -> PyResult<Self> {
        let variables = PyIterator::from_object(variables)?
            .map(|item| {
                item?
                    .extract::<PyRef<'_, PyIntVar>>()
                    .map(|variable| variable.clone())
                    .map_err(|_| PyTypeError::new_err("SearchPhase variables must be IntVar handles"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let variable_selector = VariableSelector::from_str(variable_selector).map_err(PyValueError::new_err)?;
        let value_selector = ValueSelector::from_str(value_selector).map_err(PyValueError::new_err)?;
        Ok(Self { variables, variable_selector, value_selector })
    }

    #[getter]
    fn variables(&self) -> Vec<PyIntVar> {
        self.variables.clone()
    }

    #[getter]
    fn variable_selector(&self) -> &'static str {
        self.variable_selector.as_str()
    }

    #[getter]
    fn value_selector(&self) -> &'static str {
        self.value_selector.as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchPhase(variables={}, variable_selector='{}', value_selector='{}')",
            self.variables.len(),
            self.variable_selector,
            self.value_selector
        )
    }
}

#[pyclass(name = "SearchPolicy", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
pub(super) struct PySearchPolicy {
    phases: Vec<PySearchPhase>,
}

impl PySearchPolicy {
    pub(super) fn semantic_for(&self, model_id: u64) -> PyResult<SemanticSearchPolicy> {
        let phases = self
            .phases
            .iter()
            .map(|phase| {
                let scope = phase
                    .variables
                    .iter()
                    .map(|variable| {
                        if variable.model_id != model_id {
                            return Err(PyValueError::new_err("search policy variable belongs to a different model"));
                        }
                        Ok(variable.index as usize)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(SemanticSearchPhase::new(scope, phase.variable_selector, phase.value_selector))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(SemanticSearchPolicy::new(phases))
    }
}

#[pymethods]
impl PySearchPolicy {
    #[new]
    fn new(phases: &Bound<'_, PyAny>) -> PyResult<Self> {
        let phases = PyIterator::from_object(phases)?
            .map(|item| {
                item?
                    .extract::<PyRef<'_, PySearchPhase>>()
                    .map(|phase| phase.clone())
                    .map_err(|_| PyTypeError::new_err("SearchPolicy phases must be SearchPhase objects"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self { phases })
    }

    #[getter]
    fn phases(&self) -> Vec<PySearchPhase> {
        self.phases.clone()
    }

    fn __repr__(&self) -> String {
        format!("SearchPolicy(phases={})", self.phases.len())
    }
}

pub(super) fn from_py(model_id: u64, policy: Option<&Bound<'_, PyAny>>) -> PyResult<Option<SemanticSearchPolicy>> {
    let Some(policy) = policy.filter(|policy| !policy.is_none()) else {
        return Ok(None);
    };
    let policy =
        policy.extract::<PyRef<'_, PySearchPolicy>>().map_err(|_| PyTypeError::new_err("search_policy must be a SearchPolicy or None"))?;
    policy.semantic_for(model_id).map(Some)
}
