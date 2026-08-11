use std::collections::{HashMap, HashSet};
use std::os::raw::c_int;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pyo3::class::basic::CompareOp;
use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError, PyTimeoutError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule, PyTuple};

use crate::model as shared_model;
use crate::model::list;
use crate::model::{Constraint, IntDomain, IntExpr as Expr, IntGlobalConstraint, IntVarRef, ModelObject, Relation};
use crate::orchestrator::{
    count_model_solutions_with_external_stop, enumerate_model_mus_with_external_stop, explain_model_mus_with_external_stop,
    extract_model_mus_with_external_stop, solve_model_with_external_stop, EngineKind, EngineReport, EventControl, EventSink,
    LinearBackendMode, LinearControls, ModelMusEnumeration, ModelMusResult, MusAtomRelation, RoutingControls, SearchStats,
    SemanticAssumption, SemanticAssumptionOp, SemanticNogoodRelation, SemanticSolveSession, SolveError, SolveEvent, SolveLimits, SolveMode,
    SolveRequest, SolveResult, SolveStatus, VerificationLevel,
};

mod expr {
    use super::{Expr, IntVarRef};

    pub(super) fn int(value: i64) -> Expr {
        Expr::Constant(value)
    }

    pub(super) fn var(variable: IntVarRef) -> Expr {
        Expr::Variable(variable)
    }

    pub(super) fn add(values: Vec<Expr>) -> Expr {
        Expr::Add(values)
    }

    pub(super) fn mul(values: Vec<Expr>) -> Expr {
        Expr::Mul(values)
    }

    pub(super) fn eq(left: Expr, right: Expr) -> Expr {
        Expr::Eq(Box::new(left), Box::new(right))
    }

    pub(super) fn ne(left: Expr, right: Expr) -> Expr {
        Expr::Ne(Box::new(left), Box::new(right))
    }

    pub(super) fn ge(left: Expr, right: Expr) -> Expr {
        Expr::Ge(Box::new(left), Box::new(right))
    }

    pub(super) fn le(left: Expr, right: Expr) -> Expr {
        Expr::Le(Box::new(left), Box::new(right))
    }

    pub(super) fn and(values: Vec<Expr>) -> Expr {
        Expr::And(values)
    }

    pub(super) fn or(values: Vec<Expr>) -> Expr {
        Expr::Or(values)
    }

    pub(super) fn imp(left: Expr, right: Expr) -> Expr {
        Expr::Imp(Box::new(left), Box::new(right))
    }
}

static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PythonIntervalRef(usize);

#[derive(Clone)]
struct ExprLike {
    model_id: Option<u64>,
    expr: Expr,
    text: String,
}

impl ExprLike {
    fn int(value: i64) -> Self {
        Self { model_id: None, expr: Expr::Constant(value), text: value.to_string() }
    }

    fn var(var: &PyIntVar) -> Self {
        Self { model_id: Some(var.model_id), expr: Expr::Variable(IntVarRef(var.index as usize)), text: var.display_name() }
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
    unordered: bool,
}

#[pymethods]
impl PyListVar {
    #[getter]
    fn index(&self) -> usize {
        self.index as usize
    }

    fn __repr__(&self) -> String {
        format!("{}({})", if self.unordered { "SetVar" } else { "ListVar" }, self.index)
    }
}

/// An interval (scheduling) decision variable handle. Carries its owning model
/// and the schedule generation that created it, so a stale interval (from an
/// earlier `intervals`/`alternatives` call) or one from another model is
/// rejected instead of silently aliasing a wrong interval by index.
#[pyclass(name = "IntervalVar", module = "qayd", from_py_object)]
#[derive(Clone)]
struct PyIntervalVar {
    model_id: u64,
    gen: u64,
    index: u32,
    kind: PyIntervalKind,
    interval: Option<PythonIntervalRef>,
    start: Option<IntVarRef>,
    presence: Option<IntVarRef>,
    /// Fixed duration; `None` for an `alternative` master whose realised
    /// duration depends on the selected member.
    duration: Option<i64>,
    /// Set on an `alternative` master: the member optional intervals, of which
    /// exactly one executes. The master's `start` is the shared start `S`; its
    /// realised duration is read off the chosen member's presence.
    alternative: Option<AlternativeInterval>,
}

/// The payload of an `alternative` master: parallel per-member data, aligned
/// to the input member order.
#[derive(Clone)]
struct AlternativeInterval {
    /// Display name for the master's `.end`/`.realized_duration` texts.
    name: String,
    /// The member intervals, echoed for `.members`.
    members: Vec<PyIntervalVar>,
    /// Per-member presence variable.
    presences: Vec<IntVarRef>,
    /// Per-member duration.
    durations: Vec<i64>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PyIntervalKind {
    Schedule,
    Native,
}

#[derive(Clone)]
struct NativeIntervalSpec {
    interval: PythonIntervalRef,
    start: IntVarRef,
    presence: Option<IntVarRef>,
    duration: i32,
}

#[pymethods]
impl PyIntervalVar {
    /// The backing native-interval index, or `None` for an `alternative`
    /// master (which has no single backing interval).
    #[getter]
    fn index(&self) -> Option<usize> {
        if self.alternative.is_some() {
            None
        } else {
            Some(self.index as usize)
        }
    }

    #[getter]
    fn start(&self) -> Option<PyIntVar> {
        self.start.map(|var| PyIntVar { model_id: self.model_id, index: var.0 as u32, name: None })
    }

    #[getter]
    fn presence(&self) -> Option<PyIntVar> {
        self.presence.map(|var| PyIntVar { model_id: self.model_id, index: var.0 as u32, name: None })
    }

    #[getter]
    fn optional(&self) -> bool {
        self.presence.is_some()
    }

    #[getter]
    fn duration(&self) -> Option<i64> {
        self.duration
    }

    /// `start + duration`, as an expression usable in constraints/objectives.
    /// For an `alternative` master it is the shared start plus the realised
    /// (chosen) member's duration, `S + Σ p_m·d_m`.
    #[getter]
    fn end(&self) -> PyResult<PyExpr> {
        if let Some(alt) = &self.alternative {
            let start = self.start.expect("an alternative master has a shared start");
            let mut terms = vec![expr::var(start)];
            terms.extend(alt.duration_terms());
            return Ok(PyExpr {
                inner: ExprLike { model_id: Some(self.model_id), expr: expr::add(terms), text: format!("{}.end", alt.name) },
            });
        }
        let (Some(start), Some(duration)) = (self.start, self.duration) else {
            return Err(PyValueError::new_err(
                "end is only available on fixed-duration intervals with a start variable; \
                 a schedule-engine interval (alternatives) has a mode-dependent duration",
            ));
        };
        Ok(PyExpr {
            inner: ExprLike {
                model_id: Some(self.model_id),
                expr: expr::add(vec![expr::var(start), expr::int(duration)]),
                text: format!("(interval{}.start + {})", self.index, duration),
            },
        })
    }

    /// The realised duration `Σ p_m·d_m` of an `alternative` master (exact
    /// under the exactly-one choice). Raises on a plain interval.
    #[getter]
    fn realized_duration(&self) -> PyResult<PyExpr> {
        let alt = self.require_alternative("realized_duration")?;
        Ok(PyExpr {
            inner: ExprLike {
                model_id: Some(self.model_id),
                expr: expr::add(alt.duration_terms()),
                text: format!("{}.realized_duration", alt.name),
            },
        })
    }

    /// The member intervals of an `alternative` master (empty otherwise), in
    /// the order they were given.
    #[getter]
    fn members(&self) -> Vec<PyIntervalVar> {
        self.alternative.as_ref().map(|alt| alt.members.clone()).unwrap_or_default()
    }

    fn __repr__(&self) -> String {
        if let Some(alt) = &self.alternative {
            return format!("IntervalVar({})", alt.name);
        }
        format!("IntervalVar({})", self.index)
    }
}

impl PyIntervalVar {
    fn require_alternative(&self, what: &str) -> PyResult<&AlternativeInterval> {
        self.alternative
            .as_ref()
            .ok_or_else(|| PyValueError::new_err(format!("{what} is only available on an alternative master (Model.alternative)")))
    }
}

impl AlternativeInterval {
    /// `p_m·d_m` per member (exactly one `p_m` is `1`).
    fn duration_terms(&self) -> Vec<Expr> {
        self.presences
            .iter()
            .zip(&self.durations)
            .map(|(&presence, &duration)| expr::mul(vec![expr::var(presence), expr::int(duration)]))
            .collect()
    }
}

/// An ordered view over native intervals with sequence-dependent setup times.
/// The order is derived from the solved start times, which is canonical because
/// the posted disjunctions keep every pair of present members separated.
#[pyclass(name = "SequenceVar", module = "qayd", from_py_object)]
#[derive(Clone)]
struct PySequenceVar {
    model_id: u64,
    index: u32,
    intervals: Vec<PyIntervalVar>,
    realizations: Vec<Vec<NativeIntervalSpec>>,
}

#[pymethods]
impl PySequenceVar {
    #[getter]
    fn index(&self) -> usize {
        self.index as usize
    }

    #[getter]
    fn intervals(&self) -> Vec<PyIntervalVar> {
        self.intervals.clone()
    }

    fn order(&self, solution: &PySolution) -> PyResult<Vec<PyIntervalVar>> {
        let solved = |variable: IntVarRef| solution.values.get(variable.0).copied().flatten();
        let mut present = Vec::new();
        for (logical, (interval, realizations)) in self.intervals.iter().zip(&self.realizations).enumerate() {
            if interval.model_id != self.model_id {
                return Err(PyRuntimeError::new_err("sequence contains an interval from another model"));
            }
            if interval.presence.is_some_and(|presence| solved(presence) == Some(0)) {
                continue;
            }
            let realization = realizations
                .iter()
                .find(|realization| realization.presence.is_none_or(|presence| solved(presence) == Some(1)))
                .ok_or_else(|| PyRuntimeError::new_err("present sequence member has no selected realization"))?;
            let start = solved(realization.start)
                .ok_or_else(|| PyRuntimeError::new_err("no start value is available for a present sequence member"))?;
            present.push((start, logical, interval.clone()));
        }
        present.sort_unstable_by_key(|(start, logical, _)| (*start, *logical));
        Ok(present.into_iter().map(|(_, _, interval)| interval).collect())
    }

    fn __repr__(&self) -> String {
        format!("SequenceVar({})", self.index)
    }
}

#[pyclass(name = "Constraint", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyConstraint {
    inner: ExprLike,
}

/// Context manager returned by [`PyModel::soft`]: on `__enter__` it opens a
/// fresh selector and routes posted constraints through it; on `__exit__` it
/// stops routing. Records the group so [`PyModel::mus`] can name it in a core.
#[pyclass(name = "SoftGroup", module = "qayd", unsendable)]
struct PySoftGroup {
    model: Py<PyModel>,
    name: Option<String>,
}

#[pymethods]
impl PySoftGroup {
    fn __enter__(&self, py: Python<'_>) -> PyResult<()> {
        let mut model = self.model.borrow_mut(py);
        if model.active_soft_selector.is_some() {
            return Err(PyValueError::new_err("soft groups cannot be nested"));
        }
        let sel = model.semantic.model_mut().bool_var();
        let name = self.name.clone().unwrap_or_else(|| format!("c{}", model.mus_selectors.len()));
        model.names.push(None);
        model.active_soft_selector = Some(sel);
        model.mus_selectors.push((name, sel));
        Ok(())
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.model.borrow_mut(py).active_soft_selector = None;
        Ok(false) // do not suppress exceptions
    }
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
    #[pyo3(get)]
    lp_rows: u64,
    #[pyo3(get)]
    lp_solves: u64,
    #[pyo3(get)]
    lp_certified: u64,
    #[pyo3(get)]
    lp_timeouts: u64,
    #[pyo3(get)]
    lp_refactorizations: u64,
    #[pyo3(get)]
    lp_micros: u64,
    #[pyo3(get)]
    lp_root_bound: Option<i64>,
    #[pyo3(get)]
    lp_node_prunes: u64,
}

/// Result of [`enumerate_mus`](PyModel::enumerate_mus): the minimal unsatisfiable
/// subsets and minimal satisfiable subsets found, and whether enumeration ran to
/// completion (`False` when a time limit or `limit` cap stopped it early).
#[pyclass(name = "MusEnumeration", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyMusEnumeration {
    #[pyo3(get)]
    muses: Vec<Vec<String>>,
    #[pyo3(get)]
    msses: Vec<Vec<String>>,
    #[pyo3(get)]
    complete: bool,
}

#[pymethods]
impl PyMusEnumeration {
    fn __repr__(&self) -> String {
        format!("MusEnumeration(muses={}, msses={}, complete={})", self.muses.len(), self.msses.len(), self.complete)
    }
}

#[pyclass(name = "Solution", module = "qayd")]
struct PySolution {
    status: String,
    objective: Option<i64>,
    objective_sense: Option<String>,
    objective_expr: Option<String>,
    values: Vec<Option<i32>>,
    stats: PySolveStats,
    /// List variable contents, set only for list-domain models.
    lists: Option<Vec<Vec<i32>>>,
    /// Value of each lexicographic objective tier. `objective` is the first tier.
    objectives: Vec<i64>,
    /// Interval start times, for a schedule model (empty otherwise).
    starts: Vec<Option<i64>>,
    /// Interval presence flags, for native interval models.
    presences: Vec<bool>,
    /// Chosen machine per interval, for a flexible (moded) schedule model
    /// (empty otherwise).
    machines: Vec<i64>,
    /// Certified bound for the primary objective and its gap to the incumbent.
    dual_bound: Option<i64>,
    absolute_gap: Option<u64>,
    relative_gap: Option<f64>,
    bound_method: Option<String>,
    /// Local-search work counters, present only when `profile=True`.
    alns_iterations: Option<u64>,
    candidates_evaluated: Option<u64>,
    candidates_per_second: Option<f64>,
    full_recompute_percentage: Option<f64>,
    backend_build_seconds: Option<f64>,
    construction_seconds: Option<f64>,
    time_to_first_feasible: Option<f64>,
    construction_candidates: Option<u64>,
    estimated_backend_bytes: Option<u64>,
    constructor: Option<String>,
    constructor_fleet: Option<usize>,
    constructor_cost: Option<i64>,
    /// (target_ns, observed_ns, feasible, objectives, fleet, candidates).
    anytime_checkpoints: Option<Vec<PyAnytimeCheckpoint>>,
    /// (name, uses, generated, evaluated, cpu_ns, improvements,
    /// global_bests, positive_rewards, final_weight).
    neighborhood_profile: Option<Vec<PyNeighborhoodProfile>>,
    /// Stable name/value routing scheduler and elite counters.
    routing_counters: Option<Vec<(String, u64)>>,
    /// Integer local-search counters carried by the canonical engine report.
    ls_moves: Option<u64>,
    ls_constraints: Option<usize>,
    ls_functionals: Option<usize>,
    ls_unsupported: Option<usize>,
    ls_rejected_incumbents: Option<usize>,
    ls_checkpoint_replays: Option<u64>,
}

#[derive(Clone)]
struct ObjectiveSpec {
    minimizing: bool,
    expr: ExprLike,
}

fn canonicalize_unordered_lists(mut lists: Option<Vec<Vec<i32>>>, unordered: &HashSet<usize>) -> Option<Vec<Vec<i32>>> {
    if let Some(lists) = &mut lists {
        for &index in unordered {
            if let Some(items) = lists.get_mut(index) {
                items.sort_unstable();
            }
        }
    }
    lists
}

/// Semantic collection model plus the small amount of Python handle state that
/// is not part of the solver IR. Objective declarations are kept separately so
/// Python's replacing `minimize` operation can rebuild a package without
/// exposing mutation of the model's objective arena.
#[derive(Clone, Default)]
struct SemanticConstruction {
    package: shared_model::ModelPackage,
    objectives: Vec<shared_model::Objective>,
    list_generation: u64,
    schedule_generation: u64,
    reified_values: Vec<Option<list::Reduction>>,
}

impl SemanticConstruction {
    fn model(&self) -> &shared_model::Model {
        &self.package.model
    }

    fn model_mut(&mut self) -> &mut shared_model::Model {
        &mut self.package.model
    }

    fn has_lists(&self) -> bool {
        !self.model().lists().is_empty()
    }

    fn has_intervals(&self) -> bool {
        !self.model().intervals().is_empty()
    }

    fn has_collection(&self) -> bool {
        self.has_lists() || self.has_intervals()
    }

    fn list_scope(&self) -> Vec<shared_model::ListVarRef> {
        (0..self.model().lists().len()).map(shared_model::ListVarRef).collect()
    }

    fn list_universe(&self) -> Option<&[i32]> {
        self.model().lists().first().map(|declaration| declaration.universe.as_slice())
    }

    fn schedule_makespan(&self) -> bool {
        self.objectives.iter().any(|objective| matches!(objective, shared_model::Objective::Makespan { minimize: true, .. }))
    }

    fn primary_list_sense(&self) -> Option<bool> {
        self.objectives.first().and_then(|objective| match objective {
            shared_model::Objective::ListTerms { minimize, .. } => Some(*minimize),
            shared_model::Objective::IntExpr { .. } | shared_model::Objective::Makespan { .. } => None,
        })
    }
}

fn populate_python_metadata(package: &mut shared_model::ModelPackage) {
    use shared_model::ModelObject;

    for index in 0..package.model.int_vars().len() {
        let object = ModelObject::IntVar(shared_model::IntVarRef(index));
        package.metadata.names.entry(object).or_insert_with(|| format!("int[{index}]"));
        package.metadata.frontend_ids.entry(("python".to_string(), format!("int:{index}"))).or_insert(object);
        if !package.metadata.outputs.contains(&object) {
            package.metadata.outputs.push(object);
        }
    }
    for index in 0..package.model.sets().len() {
        let object = ModelObject::SetVar(shared_model::SetVarRef(index));
        package.metadata.names.entry(object).or_insert_with(|| format!("set[{index}]"));
        package.metadata.frontend_ids.entry(("python".to_string(), format!("set:{index}"))).or_insert(object);
        if !package.metadata.outputs.contains(&object) {
            package.metadata.outputs.push(object);
        }
    }
    for (index, declaration) in package.model.lists().iter().enumerate() {
        let object = ModelObject::ListVar(shared_model::ListVarRef(index));
        package.metadata.names.entry(object).or_insert_with(|| format!("list[{index}]"));
        package.metadata.frontend_ids.entry(("python".to_string(), format!("list:{index}"))).or_insert(object);
        if declaration.role == shared_model::ListRole::Decision && !package.metadata.outputs.contains(&object) {
            package.metadata.outputs.push(object);
        }
    }
    for index in 0..package.model.intervals().len() {
        let object = ModelObject::IntervalVar(shared_model::IntervalVarRef(index));
        package.metadata.names.entry(object).or_insert_with(|| format!("interval[{index}]"));
        package.metadata.frontend_ids.entry(("python".to_string(), format!("interval:{index}"))).or_insert(object);
        if !package.metadata.outputs.contains(&object) {
            package.metadata.outputs.push(object);
        }
    }
    for index in 0..package.model.interval_modes().len() {
        let object = ModelObject::IntervalMode(shared_model::IntervalModeRef(index));
        package.metadata.names.entry(object).or_insert_with(|| format!("mode[{index}]"));
        package.metadata.frontend_ids.entry(("python".to_string(), format!("mode:{index}"))).or_insert(object);
    }
    for index in 0..package.model.constraints().len() {
        let object = ModelObject::Constraint(shared_model::ConstraintRef(index));
        package.metadata.names.entry(object).or_insert_with(|| format!("constraint[{index}]"));
    }
    for index in 0..package.model.objectives().len() {
        let object = ModelObject::Objective(shared_model::ObjectiveRef(index));
        package.metadata.names.entry(object).or_insert_with(|| format!("objective[{index}]"));
    }
}

#[pyclass(name = "Model", module = "qayd", unsendable)]
struct PyModel {
    id: u64,
    names: Vec<Option<String>>,
    objective: Option<ObjectiveSpec>,
    then_objectives: Vec<ObjectiveSpec>,
    native_intervals: Vec<NativeIntervalSpec>,
    native_sequences: usize,
    /// Canonical semantic construction for integer, list, and compact schedule
    /// models.
    /// Generations and reified-value slots only protect Python handles; all
    /// solver declarations live in the contained `ModelPackage`.
    semantic: SemanticConstruction,
    /// Soft-constraint groups for MUS extraction: `(name, selector)`. Each is a
    /// `{0,1}` variable guarding the constraints posted inside a `with
    /// model.soft(name):` block; selectors occupy a `names` slot (to keep
    /// `IntVarRef ↔ names` alignment) but are hidden from the user-facing variable
    /// enumerations.
    mus_selectors: Vec<(String, IntVarRef)>,
    active_soft_selector: Option<IntVarRef>,
    constants: HashMap<i32, IntVarRef>,
}

impl PyModel {
    fn is_selector(&self, index: u32) -> bool {
        self.mus_selectors.iter().any(|&(_, sel)| sel.0 == index as usize)
    }

    /// The user's decision variables (every named slot that is not a selector).
    fn decision_var_ids(&self) -> Vec<IntVarRef> {
        (0..self.names.len()).filter(|&i| !self.is_selector(i as u32)).map(IntVarRef).collect()
    }

    fn selector_name(&self, sel: IntVarRef) -> String {
        self.mus_selectors.iter().find(|&&(_, v)| v == sel).map(|(name, _)| name.clone()).unwrap_or_default()
    }

    /// Format a semantic atom `var rel value` using the variable's name.
    fn atom_text(&self, var: IntVarRef, rel: MusAtomRelation, value: i32) -> String {
        let name = self.names.get(var.0).and_then(|n| n.clone()).unwrap_or_else(|| format!("x{}", var.0));
        let op = match rel {
            MusAtomRelation::Eq => "==",
            MusAtomRelation::Ne => "!=",
            MusAtomRelation::Ge => ">=",
            MusAtomRelation::Le => "<=",
        };
        format!("{name} {op} {value}")
    }

    fn add_integer_constraint(&mut self, constraint: Constraint) {
        let constraint = if let Some(selector) = self.active_soft_selector {
            Constraint::Selected { selector, constraint: Box::new(constraint) }
        } else {
            constraint
        };
        self.semantic.model_mut().add_constraint(constraint);
    }

    fn integer_package(&self) -> shared_model::ModelPackage {
        let mut package = self.semantic.package.clone();
        for objective in objective_specs(&self.objective, &self.then_objectives) {
            package.model.add_objective(shared_model::Objective::IntExpr { minimize: objective.minimizing, expr: objective.expr.expr });
        }
        for (index, name) in self.names.iter().enumerate() {
            let object = ModelObject::IntVar(IntVarRef(index));
            if let Some(name) = name {
                package.metadata.names.insert(object, name.clone());
            }
            package.metadata.frontend_ids.insert(("python".to_string(), format!("int:{index}")), object);
            if !self.is_selector(index as u32) && !package.metadata.outputs.contains(&object) {
                package.metadata.outputs.push(object);
            }
        }
        populate_python_metadata(&mut package);
        package
    }

    fn solve_package(&self) -> shared_model::ModelPackage {
        let mut package = self.integer_package();
        for objective in &self.semantic.objectives {
            package.model.add_objective(objective.clone());
        }
        populate_python_metadata(&mut package);
        package
    }
}

#[pyclass(name = "SolveSession", module = "qayd", unsendable)]
struct PySolveSession {
    id: u64,
    names: Vec<Option<String>>,
    objectives: Vec<ObjectiveSpec>,
    search: Vec<usize>,
    native_intervals: Vec<NativeIntervalSpec>,
    session: SemanticSolveSession,
}

type PyRawNogood = (u32, Vec<u32>);
type PyNogoodLit = (u32, String, i32);
type PyNogood = (u32, Vec<PyNogoodLit>);

impl PyIntVar {
    fn display_name(&self) -> String {
        self.name.clone().unwrap_or_else(|| format!("x{}", self.index))
    }
}

impl From<SearchStats> for PySolveStats {
    fn from(stats: SearchStats) -> Self {
        Self {
            solutions: stats.solutions,
            nodes: stats.nodes,
            failures: stats.failures,
            learned_lits: stats.learned_lits,
            vivified_clauses: stats.vivified_clauses,
            vivified_lits: stats.vivified_lits,
            lp_rows: stats.lp_rows,
            lp_solves: stats.lp_solves,
            lp_certified: stats.lp_certified,
            lp_timeouts: stats.lp_timeouts,
            lp_refactorizations: stats.lp_refactorizations,
            lp_micros: stats.lp_micros,
            lp_root_bound: stats.lp_root_bound,
            lp_node_prunes: stats.lp_node_prunes,
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

fn interval_from_py(obj: &Bound<'_, PyAny>) -> PyResult<PyIntervalVar> {
    obj.extract::<PyRef<'_, PyIntervalVar>>().map(|iv| iv.clone()).map_err(|_| PyTypeError::new_err("expected an IntervalVar"))
}

fn interval_list_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Vec<PyIntervalVar>> {
    let iter = PyIterator::from_object(obj)?;
    iter.map(|item| interval_from_py(&item?)).collect()
}

fn ids_for(model_id: u64, vars: &[PyIntVar]) -> PyResult<Vec<IntVarRef>> {
    let mut out = Vec::with_capacity(vars.len());
    for var in vars {
        if var.model_id != model_id {
            return Err(PyValueError::new_err("variable belongs to a different model"));
        }
        out.push(IntVarRef(var.index as usize));
    }
    Ok(out)
}

fn one_id_for(model_id: u64, var: &PyIntVar) -> PyResult<IntVarRef> {
    if var.model_id != model_id {
        return Err(PyValueError::new_err("variable belongs to a different model"));
    }
    Ok(IntVarRef(var.index as usize))
}

fn search_ids_for(model_id: u64, num_vars: usize, search: Option<&Bound<'_, PyAny>>, extra: Option<IntVarRef>) -> PyResult<Vec<IntVarRef>> {
    search_ids_with_default(model_id, (0..num_vars).map(IntVarRef).collect(), search, extra)
}

fn search_ids_with_default(
    model_id: u64,
    default_vars: Vec<IntVarRef>,
    search: Option<&Bound<'_, PyAny>>,
    extra: Option<IntVarRef>,
) -> PyResult<Vec<IntVarRef>> {
    let mut vars = match search {
        Some(obj) if !obj.is_none() => ids_for(model_id, &var_list_from_py(obj)?)?,
        _ => default_vars,
    };
    if let Some(var) = extra {
        if !vars.contains(&var) {
            vars.push(var);
        }
    }
    Ok(vars)
}

fn search_ids(model: &PyModel, search: Option<&Bound<'_, PyAny>>, extra: Option<IntVarRef>) -> PyResult<Vec<IntVarRef>> {
    search_ids_with_default(model.id, model.decision_var_ids(), search, extra)
}

fn attach_native_interval_solution(solution: &mut PySolution, intervals: &[NativeIntervalSpec]) {
    if intervals.is_empty() || solution.values.iter().all(Option::is_none) {
        return;
    }
    let mut starts = Vec::with_capacity(intervals.len());
    let mut presences = Vec::with_capacity(intervals.len());
    for interval in intervals {
        let present = interval.presence.and_then(|var| solution.values.get(var.0).copied().flatten()) != Some(0);
        presences.push(present);
        starts.push(if present { solution.values.get(interval.start.0).copied().flatten().map(i64::from) } else { None });
    }
    solution.starts = starts;
    solution.presences = presences;
}

fn checked_interval_start_max(horizon: i64, duration: i64) -> PyResult<i32> {
    if duration < 0 {
        return Err(PyValueError::new_err("interval duration must be non-negative"));
    }
    let start_max = horizon.checked_sub(duration).ok_or_else(|| PyValueError::new_err("interval horizon minus duration overflows"))?;
    if start_max < 0 {
        return Err(PyValueError::new_err("interval duration exceeds its horizon"));
    }
    checked_i32(start_max, "interval start upper bound")
}

#[allow(clippy::too_many_arguments)]
fn make_solution(
    status: &str,
    vars: &[IntVarRef],
    assignment: Option<&[i32]>,
    objective: Option<i64>,
    objective_sense: Option<&str>,
    objective_expr: Option<&str>,
    stats: SearchStats,
    num_vars: usize,
) -> PySolution {
    let mut values = vec![None; num_vars];
    if let Some(assignment) = assignment {
        for (&var, &value) in vars.iter().zip(assignment) {
            if let Some(slot) = values.get_mut(var.0) {
                *slot = Some(value);
            }
        }
    }
    let exact_bound = (status == "OPTIMAL").then_some(objective).flatten();
    PySolution {
        status: status.to_string(),
        objective,
        objective_sense: objective_sense.map(str::to_string),
        objective_expr: objective_expr.map(str::to_string),
        values,
        stats: stats.into(),
        lists: None,
        objectives: objective.map(|o| vec![o]).unwrap_or_default(),
        starts: Vec::new(),
        presences: Vec::new(),
        machines: Vec::new(),
        dual_bound: exact_bound,
        absolute_gap: exact_bound.map(|_| 0),
        relative_gap: exact_bound.map(|_| 0.0),
        bound_method: exact_bound.map(|_| "exact proof".to_string()),
        alns_iterations: None,
        candidates_evaluated: None,
        candidates_per_second: None,
        full_recompute_percentage: None,
        backend_build_seconds: None,
        construction_seconds: None,
        time_to_first_feasible: None,
        construction_candidates: None,
        estimated_backend_bytes: None,
        constructor: None,
        constructor_fleet: None,
        constructor_cost: None,
        anytime_checkpoints: None,
        neighborhood_profile: None,
        routing_counters: None,
        ls_moves: None,
        ls_constraints: None,
        ls_functionals: None,
        ls_unsupported: None,
        ls_rejected_incumbents: None,
        ls_checkpoint_replays: None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PythonEngine {
    Auto,
    Exact,
    Ls,
}

impl PythonEngine {
    fn solve_mode(self) -> SolveMode {
        match self {
            Self::Auto => SolveMode::Auto,
            Self::Exact => SolveMode::Exact,
            Self::Ls => SolveMode::LocalSearch,
        }
    }
}

fn parse_engine(engine: &str) -> PyResult<PythonEngine> {
    match engine {
        "auto" => Ok(PythonEngine::Auto),
        "exact" => Ok(PythonEngine::Exact),
        "ls" => Ok(PythonEngine::Ls),
        _ => Err(PyValueError::new_err("engine must be 'auto', 'exact', or 'ls'")),
    }
}

fn parse_linear_backend(backend: &str) -> PyResult<LinearBackendMode> {
    match backend {
        "auto" => Ok(LinearBackendMode::Auto),
        "native" => Ok(LinearBackendMode::Native),
        "amthal" => Ok(LinearBackendMode::Amthal),
        _ => Err(PyValueError::new_err("linear_backend must be 'auto', 'native', or 'amthal'")),
    }
}

#[allow(clippy::too_many_arguments)]
fn linear_controls(
    backend: &str,
    root_millis: u64,
    node_millis: u64,
    node_depth_interval: usize,
    max_variables: usize,
    max_rows: usize,
    max_nonzeros: usize,
    min_coverage_percent: usize,
    phase_max_variables: usize,
) -> PyResult<LinearControls> {
    if node_depth_interval == 0 {
        return Err(PyValueError::new_err("lp_node_depth_interval must be a positive integer"));
    }
    Ok(LinearControls {
        backend: parse_linear_backend(backend)?,
        root_time: Duration::from_millis(root_millis),
        node_time: Duration::from_millis(node_millis),
        node_depth_interval,
        max_variables,
        max_rows,
        max_nonzeros,
        min_coverage_percent,
        phase_max_variables,
    })
}

#[derive(Default)]
struct CollectionEventSummary {
    improvements: u64,
    first_feasible_at: Option<f64>,
}

struct PythonCollectionEventSink {
    primary_sense: &'static str,
    verbose: bool,
    summary: CollectionEventSummary,
}

impl PythonCollectionEventSink {
    fn new(primary_sense: &'static str, verbose: bool) -> Self {
        Self { primary_sense, verbose, summary: CollectionEventSummary::default() }
    }
}

impl EventSink for PythonCollectionEventSink {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        // Announce each resolved stage as it starts, so a multi-stage pipeline
        // (e.g. warm-start LS -> exact) shows its phases live during the search
        if let SolveEvent::StageStarted { engine, warm_start } = event {
            if self.verbose {
                if warm_start {
                    println!("c warm-start {}", engine.name());
                } else {
                    println!("c {}", engine.name());
                }
            }
            return Ok(EventControl::Continue);
        }
        let SolveEvent::Progress { engine, objectives, elapsed } = event else {
            return Ok(EventControl::Continue);
        };
        let Some(&objective) = objectives.first() else {
            return Ok(EventControl::Continue);
        };
        self.summary.improvements += 1;
        self.summary.first_feasible_at.get_or_insert(elapsed.as_secs_f64());
        if self.verbose {
            match engine {
                EngineKind::ListExact | EngineKind::ScheduleExact => {
                    println!("  o {objective}  ({})", self.primary_sense);
                }
                EngineKind::RoutingExact
                | EngineKind::ListLocalSearch
                | EngineKind::RoutingLocalSearch
                | EngineKind::ScheduleLocalSearch => {
                    println!("  o {objective}  ({}, {:.2}s)", self.primary_sense, elapsed.as_secs_f64());
                }
                EngineKind::IntegerExact | EngineKind::IntegerLocalSearch | EngineKind::Linear | EngineKind::Verifier => {}
            }
        }
        Ok(EventControl::Continue)
    }
}

struct CollectionRun {
    result: SolveResult,
    engine: EngineKind,
    events: CollectionEventSummary,
}

fn verbose_collection_start(construction: &SemanticConstruction, request: &SolveRequest, time_limit: Option<u64>) {
    let model = construction.model();
    println!("qayd solve (semantic collection)");
    if construction.has_intervals() {
        let moded = model.intervals().iter().any(|interval| !interval.modes.is_empty());
        println!("  kind: {}", if moded { "intervals (machine choice)" } else { "intervals" });
        println!("  operations: {}", model.intervals().len());
    } else {
        println!("  kind: lists");
        println!("  items: {}", construction.list_universe().map_or(0, <[i32]>::len));
        println!("  lists: {}", model.lists().len());
    }
    println!("  requested mode: {:?}", request.mode);
    println!("  constraints: {}", model.constraints().len());
    println!("  objective tiers: {}", construction.objectives.len());
    println!("  threads: {}", request.threads);
    match time_limit {
        Some(seconds) => println!("  time limit: {seconds}s"),
        None => println!("  time limit: none"),
    }
}

fn result_engine_report(result: &SolveResult, engine: EngineKind) -> Option<&EngineReport> {
    result.reports().iter().find(|report| report.engine == Some(engine)).or_else(|| result.reports().first())
}

fn collection_result_engine(result: &SolveResult) -> EngineKind {
    result
        .reports()
        .iter()
        .rev()
        .find_map(|report| report.engine)
        .or_else(|| result.primal().map(|candidate| candidate.source()))
        .unwrap_or(EngineKind::Verifier)
}

fn report_metadata<'a>(report: Option<&'a EngineReport>, key: &str) -> Option<&'a str> {
    report?.metadata.iter().find(|(name, _)| name == key).map(|(_, value)| value.as_str())
}

fn parse_report_metadata<T: std::str::FromStr>(report: Option<&EngineReport>, key: &str) -> Option<T> {
    report_metadata(report, key)?.parse().ok()
}

type PyAnytimeCheckpoint = (u64, u64, bool, Vec<i64>, Option<usize>, u64);
type PyNeighborhoodProfile = (String, u64, u64, u64, u64, u64, u64, u64, f64);

fn parse_anytime_checkpoints(report: Option<&EngineReport>) -> Option<Vec<PyAnytimeCheckpoint>> {
    let encoded = report_metadata(report, "anytime_checkpoints")?;
    let mut records = encoded.split(';');
    if records.next()? != "1" {
        return None;
    }
    records
        .map(|record| {
            let fields = record.split(',').collect::<Vec<_>>();
            if fields.len() != 6 {
                return None;
            }
            let objectives = if fields[5].is_empty() {
                Vec::new()
            } else {
                fields[5].split(':').map(str::parse).collect::<Result<Vec<i64>, _>>().ok()?
            };
            Some((
                fields[0].parse().ok()?,
                fields[1].parse().ok()?,
                match fields[2] {
                    "0" => false,
                    "1" => true,
                    _ => return None,
                },
                objectives,
                if fields[3].is_empty() { None } else { Some(fields[3].parse().ok()?) },
                fields[4].parse().ok()?,
            ))
        })
        .collect()
}

fn parse_neighborhood_profile(report: Option<&EngineReport>) -> Option<Vec<PyNeighborhoodProfile>> {
    let encoded = report_metadata(report, "neighborhood_profile")?;
    let mut records = encoded.split(';');
    if records.next()? != "1" {
        return None;
    }
    records
        .map(|record| {
            let fields = record.split(',').collect::<Vec<_>>();
            if fields.len() != 9 {
                return None;
            }
            let weight_milli = fields[8].parse::<u64>().ok()?;
            Some((
                decode_metadata_component(fields[0])?,
                fields[1].parse().ok()?,
                fields[2].parse().ok()?,
                fields[3].parse().ok()?,
                fields[4].parse().ok()?,
                fields[5].parse().ok()?,
                fields[6].parse().ok()?,
                fields[7].parse().ok()?,
                weight_milli as f64 / 1_000.0,
            ))
        })
        .collect()
}

fn decode_metadata_component(encoded: &str) -> Option<String> {
    let bytes = encoded.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            decoded.push(bytes[index]);
            index += 1;
            continue;
        }
        let high = hex_digit(*bytes.get(index + 1)?)?;
        let low = hex_digit(*bytes.get(index + 2)?)?;
        decoded.push((high << 4) | low);
        index += 3;
    }
    String::from_utf8(decoded).ok()
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

const ROUTING_COUNTER_KEYS: [&str; 16] = [
    "routing_slices",
    "routing_descent_slices",
    "routing_alns_slices",
    "routing_relink_slices",
    "routing_global_scan_slices",
    "routing_route_elimination_attempts",
    "routing_ejection_chain_attempts",
    "routing_chain_relocate_attempts",
    "routing_guided_segment_exchange_attempts",
    "routing_macro_candidates_built",
    "routing_macro_budget_exhaustions",
    "routing_elite_insertions",
    "routing_elite_rejections",
    "routing_path_relink_attempts",
    "routing_path_relink_steps",
    "routing_path_relink_budget_exhaustions",
];

fn parse_routing_counters(report: Option<&EngineReport>) -> Option<Vec<(String, u64)>> {
    ROUTING_COUNTER_KEYS
        .iter()
        .map(|key| Some((key.trim_start_matches("routing_").to_string(), parse_report_metadata(report, key)?)))
        .collect()
}

fn attach_result_profile(solution: &mut PySolution, result: &SolveResult, engine: EngineKind) {
    let orchestration_report = result.reports().first();
    let engine_report = result_engine_report(result, engine);
    solution.backend_build_seconds = parse_report_metadata(orchestration_report, "backend_build_seconds");
    solution.estimated_backend_bytes = parse_report_metadata(orchestration_report, "estimated_backend_bytes");
    solution.ls_moves = parse_report_metadata(engine_report, "ls_moves");
    solution.ls_constraints = parse_report_metadata(engine_report, "ls_constraints");
    solution.ls_functionals = parse_report_metadata(engine_report, "ls_functionals");
    solution.ls_unsupported = parse_report_metadata(engine_report, "ls_unsupported");
    solution.ls_rejected_incumbents = parse_report_metadata(engine_report, "ls_rejected_incumbents");
    solution.ls_checkpoint_replays = parse_report_metadata(engine_report, "ls_checkpoint_replays");
}

fn collection_bound_fields(
    result: &SolveResult,
    primal: Option<i64>,
    minimizing: bool,
) -> (Option<i64>, Option<u64>, Option<f64>, Option<String>) {
    let (Some(_primal), Some(bound)) = (primal, result.bounds().iter().find(|bound| bound.tier == 0).or_else(|| result.bounds().first()))
    else {
        return (None, None, None, None);
    };
    let Some(gap) = result.optimality_gap(0, minimizing) else {
        return (None, None, None, None);
    };
    (Some(bound.value), Some(gap.absolute), Some(gap.relative), Some(bound.method.clone()))
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
    if solution.stats.lp_rows > 0 {
        println!("  lp_rows: {}", solution.stats.lp_rows);
        println!("  lp_root_bound: {}", solution.stats.lp_root_bound.map_or_else(|| "?".to_string(), |bound| bound.to_string()));
        println!("  lp_solves: {}", solution.stats.lp_solves);
        println!("  lp_certified: {}", solution.stats.lp_certified);
        println!("  lp_node_prunes: {}", solution.stats.lp_node_prunes);
        println!("  lp_time_ms: {:.3}", solution.stats.lp_micros as f64 / 1000.0);
    }
}

fn verbose_collection_finish(solution: &PySolution, run: &CollectionRun, profile: bool) {
    let report = result_engine_report(&run.result, run.engine);
    match run.engine {
        EngineKind::RoutingExact => {
            println!("qayd result (integer routing)");
            println!("  status: {}", solution.status);
            if run.result.primal().is_some() {
                println!("  objectives: {:?}", solution.objectives);
            }
            verbose_collection_bound(solution);
            println!("  improvements: {}", report.map_or(0, |report| report.improvements));
            println!("  solutions: {}", solution.stats.solutions);
            println!("  nodes: {}", solution.stats.nodes);
            println!("  failures: {}", solution.stats.failures);
        }
        EngineKind::ListExact => {
            println!("qayd result (domain exact)");
            println!("  status: {}", solution.status);
            if !solution.objectives.is_empty() {
                println!("  objectives: {:?}", solution.objectives);
            }
            verbose_collection_bound(solution);
            println!("  solutions: {}", solution.stats.solutions);
            println!("  nodes: {}", solution.stats.nodes);
            println!("  failures: {}", solution.stats.failures);
        }
        EngineKind::ScheduleExact => {
            println!("qayd result (domain exact)");
            println!("  status: {}", solution.status);
            if let Some(objective) = solution.objective {
                println!("  objective: {objective}");
            }
            verbose_collection_bound(solution);
            println!("  solutions: {}", solution.stats.solutions);
            println!("  nodes: {}", solution.stats.nodes);
            println!("  failures: {}", solution.stats.failures);
        }
        EngineKind::ListLocalSearch | EngineKind::RoutingLocalSearch | EngineKind::ScheduleLocalSearch => {
            println!("qayd result (collection)");
            println!("  status: {}", solution.status);
            if run.result.primal().is_some() {
                println!("  objectives: {:?}", solution.objectives);
            }
            verbose_collection_bound(solution);
            println!("  improvements: {}", run.events.improvements);
            if profile {
                println!("  constructor: {}", solution.constructor.as_deref().unwrap_or("none"));
                println!("  construction: {:.6}s", solution.construction_seconds.unwrap_or(0.0));
                println!("  construction candidates: {}", solution.construction_candidates.unwrap_or(0));
                println!("  ALNS iterations: {}", solution.alns_iterations.unwrap_or(0));
                println!("  candidates: {}", solution.candidates_evaluated.unwrap_or(0));
                println!("  candidates/s: {:.1}", solution.candidates_per_second.unwrap_or(0.0));
                println!("  full recomputations: {:.2}%", solution.full_recompute_percentage.unwrap_or(0.0));
            }
        }
        EngineKind::IntegerExact | EngineKind::IntegerLocalSearch | EngineKind::Linear | EngineKind::Verifier => {}
    }
}

fn verbose_collection_bound(solution: &PySolution) {
    if let Some(dual) = solution.dual_bound {
        println!("  dual: {dual}");
        println!("  gap: {:.2}%", 100.0 * solution.relative_gap.unwrap_or(0.0));
        if let Some(method) = &solution.bound_method {
            println!("  bound method: {method}");
        }
    }
}

/// SIGINT (Ctrl-C), the same value on Unix and Windows.
const SIGINT: c_int = 2;

/// Set from our SIGINT handler while a solve is running. `check_signals` is a
/// no-op off the main thread, and the main thread is detached in native compute,
/// so an OS handler is the only thing that can observe Ctrl-C mid-solve.
static SIGINT_TRIPPED: AtomicBool = AtomicBool::new(false);

type SigHandler = unsafe extern "C" fn(c_int);

struct SigintInstall {
    /// Number of solves currently arming the handler; the interpreter's own
    /// handler is saved on the first and restored on the last.
    depth: usize,
    prev: Option<SigHandler>,
}

static SIGINT_STATE: Mutex<SigintInstall> = Mutex::new(SigintInstall { depth: 0, prev: None });

unsafe extern "C" fn handle_sigint(_sig: c_int) {
    SIGINT_TRIPPED.store(true, Ordering::SeqCst);
}

/// Installs the SIGINT handler for the duration of a solve (nesting-safe), and
/// restores the interpreter's handler when the last active solve finishes.
struct SigintGuard;

impl SigintGuard {
    fn arm() -> Self {
        let mut state = SIGINT_STATE.lock().unwrap();
        if state.depth == 0 {
            SIGINT_TRIPPED.store(false, Ordering::SeqCst);
            state.prev = Some(unsafe { ffi::PyOS_setsig(SIGINT, handle_sigint) });
        }
        state.depth += 1;
        SigintGuard
    }
}

impl Drop for SigintGuard {
    fn drop(&mut self) {
        let mut state = SIGINT_STATE.lock().unwrap();
        state.depth -= 1;
        if state.depth == 0 {
            if let Some(prev) = state.prev.take() {
                unsafe { ffi::PyOS_setsig(SIGINT, prev) };
            }
        }
    }
}

/// Run the pure-Rust `compute` region of a solve with the GIL released, so
/// Python background threads keep running. Canonical orchestrator services
/// observe the SIGINT flag directly, and a `KeyboardInterrupt` is raised when
/// the interrupted solve returns.
fn with_interrupts<T, F>(py: Python<'_>, compute: F) -> PyResult<T>
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    let _sigint = SigintGuard::arm();
    let result = py.detach(compute);
    if SIGINT_TRIPPED.load(Ordering::SeqCst) {
        return Err(PyKeyboardInterrupt::new_err("solve interrupted"));
    }
    Ok(result)
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

fn checked_nonnegative_i32(value: i64, name: &str) -> PyResult<i32> {
    if value < 0 {
        return Err(PyValueError::new_err(format!("{name} must be non-negative")));
    }
    checked_i32(value, name)
}

fn relation_to_assumption_op(relation: &str) -> PyResult<SemanticAssumptionOp> {
    match relation {
        "==" | "=" | "eq" | "Eq" => Ok(SemanticAssumptionOp::Eq),
        "!=" | "<>" | "ne" | "Ne" => Ok(SemanticAssumptionOp::Ne),
        "<=" | "le" | "Le" => Ok(SemanticAssumptionOp::Le),
        "<" | "lt" | "Lt" => Ok(SemanticAssumptionOp::Lt),
        ">=" | "ge" | "Ge" => Ok(SemanticAssumptionOp::Ge),
        ">" | "gt" | "Gt" => Ok(SemanticAssumptionOp::Gt),
        _ => Err(PyValueError::new_err("assumption relation must be one of == != <= < >= >")),
    }
}

fn flipped_op(op: SemanticAssumptionOp) -> SemanticAssumptionOp {
    match op {
        SemanticAssumptionOp::Eq => SemanticAssumptionOp::Eq,
        SemanticAssumptionOp::Ne => SemanticAssumptionOp::Ne,
        SemanticAssumptionOp::Le => SemanticAssumptionOp::Ge,
        SemanticAssumptionOp::Lt => SemanticAssumptionOp::Gt,
        SemanticAssumptionOp::Ge => SemanticAssumptionOp::Le,
        SemanticAssumptionOp::Gt => SemanticAssumptionOp::Lt,
    }
}

fn var_const_assumption(lhs: &Expr, rhs: &Expr, op: SemanticAssumptionOp) -> Option<SemanticAssumption> {
    match (lhs, rhs) {
        (Expr::Variable(var), Expr::Constant(value)) => {
            i32::try_from(*value).ok().map(|value| SemanticAssumption { variable: var.0, operation: op, value })
        }
        (Expr::Constant(value), Expr::Variable(var)) => {
            i32::try_from(*value).ok().map(|value| SemanticAssumption { variable: var.0, operation: flipped_op(op), value })
        }
        _ => None,
    }
}

fn simple_assumption_expr(expr: &Expr) -> Option<SemanticAssumption> {
    match expr {
        Expr::Eq(lhs, rhs) => var_const_assumption(lhs, rhs, SemanticAssumptionOp::Eq),
        Expr::Ne(lhs, rhs) => var_const_assumption(lhs, rhs, SemanticAssumptionOp::Ne),
        Expr::Le(lhs, rhs) => var_const_assumption(lhs, rhs, SemanticAssumptionOp::Le),
        Expr::Lt(lhs, rhs) => var_const_assumption(lhs, rhs, SemanticAssumptionOp::Lt),
        Expr::Ge(lhs, rhs) => var_const_assumption(lhs, rhs, SemanticAssumptionOp::Ge),
        Expr::Gt(lhs, rhs) => var_const_assumption(lhs, rhs, SemanticAssumptionOp::Gt),
        Expr::Not(inner) => simple_assumption_expr(inner).map(|assumption| {
            let operation = match assumption.operation {
                SemanticAssumptionOp::Eq => SemanticAssumptionOp::Ne,
                SemanticAssumptionOp::Ne => SemanticAssumptionOp::Eq,
                SemanticAssumptionOp::Le => SemanticAssumptionOp::Gt,
                SemanticAssumptionOp::Lt => SemanticAssumptionOp::Ge,
                SemanticAssumptionOp::Ge => SemanticAssumptionOp::Lt,
                SemanticAssumptionOp::Gt => SemanticAssumptionOp::Le,
            };
            SemanticAssumption { operation, ..assumption }
        }),
        Expr::Variable(var) => Some(SemanticAssumption { variable: var.0, operation: SemanticAssumptionOp::Eq, value: 1 }),
        _ => None,
    }
}

fn assumption_from_py(model_id: u64, item: &Bound<'_, PyAny>) -> PyResult<SemanticAssumption> {
    if let Ok(constraint) = item.extract::<PyRef<'_, PyConstraint>>() {
        if let Some(owner) = constraint.inner.model_id {
            if owner != model_id {
                return Err(PyValueError::new_err("assumption belongs to a different model"));
            }
        }
        return simple_assumption_expr(&constraint.inner.expr)
            .ok_or_else(|| PyValueError::new_err("assumptions must be simple variable bounds such as x == v, x <= v, or x >= v"));
    }
    if let Ok(var) = item.extract::<PyRef<'_, PyIntVar>>() {
        return Ok(SemanticAssumption { variable: one_id_for(model_id, &var)?.0, operation: SemanticAssumptionOp::Eq, value: 1 });
    }
    if let Ok((var, value)) = item.extract::<(PyRef<'_, PyIntVar>, i32)>() {
        return Ok(SemanticAssumption { variable: one_id_for(model_id, &var)?.0, operation: SemanticAssumptionOp::Eq, value });
    }
    if let Ok((var, relation, value)) = item.extract::<(PyRef<'_, PyIntVar>, String, i32)>() {
        return Ok(SemanticAssumption { variable: one_id_for(model_id, &var)?.0, operation: relation_to_assumption_op(&relation)?, value });
    }
    Err(PyTypeError::new_err("assumption must be a simple Constraint, an IntVar, (IntVar, value), or (IntVar, relation, value)"))
}

fn assumptions_from_py(model_id: u64, assumptions: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<SemanticAssumption>> {
    let Some(assumptions) = assumptions else {
        return Ok(Vec::new());
    };
    if assumptions.is_none() {
        return Ok(Vec::new());
    }
    let iter = PyIterator::from_object(assumptions)?;
    iter.map(|item| assumption_from_py(model_id, &item?)).collect()
}

fn hint_pairs_from_py(model_id: u64, hints: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<(usize, i32)>> {
    let Some(hints) = hints else {
        return Ok(Vec::new());
    };
    if hints.is_none() {
        return Ok(Vec::new());
    }
    let iter = PyIterator::from_object(hints)?;
    let mut out = Vec::new();
    for item in iter {
        let item = item?;
        if let Ok((var, value)) = item.extract::<(PyRef<'_, PyIntVar>, i32)>() {
            out.push((one_id_for(model_id, &var)?.0, value));
        } else {
            return Err(PyTypeError::new_err("hints must be an iterable of (IntVar, value) pairs"));
        }
    }
    Ok(out)
}

fn branch_order_from_py(model_id: u64, branch_order: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<usize>> {
    let Some(branch_order) = branch_order else {
        return Ok(Vec::new());
    };
    if branch_order.is_none() {
        return Ok(Vec::new());
    }
    Ok(ids_for(model_id, &var_list_from_py(branch_order)?)?.into_iter().map(|variable| variable.0).collect())
}

fn objective_specs(primary: &Option<ObjectiveSpec>, tiers: &[ObjectiveSpec]) -> Vec<ObjectiveSpec> {
    let mut out = Vec::new();
    if let Some(objective) = primary {
        out.push(objective.clone());
    }
    out.extend_from_slice(tiers);
    out
}

fn call_incumbent_candidate(
    py: Python<'_>,
    callback: &Bound<'_, PyAny>,
    candidate: &crate::orchestrator::CandidateSolution,
) -> PyResult<()> {
    let Some(&value) = candidate.objectives().last() else {
        return Ok(());
    };
    let values = PyDict::new(py);
    for (index, value) in candidate.assignment().integers.iter().enumerate() {
        if let Some(value) = value {
            values.set_item(index as u32, value)?;
        }
    }
    callback.call1((value, values))?;
    Ok(())
}

fn integer_solve_error(error: SolveError) -> PyErr {
    match error {
        SolveError::InvalidRequest(message) | SolveError::Unsupported(message) | SolveError::Compile(message) => {
            PyValueError::new_err(message)
        }
        SolveError::Engine(message) | SolveError::InvalidResult(message) | SolveError::Interrupted(message) => {
            PyRuntimeError::new_err(message)
        }
    }
}

struct PythonIntegerEventSink {
    verbose: bool,
    callback: Option<Py<PyAny>>,
    callback_error: Option<PyErr>,
}

impl PythonIntegerEventSink {
    fn new(verbose: bool, callback: Option<Py<PyAny>>) -> Self {
        Self { verbose, callback, callback_error: None }
    }
}

impl EventSink for PythonIntegerEventSink {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        match event {
            SolveEvent::Progress { engine, objectives, .. } if self.verbose => {
                if let Some(&value) = objectives.last() {
                    if engine == EngineKind::IntegerLocalSearch {
                        println!("  incumbent: {value}");
                    } else {
                        println!("  incumbent tier {}: {value}", objectives.len().saturating_sub(1));
                    }
                }
            }
            SolveEvent::Candidate(candidate)
                if candidate.verification() == VerificationLevel::Transfer && self.callback.is_some() && self.callback_error.is_none() =>
            {
                let result = Python::attach(|py| {
                    let callback = self.callback.as_ref().expect("callback checked above").bind(py);
                    call_incumbent_candidate(py, callback, &candidate)
                });
                if let Err(error) = result {
                    self.callback_error = Some(error);
                    return Err(SolveError::Engine("Python incumbent callback failed".to_string()));
                }
            }
            _ => {}
        }
        Ok(EventControl::Continue)
    }
}

struct PythonSolveEventSink {
    integer: PythonIntegerEventSink,
    collection: PythonCollectionEventSink,
}

/// Engine an event is attributed to, when it carries one.
fn engine_of(event: &SolveEvent) -> Option<EngineKind> {
    match event {
        SolveEvent::Progress { engine, .. } | SolveEvent::StageStarted { engine, .. } => Some(*engine),
        _ => None,
    }
}

impl PythonSolveEventSink {
    fn new(verbose: bool, callback: Option<Py<PyAny>>, primary_sense: &'static str) -> Self {
        Self { integer: PythonIntegerEventSink::new(verbose, callback), collection: PythonCollectionEventSink::new(primary_sense, verbose) }
    }
}

impl EventSink for PythonSolveEventSink {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        let collection_engine = matches!(
            engine_of(&event),
            Some(
                EngineKind::RoutingExact
                    | EngineKind::ListExact
                    | EngineKind::ScheduleExact
                    | EngineKind::ListLocalSearch
                    | EngineKind::RoutingLocalSearch
                    | EngineKind::ScheduleLocalSearch
            )
        );
        // Route both progress and stage-start markers of a collection engine to
        // the collection sink; everything else (and engineless events) goes to
        // the integer sink.
        if collection_engine && matches!(&event, SolveEvent::Progress { .. } | SolveEvent::StageStarted { .. }) {
            self.collection.emit(event)
        } else {
            self.integer.emit(event)
        }
    }
}

fn result_search_stats(result: &SolveResult, _engine: EngineKind) -> SearchStats {
    result.aggregate_search_stats()
}

fn visible_integer_values(result: &SolveResult, num_vars: usize) -> PyResult<Vec<Option<i32>>> {
    let Some(candidate) = result.primal() else {
        return Ok(vec![None; num_vars]);
    };
    if candidate.assignment().integers.len() < num_vars {
        return Err(PyRuntimeError::new_err(format!(
            "canonical integer assignment has {} entries, expected {num_vars}",
            candidate.assignment().integers.len()
        )));
    }
    candidate
        .assignment()
        .integers
        .iter()
        .take(num_vars)
        .map(|value| {
            value
                .map(|value| i32::try_from(value).map_err(|_| PyRuntimeError::new_err("canonical integer assignment is outside i32")))
                .transpose()
        })
        .collect()
}

fn integer_solution_from_result(
    result: &SolveResult,
    objectives: &[ObjectiveSpec],
    num_vars: usize,
    engine: EngineKind,
    profile: bool,
) -> PyResult<PySolution> {
    let objective_values = result.primal().map(|candidate| candidate.objectives().to_vec()).unwrap_or_default();
    let objective = objective_values.first().copied();
    let primary = objectives.first();
    let sense = primary.map(|objective| if objective.minimizing { "min" } else { "max" });
    let text = primary.map(|objective| objective.expr.text.as_str());
    let mut solution =
        make_solution(result.status().as_str(), &[], None, objective, sense, text, result_search_stats(result, engine), num_vars);
    solution.values = visible_integer_values(result, num_vars)?;
    solution.objectives = objective_values;
    if let Some(primary) = primary.filter(|_| engine == EngineKind::IntegerExact) {
        let (dual_bound, absolute_gap, relative_gap, bound_method) = collection_bound_fields(result, objective, primary.minimizing);
        solution.dual_bound = dual_bound;
        solution.absolute_gap = absolute_gap;
        solution.relative_gap = relative_gap;
        solution.bound_method = if result.status() == SolveStatus::Optimal { Some("exact proof".to_string()) } else { bound_method };
    }
    if profile {
        attach_result_profile(&mut solution, result, engine);
    }
    Ok(solution)
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
        format!("SearchStats(solutions={}, nodes={}, failures={})", self.solutions, self.nodes, self.failures)
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

    /// Certified lower bound for minimization, or upper bound for maximization.
    #[getter]
    fn dual_bound(&self) -> Option<i64> {
        self.dual_bound
    }

    #[getter]
    fn absolute_gap(&self) -> Option<u64> {
        self.absolute_gap
    }

    /// Relative primal-dual gap as a ratio. A value of `0.01` means one percent.
    #[getter]
    fn relative_gap(&self) -> Option<f64> {
        self.relative_gap
    }

    #[getter]
    fn bound_method(&self) -> Option<String> {
        self.bound_method.clone()
    }

    #[getter]
    fn alns_iterations(&self) -> Option<u64> {
        self.alns_iterations
    }

    #[getter]
    fn candidates_evaluated(&self) -> Option<u64> {
        self.candidates_evaluated
    }

    #[getter]
    fn candidates_per_second(&self) -> Option<f64> {
        self.candidates_per_second
    }

    #[getter]
    fn full_recompute_percentage(&self) -> Option<f64> {
        self.full_recompute_percentage
    }

    #[getter]
    fn backend_build_seconds(&self) -> Option<f64> {
        self.backend_build_seconds
    }

    #[getter]
    fn construction_seconds(&self) -> Option<f64> {
        self.construction_seconds
    }

    #[getter]
    fn time_to_first_feasible(&self) -> Option<f64> {
        self.time_to_first_feasible
    }

    #[getter]
    fn construction_candidates(&self) -> Option<u64> {
        self.construction_candidates
    }

    #[getter]
    fn estimated_backend_bytes(&self) -> Option<u64> {
        self.estimated_backend_bytes
    }

    #[getter]
    fn constructor(&self) -> Option<String> {
        self.constructor.clone()
    }

    #[getter]
    fn constructor_fleet(&self) -> Option<usize> {
        self.constructor_fleet
    }

    #[getter]
    fn constructor_cost(&self) -> Option<i64> {
        self.constructor_cost
    }

    /// Internal anytime records as `(target_ns, observed_ns, feasible,
    /// objectives, fleet, candidates)` tuples.
    #[getter]
    fn anytime_checkpoints(&self) -> Option<Vec<PyAnytimeCheckpoint>> {
        self.anytime_checkpoints.clone()
    }

    /// Per-neighborhood records as `(name, uses, generated, evaluated,
    /// cpu_ns, improvements, global_bests, positive_rewards, weight)` tuples.
    #[getter]
    fn neighborhood_profile(&self) -> Option<Vec<PyNeighborhoodProfile>> {
        self.neighborhood_profile.clone()
    }

    /// Routing scheduler and elite counters as stable name/value pairs.
    #[getter]
    fn routing_counters(&self) -> Option<Vec<(String, u64)>> {
        self.routing_counters.clone()
    }

    #[getter]
    fn ls_moves(&self) -> Option<u64> {
        self.ls_moves
    }

    #[getter]
    fn ls_constraints(&self) -> Option<usize> {
        self.ls_constraints
    }

    #[getter]
    fn ls_functionals(&self) -> Option<usize> {
        self.ls_functionals
    }

    #[getter]
    fn ls_unsupported(&self) -> Option<usize> {
        self.ls_unsupported
    }

    #[getter]
    fn ls_rejected_incumbents(&self) -> Option<usize> {
        self.ls_rejected_incumbents
    }

    #[getter]
    fn ls_checkpoint_replays(&self) -> Option<u64> {
        self.ls_checkpoint_replays
    }

    /// Per-tier objective values for a lexicographic (multi-objective) model.
    #[getter]
    fn objectives(&self) -> Vec<i64> {
        self.objectives.clone()
    }

    /// Interval start times for a schedule model (empty otherwise).
    #[getter]
    fn starts(&self) -> Vec<Option<i64>> {
        self.starts.clone()
    }

    /// Interval presence flags for a native interval model.
    #[getter]
    fn presences(&self) -> Vec<bool> {
        self.presences.clone()
    }

    /// Chosen machine per interval for a flexible schedule model (empty otherwise).
    #[getter]
    fn machines(&self) -> Vec<i64> {
        self.machines.clone()
    }

    #[getter]
    fn objective_sense(&self) -> Option<String> {
        self.objective_sense.clone()
    }

    #[getter]
    fn objective_expr(&self) -> Option<String> {
        self.objective_expr.clone()
    }

    /// Values of the integer variables declared through the Python model.
    #[getter]
    fn values(&self) -> Vec<Option<i32>> {
        self.values.clone()
    }

    /// List variable contents (one list per list variable), or `None` otherwise.
    #[getter]
    fn lists(&self) -> Option<Vec<Vec<i32>>> {
        self.lists.clone()
    }

    #[getter]
    fn stats(&self) -> PySolveStats {
        self.stats.clone()
    }

    fn is_sat(&self) -> bool {
        matches!(self.status.as_str(), "SATISFIABLE" | "OPTIMAL")
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
impl PySolveSession {
    #[getter]
    fn learned_nogoods(&self) -> usize {
        self.session.learned_nogoods()
    }

    fn clear_nogoods(&mut self) {
        self.session.clear_nogoods();
    }

    #[pyo3(signature = (limit=None))]
    fn raw_nogoods(&self, limit: Option<usize>) -> Vec<PyRawNogood> {
        self.session.raw_nogoods(limit)
    }

    #[pyo3(signature = (limit=None))]
    fn nogoods(&self, limit: Option<usize>) -> PyResult<Vec<PyNogood>> {
        Ok(self
            .session
            .nogoods(limit)
            .map_err(integer_solve_error)?
            .into_iter()
            .map(|(lbd, literals)| {
                let literals = literals
                    .into_iter()
                    .map(|literal| {
                        let relation = match literal.relation {
                            SemanticNogoodRelation::Eq => "==",
                            SemanticNogoodRelation::Ne => "!=",
                            SemanticNogoodRelation::Ge => ">=",
                            SemanticNogoodRelation::Lt => "<",
                        };
                        (literal.variable as u32, relation.to_string(), literal.value)
                    })
                    .collect();
                (lbd, literals)
            })
            .collect())
    }

    #[pyo3(signature = (*, search=None, assumptions=None, hints=None, branch_order=None, on_incumbent=None, verbose=false, time_limit=None, seed=0, conflict_budget=None, linear_backend="auto", lp_root_ms=50, lp_node_ms=0, lp_node_depth_interval=8, lp_max_variables=2000, lp_max_rows=1000, lp_max_nonzeros=100000, lp_min_coverage_percent=1, lp_phase_max_variables=1000))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &mut self,
        py: Python<'_>,
        search: Option<&Bound<'_, PyAny>>,
        assumptions: Option<&Bound<'_, PyAny>>,
        hints: Option<&Bound<'_, PyAny>>,
        branch_order: Option<&Bound<'_, PyAny>>,
        on_incumbent: Option<&Bound<'_, PyAny>>,
        verbose: bool,
        time_limit: Option<u64>,
        seed: u64,
        conflict_budget: Option<u64>,
        linear_backend: &str,
        lp_root_ms: u64,
        lp_node_ms: u64,
        lp_node_depth_interval: usize,
        lp_max_variables: usize,
        lp_max_rows: usize,
        lp_max_nonzeros: usize,
        lp_min_coverage_percent: usize,
        lp_phase_max_variables: usize,
    ) -> PyResult<PySolution> {
        if let Some(callback) = on_incumbent {
            if !callback.is_callable() {
                return Err(PyTypeError::new_err("on_incumbent must be callable"));
            }
        }
        if verbose {
            verbose_start(self.names.len(), 0, !self.objectives.is_empty());
        }
        let assumptions = assumptions_from_py(self.id, assumptions)?;
        let hints = hint_pairs_from_py(self.id, hints)?;
        let mut guidance = branch_order_from_py(self.id, branch_order)?;
        let search_vars = match search {
            Some(obj) if !obj.is_none() => {
                search_ids_for(self.id, self.names.len(), Some(obj), None)?.into_iter().map(|variable| variable.0).collect()
            }
            _ => self.search.clone(),
        };
        for variable in search_vars {
            if !guidance.contains(&variable) {
                guidance.push(variable);
            }
        }
        let request = SolveRequest {
            mode: SolveMode::Exact,
            seed,
            limits: SolveLimits { time: time_limit.map(Duration::from_secs), conflicts: conflict_budget, ..SolveLimits::default() },
            assumptions,
            hints,
            branch_order: guidance,
            publish_incumbent_assignments: on_incumbent.is_some(),
            linear: linear_controls(
                linear_backend,
                lp_root_ms,
                lp_node_ms,
                lp_node_depth_interval,
                lp_max_variables,
                lp_max_rows,
                lp_max_nonzeros,
                lp_min_coverage_percent,
                lp_phase_max_variables,
            )?,
            ..SolveRequest::default()
        };
        let on_incumbent = on_incumbent.map(|cb| cb.clone().unbind());
        let (num_vars, objectives) = (self.names.len(), self.objectives.clone());
        let result = with_interrupts(py, || {
            let mut sink = PythonIntegerEventSink::new(verbose, on_incumbent);
            let result = self.session.solve_with_external_stop(&request, &SIGINT_TRIPPED, &mut sink);
            if let Some(error) = sink.callback_error {
                return Err(error);
            }
            result.map_err(integer_solve_error)
        })??;
        let mut solution = integer_solution_from_result(&result, &objectives, num_vars, EngineKind::IntegerExact, false)?;
        attach_native_interval_solution(&mut solution, &self.native_intervals);
        if verbose {
            verbose_finish(&solution);
        }
        Ok(solution)
    }

    fn __repr__(&self) -> String {
        format!(
            "SolveSession(num_vars={}, objectives={}, learned_nogoods={})",
            self.names.len(),
            self.objectives.len(),
            self.session.learned_nogoods()
        )
    }
}

#[pymethods]
impl PyModel {
    #[new]
    fn new() -> Self {
        Self {
            id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            names: Vec::new(),
            objective: None,
            then_objectives: Vec::new(),
            native_intervals: Vec::new(),
            native_sequences: 0,
            semantic: SemanticConstruction::default(),
            mus_selectors: Vec::new(),
            active_soft_selector: None,
            constants: HashMap::new(),
        }
    }

    #[getter]
    fn num_vars(&self) -> usize {
        self.names.len() - self.mus_selectors.len()
    }

    #[getter]
    fn num_constraints(&self) -> usize {
        self.semantic.model().constraints().len()
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
                self.semantic.model_mut().int_set(values)
            }
            (Some(lo), Some(hi), None) => {
                if lo > hi {
                    return Err(PyValueError::new_err("lower bound must be <= upper bound"));
                }
                self.semantic.model_mut().int_range(lo, hi)
            }
            _ => return Err(PyValueError::new_err("use int_var(lo, hi, name=...) or int_var(values=[...], name=...)")),
        };
        self.names.push(name.clone());
        Ok(PyIntVar { model_id: self.id, index: id.0 as u32, name })
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
        self.names
            .iter()
            .enumerate()
            .filter(|&(index, _)| !self.is_selector(index as u32))
            .map(|(index, name)| PyIntVar { model_id: self.id, index: index as u32, name: name.clone() })
            .collect()
    }

    fn add(&mut self, constraint: &Bound<'_, PyAny>) -> PyResult<()> {
        // A term comparison is a constraint on the list it references.
        if let Ok(lc) = constraint.extract::<PyRef<'_, PyListConstraint>>() {
            self.check_term_scope(lc.model_id, lc.gen)?;
            self.semantic.model_mut().add_constraint(shared_model::Constraint::ListReduction(list::Constraint {
                reduction: lc.reduction.clone(),
                op: lc.op,
                rhs: lc.rhs,
            }));
            return Ok(());
        }
        // `add(scan == v)` binds a scan total to a `list_value()` handle. The
        // binding is a definition, not a restriction, so it is recorded against
        // the value slot and resolved when the objective tier is pushed.
        if let Ok(reified) = constraint.extract::<PyRef<'_, PyReified>>() {
            self.check_term_scope(reified.model_id, reified.gen)?;
            let slot = self
                .semantic
                .reified_values
                .get_mut(reified.value)
                .ok_or_else(|| PyValueError::new_err("this list value is stale; rebuild it from the current list_vars()"))?;
            *slot = Some(reified.reduction.clone());
            return Ok(());
        }
        if let Ok(constraint) = constraint_from_py(constraint) {
            if let Some(model_id) = constraint.inner.model_id {
                if model_id != self.id {
                    return Err(PyValueError::new_err("constraint belongs to a different model"));
                }
            }
            self.add_integer_constraint(Constraint::Intension(constraint.inner.expr));
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
        self.add_integer_constraint(Constraint::Linear { terms: coeffs.into_iter().zip(vars).collect(), relation: rel, rhs });
        Ok(())
    }

    #[pyo3(signature = (vars, relation, rhs))]
    fn sum(&mut self, vars: &Bound<'_, PyAny>, relation: &str, rhs: i64) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        let rel = parse_relation(relation)?;
        self.add_integer_constraint(Constraint::Linear {
            terms: vars.into_iter().map(|variable| (1, variable)).collect(),
            relation: rel,
            rhs,
        });
        Ok(())
    }

    fn all_different(&mut self, vars: &Bound<'_, PyAny>) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { variables: vars, except: Vec::new() }));
        Ok(())
    }

    fn all_equal(&mut self, vars: &Bound<'_, PyAny>) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::AllEqual(vars)));
        Ok(())
    }

    fn not_equal(&mut self, x: &PyIntVar, y: &PyIntVar) -> PyResult<()> {
        let x = one_id_for(self.id, x)?;
        let y = one_id_for(self.id, y)?;
        self.add_integer_constraint(Constraint::Intension(expr::ne(expr::var(x), expr::var(y))));
        Ok(())
    }

    #[pyo3(signature = (x, y, offset=0))]
    fn not_equal_offset(&mut self, x: &PyIntVar, y: &PyIntVar, offset: i32) -> PyResult<()> {
        let x = one_id_for(self.id, x)?;
        let y = one_id_for(self.id, y)?;
        self.add_integer_constraint(Constraint::Intension(expr::ne(expr::var(x), expr::add(vec![expr::var(y), expr::int(offset as i64)]))));
        Ok(())
    }

    fn ordered(&mut self, vars: &Bound<'_, PyAny>, relation: &str) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        let rel = parse_relation(relation)?;
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Ordered { variables: vars, relation: rel }));
        Ok(())
    }

    fn instantiate(&mut self, vars: &Bound<'_, PyAny>, values: Vec<i32>) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if vars.len() != values.len() {
            return Err(PyValueError::new_err("vars and values must have the same length"));
        }
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Instantiation { variables: vars, values }));
        Ok(())
    }

    fn minimum(&mut self, target: &PyIntVar, vars: &Bound<'_, PyAny>) -> PyResult<()> {
        let target = one_id_for(self.id, target)?;
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if vars.is_empty() {
            return Err(PyValueError::new_err("minimum requires at least one variable"));
        }
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Minimum { target, variables: vars }));
        Ok(())
    }

    fn maximum(&mut self, target: &PyIntVar, vars: &Bound<'_, PyAny>) -> PyResult<()> {
        let target = one_id_for(self.id, target)?;
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if vars.is_empty() {
            return Err(PyValueError::new_err("maximum requires at least one variable"));
        }
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Maximum { target, variables: vars }));
        Ok(())
    }

    fn element(&mut self, array: &Bound<'_, PyAny>, index: &PyIntVar, value: &PyIntVar) -> PyResult<()> {
        let array = ids_for(self.id, &var_list_from_py(array)?)?;
        if array.is_empty() {
            return Err(PyValueError::new_err("element requires a non-empty array"));
        }
        let index = one_id_for(self.id, index)?;
        let value = one_id_for(self.id, value)?;
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Element { array, index, value }));
        Ok(())
    }

    fn element_const(&mut self, array: Vec<i32>, index: &PyIntVar, value: &PyIntVar) -> PyResult<()> {
        if array.is_empty() {
            return Err(PyValueError::new_err("element_const requires a non-empty array"));
        }
        let index = one_id_for(self.id, index)?;
        let value = one_id_for(self.id, value)?;
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::ElementConst { array, index, value }));
        Ok(())
    }

    fn count(&mut self, vars: &Bound<'_, PyAny>, value: i32, relation: &str, k: i64) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        let rel = parse_relation(relation)?;
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Count {
            variables: vars,
            value,
            relation: rel,
            count: k,
        }));
        Ok(())
    }

    #[pyo3(signature = (vars, values, low, high, *, closed=false))]
    fn cardinality(&mut self, vars: &Bound<'_, PyAny>, values: Vec<i32>, low: Vec<i64>, high: Vec<i64>, closed: bool) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if values.len() != low.len() || values.len() != high.len() {
            return Err(PyValueError::new_err("values, low, and high must have the same length"));
        }
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Cardinality {
            variables: vars,
            values,
            lower: low,
            upper: high,
            closed,
        }));
        Ok(())
    }

    fn n_values(&mut self, vars: &Bound<'_, PyAny>, relation: &str, k: i64) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        let rel = parse_relation(relation)?;
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::NValues { variables: vars, relation: rel, count: k }));
        Ok(())
    }

    #[pyo3(signature = (vars, tuples, *, positive=true))]
    fn table(&mut self, vars: &Bound<'_, PyAny>, tuples: Vec<Vec<i32>>, positive: bool) -> PyResult<()> {
        let vars = ids_for(self.id, &var_list_from_py(vars)?)?;
        if tuples.iter().any(|tuple| tuple.len() != vars.len()) {
            return Err(PyValueError::new_err("every tuple must match the variable arity"));
        }
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table { variables: vars, tuples, positive }));
        Ok(())
    }

    #[pyo3(signature = (x, y, *, strict=false))]
    fn lex(&mut self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, strict: bool) -> PyResult<()> {
        let x = ids_for(self.id, &var_list_from_py(x)?)?;
        let y = ids_for(self.id, &var_list_from_py(y)?)?;
        if x.len() != y.len() {
            return Err(PyValueError::new_err("lex vectors must have the same length"));
        }
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Lex { left: x, right: y, strict }));
        Ok(())
    }

    #[pyo3(signature = (rows, *, strict=false))]
    fn lex_chain(&mut self, rows: &Bound<'_, PyAny>, strict: bool) -> PyResult<()> {
        let rows = var_matrix_from_py(rows)?;
        let rows: Vec<Vec<IntVarRef>> = rows.iter().map(|row| ids_for(self.id, row)).collect::<PyResult<_>>()?;
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::LexChain { rows, strict }));
        Ok(())
    }

    fn channel(&mut self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let x = ids_for(self.id, &var_list_from_py(x)?)?;
        let y = ids_for(self.id, &var_list_from_py(y)?)?;
        if x.len() != y.len() {
            return Err(PyValueError::new_err("channel vectors must have the same length"));
        }
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Channel { left: x, right: y }));
        Ok(())
    }

    #[pyo3(signature = (items, durations=None))]
    fn no_overlap(&mut self, items: &Bound<'_, PyAny>, durations: Option<Vec<i64>>) -> PyResult<()> {
        if let Some(durations) = durations {
            let starts = ids_for(self.id, &var_list_from_py(items)?)?;
            if starts.len() != durations.len() {
                return Err(PyValueError::new_err("starts and durations must have the same length"));
            }
            self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap { starts, durations }));
            return Ok(());
        }

        let intervals = interval_list_from_py(items)?;
        if intervals.iter().all(|iv| iv.kind == PyIntervalKind::Native) {
            let (specs, _) = self.flatten_native_intervals(&intervals)?;
            let mut distinct = HashSet::with_capacity(specs.len());
            if specs.iter().any(|spec| !distinct.insert(spec.interval)) {
                return Err(PyValueError::new_err("no_overlap intervals must be distinct"));
            }
            self.post_optional_no_overlap(&specs);
        } else if intervals.iter().all(|iv| iv.kind == PyIntervalKind::Schedule) {
            for iv in &intervals {
                self.check_interval_scope(iv)?;
            }
            let indices = intervals.iter().map(|iv| iv.index as usize).collect();
            self.semantic.model_mut().add_constraint(shared_model::Constraint::IntervalResource(list::Resource::NoOverlap(indices)));
        } else {
            return Err(PyValueError::new_err("cannot mix native intervals and schedule intervals in one no_overlap"));
        }
        Ok(())
    }

    /// Create a total order over native intervals. `transitions[i][j]` is the
    /// non-negative setup time required when logical member `i` precedes `j`.
    fn sequence(&mut self, items: &Bound<'_, PyAny>, transitions: Vec<Vec<i64>>) -> PyResult<PySequenceVar> {
        let intervals = interval_list_from_py(items)?;
        if intervals.is_empty() {
            return Err(PyValueError::new_err("sequence needs at least one interval"));
        }
        if transitions.len() != intervals.len() || transitions.iter().any(|row| row.len() != intervals.len()) {
            return Err(PyValueError::new_err("sequence transition matrix must be square and match the interval count"));
        }
        let transitions = transitions
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|value| {
                        if value < 0 {
                            return Err(PyValueError::new_err("sequence transition times must be non-negative"));
                        }
                        checked_i32(value, "sequence transition time")
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        let realizations = intervals.iter().map(|interval| self.native_interval_realizations(interval)).collect::<PyResult<Vec<_>>>()?;
        let (specs, owners) = self.flatten_native_intervals(&intervals)?;
        let mut distinct = HashSet::with_capacity(specs.len());
        if specs.iter().any(|spec| !distinct.insert(spec.interval)) {
            return Err(PyValueError::new_err("sequence intervals must be distinct"));
        }
        for left in 0..specs.len() {
            for right in (left + 1)..specs.len() {
                if owners[left] == owners[right] {
                    continue;
                }
                self.post_native_pair_order(
                    &specs[left],
                    &specs[right],
                    transitions[owners[left]][owners[right]],
                    transitions[owners[right]][owners[left]],
                );
            }
        }
        let index = self.native_sequences;
        self.native_sequences += 1;
        Ok(PySequenceVar { model_id: self.id, index: index as u32, intervals, realizations })
    }

    fn cumulative(&mut self, starts: &Bound<'_, PyAny>, durations: Vec<i64>, heights: Vec<i64>, capacity: i64) -> PyResult<()> {
        let starts = ids_for(self.id, &var_list_from_py(starts)?)?;
        if starts.len() != durations.len() || starts.len() != heights.len() {
            return Err(PyValueError::new_err("starts, durations, and heights must have the same length"));
        }
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Cumulative {
            starts,
            durations,
            demands: heights,
            capacity,
        }));
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
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::CumulativeVar {
            starts,
            durations,
            demands: heights,
            capacity,
        }));
        Ok(())
    }

    fn bin_packing(&mut self, items: &Bound<'_, PyAny>, sizes: Vec<i64>, capacities: Vec<i64>) -> PyResult<()> {
        let items = ids_for(self.id, &var_list_from_py(items)?)?;
        if items.len() != sizes.len() {
            return Err(PyValueError::new_err("items and sizes must have the same length"));
        }
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::BinPacking { items, sizes, capacities }));
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
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Knapsack {
            variables: vars,
            weights,
            profits,
            weight_relation: parse_relation(weight_relation)?,
            weight_limit,
            profit_relation: parse_relation(profit_relation)?,
            profit_limit,
        }));
        Ok(())
    }

    fn circuit(&mut self, successors: &Bound<'_, PyAny>) -> PyResult<()> {
        let successors = ids_for(self.id, &var_list_from_py(successors)?)?;
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Circuit { successors, cutset: false }));
        Ok(())
    }

    /// Create `count` ordered list variables over `items`.
    ///
    /// With `optional=True`, an extra hidden pool list is added that no returned
    /// handle references, so items may remain unassigned to the visible lists.
    #[pyo3(signature = (items, count, *, optional=false))]
    fn list_vars(&mut self, items: Vec<i32>, count: usize, optional: bool) -> PyResult<Vec<PyListVar>> {
        self.create_semantic_lists(items, count, optional, shared_model::ListOrdering::Ordered)
    }

    /// Declare unordered set variables partitioning the item universe.
    #[pyo3(signature = (items, count, *, optional=false))]
    fn set_vars(&mut self, items: Vec<i32>, count: usize, optional: bool) -> PyResult<Vec<PyListVar>> {
        self.create_semantic_lists(items, count, optional, shared_model::ListOrdering::Unordered)
    }

    /// Mint a collection-scoped scalar to reify a scan total into: bind it with
    /// `add(scan_sum(r, ...) == v)`, then use `v` in a linear objective. Requires
    /// an active `list_vars` generation.
    fn list_value(&mut self) -> PyResult<ListValue> {
        if !self.semantic.has_lists() {
            return Err(PyValueError::new_err("call list_vars() before list_value()"));
        }
        let value = self.semantic.reified_values.len();
        self.semantic.reified_values.push(None);
        Ok(ListValue { model_id: self.id, gen: self.semantic.list_generation, value })
    }

    /// Precedence over list items or interval variables.
    fn precedence(&mut self, before: &Bound<'_, PyAny>, after: &Bound<'_, PyAny>) -> PyResult<()> {
        if let (Ok(a), Ok(b)) = (before.extract::<PyRef<'_, PyIntervalVar>>(), after.extract::<PyRef<'_, PyIntervalVar>>()) {
            if a.kind != b.kind {
                return Err(PyValueError::new_err("cannot mix native intervals and schedule intervals in one precedence"));
            }
            if a.kind == PyIntervalKind::Native {
                let before = self.native_interval_realizations(&a)?;
                let after = self.native_interval_realizations(&b)?;
                for first in &before {
                    for second in &after {
                        self.post_native_precedence(first, second, 0);
                    }
                }
            } else {
                self.check_interval_scope(&a)?;
                self.check_interval_scope(&b)?;
                self.semantic.model_mut().add_constraint(shared_model::Constraint::IntervalPrecedence {
                    before: shared_model::IntervalVarRef(a.index as usize),
                    after: shared_model::IntervalVarRef(b.index as usize),
                });
            }
            return Ok(());
        }
        let before =
            before.extract::<i32>().map_err(|_| PyTypeError::new_err("precedence expects two item ids or two IntervalVar handles"))?;
        let after =
            after.extract::<i32>().map_err(|_| PyTypeError::new_err("precedence expects two item ids or two IntervalVar handles"))?;
        let lists = self.semantic.list_scope();
        self.semantic.model_mut().add_constraint(shared_model::Constraint::ItemPrecedence { lists, before, after });
        Ok(())
    }

    /// Require two items to share a list (same vehicle, same bin).
    fn same_list(&mut self, a: i32, b: i32) {
        let lists = self.semantic.list_scope();
        self.semantic.model_mut().add_constraint(shared_model::Constraint::SameList { lists, a, b });
    }

    /// Require two items to be assigned to different lists or sets.
    fn different_list(&mut self, a: i32, b: i32) {
        self.semantic
            .model_mut()
            .add_constraint(shared_model::Constraint::CollectionGlobal(list::GlobalConstraint::DifferentList { a, b }));
    }

    /// Require all supplied items to share one list or set.
    fn all_same_list(&mut self, items: Vec<i32>) {
        self.semantic
            .model_mut()
            .add_constraint(shared_model::Constraint::CollectionGlobal(list::GlobalConstraint::AllSameList { items: Arc::new(items) }));
    }

    /// Require all supplied items to have distinct owner lists or sets.
    fn all_different_lists(&mut self, items: Vec<i32>) {
        self.semantic.model_mut().add_constraint(shared_model::Constraint::CollectionGlobal(list::GlobalConstraint::AllDifferentLists {
            items: Arc::new(items),
        }));
    }

    /// Bound the absolute difference between the owner-list indices.
    #[pyo3(signature = (a, b, *, min=0, max=usize::MAX))]
    fn list_distance(&mut self, a: i32, b: i32, min: usize, max: usize) {
        self.semantic.model_mut().add_constraint(shared_model::Constraint::CollectionGlobal(list::GlobalConstraint::ListDistance {
            a,
            b,
            min,
            max,
        }));
    }

    /// Create one fixed-duration native interval.
    #[pyo3(signature = (duration, horizon, *, optional=false, name=None))]
    fn interval(&mut self, duration: i64, horizon: i64, optional: bool, name: Option<String>) -> PyResult<PyIntervalVar> {
        self.create_native_interval(duration, horizon, optional, name)
    }

    /// Create fixed-duration native intervals.
    #[pyo3(signature = (durations, horizon, *, optional=false, name=None))]
    fn intervals(&mut self, durations: Vec<i64>, horizon: i64, optional: bool, name: Option<String>) -> PyResult<Vec<PyIntervalVar>> {
        let mut intervals = Vec::with_capacity(durations.len());
        for (i, duration) in durations.into_iter().enumerate() {
            let var_name = name.as_ref().map(|prefix| format!("{prefix}[{i}]"));
            intervals.push(self.create_native_interval(duration, horizon, optional, var_name)?);
        }
        Ok(intervals)
    }

    /// The `alternative` constraint over optional intervals the caller created:
    /// exactly one member executes while the master is present, and the
    /// returned master synchronises with it. An optional master may be absent,
    /// in which case every member is absent too.
    #[pyo3(signature = (members, *, optional=false, name=None))]
    fn alternative(&mut self, members: &Bound<'_, PyAny>, optional: bool, name: Option<String>) -> PyResult<PyIntervalVar> {
        let members = interval_list_from_py(members)?;
        if members.is_empty() {
            return Err(PyValueError::new_err("alternative needs at least one member interval"));
        }
        let mut ids: Vec<PythonIntervalRef> = Vec::with_capacity(members.len());
        let mut starts: Vec<IntVarRef> = Vec::with_capacity(members.len());
        let mut presences: Vec<IntVarRef> = Vec::with_capacity(members.len());
        let mut durations: Vec<i64> = Vec::with_capacity(members.len());
        let (mut start_lo, mut start_hi) = (i32::MAX, i32::MIN);
        for member in &members {
            let spec = self.native_interval_spec(member)?;
            let Some(presence) = spec.presence else {
                return Err(PyValueError::new_err("alternative members must be optional intervals (Model.interval(..., optional=True))"));
            };
            if ids.contains(&spec.interval) {
                return Err(PyValueError::new_err("alternative members must be distinct intervals"));
            }
            start_lo = start_lo.min(self.domain_min(spec.start));
            start_hi = start_hi.max(self.domain_max(spec.start));
            ids.push(spec.interval);
            starts.push(spec.start);
            presences.push(presence);
            durations.push(i64::from(spec.duration));
        }

        let shared_start = self.semantic.model_mut().int_range(start_lo, start_hi);
        self.register_native_backing_var(shared_start, name.as_ref().map(|name| format!("{name}.start")));
        let master_presence = optional.then(|| self.semantic.model_mut().bool_var());
        if let Some(presence) = master_presence {
            self.register_native_backing_var(presence, name.as_ref().map(|name| format!("{name}.presence")));
            let mut terms = presences.iter().map(|&member| (1, member)).collect::<Vec<_>>();
            terms.push((-1, presence));
            self.add_integer_constraint(Constraint::Linear { terms, relation: Relation::Eq, rhs: 0 });
            for (&start, &member_presence) in starts.iter().zip(&presences) {
                self.add_integer_constraint(Constraint::Intension(expr::imp(
                    expr::eq(expr::var(member_presence), expr::int(1)),
                    expr::eq(expr::var(shared_start), expr::var(start)),
                )));
            }
        } else {
            self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::AlternativeChannel {
                shared_start,
                starts,
                durations: durations.clone(),
                presences: presences.clone(),
            }));
        }

        let name = name.unwrap_or_else(|| format!("alternative{}", shared_start.0));
        Ok(PyIntervalVar {
            model_id: self.id,
            gen: 0,
            index: u32::MAX,
            kind: PyIntervalKind::Native,
            interval: None,
            start: Some(shared_start),
            presence: master_presence,
            duration: None,
            alternative: Some(AlternativeInterval { name, members, presences, durations }),
        })
    }

    /// Create intervals whose mode is selected from `(machine, duration)` pairs.
    fn alternatives(&mut self, modes: Vec<Vec<(usize, i64)>>, horizon: i64) -> PyResult<Vec<PyIntervalVar>> {
        if modes.iter().any(|m| m.is_empty()) {
            return Err(PyValueError::new_err("each interval needs at least one (machine, duration) mode"));
        }
        let start_windows = modes
            .iter()
            .map(|options| {
                options
                    .iter()
                    .map(|&(_, duration)| checked_interval_start_max(horizon, duration).map(i64::from))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.enter_schedule_mode()?;
        let model = self.semantic.model_mut();
        model.clear_interval_family();
        let mut intervals = Vec::with_capacity(modes.len());
        for (options, windows) in modes.iter().zip(&start_windows) {
            let start_max = windows.iter().copied().max().unwrap_or(0);
            let interval = model.interval(0, start_max, 0);
            for (&(machine, duration), &mode_start_max) in options.iter().zip(windows) {
                model.add_interval_mode(interval, machine, duration, Some((0, mode_start_max))).map_err(PyValueError::new_err)?;
            }
            intervals.push(interval);
        }
        self.semantic.objectives.retain(|objective| !matches!(objective, shared_model::Objective::Makespan { .. }));
        self.semantic.objectives.push(shared_model::Objective::Makespan { minimize: true, intervals: intervals.clone() });
        let gen = self.semantic.schedule_generation;
        Ok((0..modes.len())
            .map(|i| PyIntervalVar {
                model_id: self.id,
                gen,
                index: i as u32,
                kind: PyIntervalKind::Schedule,
                interval: None,
                start: None,
                presence: None,
                duration: None,
                alternative: None,
            })
            .collect())
    }

    /// Create fixed-duration intervals directly in the compact schedule IR.
    ///
    /// Unlike `intervals`, this does not allocate CP variables or propagators at
    /// model-construction time. It is the intended primitive for the scheduling
    /// convenience API, whose backend is selected only when `solve` is called.
    #[pyo3(signature = (durations, horizon, *, optional=false))]
    fn schedule_intervals(&mut self, durations: Vec<i64>, horizon: i64, optional: bool) -> PyResult<Vec<PyIntervalVar>> {
        if durations.is_empty() {
            return Err(PyValueError::new_err("a schedule needs at least one interval"));
        }
        for &duration in &durations {
            checked_interval_start_max(horizon, duration)?;
        }
        self.enter_schedule_mode()?;
        let model = self.semantic.model_mut();
        model.clear_interval_family();
        let intervals = durations
            .iter()
            .map(|&duration| {
                let start_max = horizon - duration;
                if optional {
                    model.optional_interval(0, start_max, duration)
                } else {
                    model.interval(0, start_max, duration)
                }
            })
            .collect::<Vec<_>>();
        self.semantic.objectives.retain(|objective| !matches!(objective, shared_model::Objective::Makespan { .. }));
        self.semantic.objectives.push(shared_model::Objective::Makespan { minimize: true, intervals });
        let gen = self.semantic.schedule_generation;
        Ok(durations
            .into_iter()
            .enumerate()
            .map(|(index, duration)| PyIntervalVar {
                model_id: self.id,
                gen,
                index: index as u32,
                kind: PyIntervalKind::Schedule,
                interval: None,
                start: None,
                presence: None,
                duration: Some(duration),
                alternative: None,
            })
            .collect())
    }

    /// Moded intervals that choose the same machine never overlap.
    fn no_overlap_by_machine(&mut self) -> PyResult<()> {
        if !self.semantic.has_intervals() {
            return Err(PyValueError::new_err("create alternatives before no_overlap_by_machine"));
        }
        self.semantic.model_mut().add_constraint(shared_model::Constraint::IntervalResource(list::Resource::MachineNoOverlap));
        Ok(())
    }

    /// Keep the makespan objective explicit in Python while the Rust model owns
    /// the scheduling semantics.
    #[pyo3(signature = (intervals=None))]
    fn minimize_makespan(&mut self, intervals: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        if !self.native_intervals.is_empty()
            || intervals
                .is_some_and(|items| interval_list_from_py(items).is_ok_and(|ivs| ivs.iter().any(|iv| iv.kind == PyIntervalKind::Native)))
        {
            let selected = if let Some(intervals) = intervals {
                let intervals = interval_list_from_py(intervals)?;
                self.flatten_native_intervals(&intervals)?.0
            } else {
                self.native_intervals.clone()
            };
            return self.set_native_makespan_objective(&selected);
        }
        if let Some(intervals) = intervals {
            for iv in interval_list_from_py(intervals)? {
                self.check_interval_scope(&iv)?;
            }
        }
        if !self.semantic.has_intervals() {
            return Err(PyValueError::new_err("create intervals before minimize_makespan"));
        }
        if !self.semantic.schedule_makespan() {
            let intervals = (0..self.semantic.model().intervals().len()).map(shared_model::IntervalVarRef).collect();
            self.semantic.objectives.push(shared_model::Objective::Makespan { minimize: true, intervals });
        }
        Ok(())
    }

    /// A renewable resource of `capacity`: `demands` are `(interval, amount)`
    /// pairs whose total over any instant may not exceed the capacity.
    fn resource(&mut self, demands: Vec<(PyIntervalVar, i64)>, capacity: i64) -> PyResult<()> {
        if demands.iter().all(|(iv, _)| iv.kind == PyIntervalKind::Native) {
            let capacity = checked_nonnegative_i32(capacity, "cumulative capacity")?;
            let mut expanded = Vec::new();
            for (iv, amount) in &demands {
                let amount = checked_nonnegative_i32(*amount, "cumulative demand")?;
                expanded.extend(self.native_interval_realizations(iv)?.into_iter().map(|interval| (interval, amount)));
            }
            self.post_native_cumulative(expanded, capacity, &[]);
        } else if demands.iter().all(|(iv, _)| iv.kind == PyIntervalKind::Schedule) {
            for (iv, _) in &demands {
                self.check_interval_scope(iv)?;
            }
            let demands = demands.iter().map(|(iv, amount)| (iv.index as usize, *amount)).collect();
            self.semantic
                .model_mut()
                .add_constraint(shared_model::Constraint::IntervalResource(list::Resource::Cumulative { demands, capacity }));
        } else {
            return Err(PyValueError::new_err("cannot mix native intervals and schedule intervals in one resource"));
        }
        Ok(())
    }

    /// Renewable resource with piecewise-constant capacity overrides on
    /// half-open `[start, end)` calendar segments.
    fn resource_calendar(
        &mut self,
        demands: Vec<(PyIntervalVar, i64)>,
        default_capacity: i64,
        calendar: Vec<(i64, i64, i64)>,
    ) -> PyResult<()> {
        let default_capacity = checked_nonnegative_i32(default_capacity, "default cumulative capacity")?;
        let mut calendar = calendar
            .into_iter()
            .map(|(start, end, capacity)| {
                let start = checked_i32(start, "calendar segment start")?;
                let end = checked_i32(end, "calendar segment end")?;
                if start >= end {
                    return Err(PyValueError::new_err("calendar segments need start < end"));
                }
                Ok((start, end, checked_nonnegative_i32(capacity, "calendar capacity")?))
            })
            .collect::<PyResult<Vec<_>>>()?;
        calendar.sort_unstable_by_key(|&(start, end, _)| (start, end));
        if calendar.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(PyValueError::new_err("calendar segments must not overlap"));
        }

        let mut expanded = Vec::new();
        for (interval, demand) in &demands {
            let demand = checked_nonnegative_i32(*demand, "cumulative demand")?;
            expanded.extend(self.native_interval_realizations(interval)?.into_iter().map(|interval| (interval, demand)));
        }
        let horizon = expanded
            .iter()
            .map(|(interval, _)| self.domain_max(interval.start).saturating_add(interval.duration))
            .max()
            .unwrap_or(0)
            .max(0);
        let capacity = calendar.iter().map(|segment| segment.2).fold(default_capacity, i32::max);
        let mut points = vec![0, horizon];
        for &(start, end, _) in &calendar {
            let start = start.clamp(0, horizon);
            let end = end.clamp(0, horizon);
            if start < end {
                points.extend([start, end]);
            }
        }
        points.sort_unstable();
        points.dedup();
        let mut blockers = Vec::new();
        for window in points.windows(2) {
            let (start, end) = (window[0], window[1]);
            let available = calendar
                .iter()
                .find_map(|&(segment_start, segment_end, value)| (segment_start <= start && start < segment_end).then_some(value))
                .unwrap_or(default_capacity);
            if available < capacity {
                blockers.push((start, end - start, capacity - available));
            }
        }
        self.post_native_cumulative(expanded, capacity, &blockers);
        Ok(())
    }

    /// State/resource function. Equal-state intervals may overlap. Intervals
    /// requiring different states are ordered with transition-dependent setup.
    fn state_function(&mut self, states: Vec<(PyIntervalVar, usize)>, transitions: Vec<Vec<i64>>) -> PyResult<()> {
        let state_count = transitions.len();
        if state_count == 0 || transitions.iter().any(|row| row.len() != state_count) {
            return Err(PyValueError::new_err("state transition matrix must be non-empty and square"));
        }
        let transitions = transitions
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|value| {
                        if value < 0 {
                            return Err(PyValueError::new_err("state transition times must be non-negative"));
                        }
                        checked_i32(value, "state transition time")
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut expanded = Vec::new();
        let mut distinct = HashSet::new();
        for (interval, state) in states {
            if state >= state_count {
                return Err(PyValueError::new_err("state function references an unknown state"));
            }
            for realization in self.native_interval_realizations(&interval)? {
                if !distinct.insert(realization.interval) {
                    return Err(PyValueError::new_err("state function intervals must be distinct"));
                }
                expanded.push((realization, state));
            }
        }
        for left in 0..expanded.len() {
            for right in (left + 1)..expanded.len() {
                let (first, first_state) = &expanded[left];
                let (second, second_state) = &expanded[right];
                if first_state != second_state {
                    self.post_native_pair_order(
                        first,
                        second,
                        transitions[*first_state][*second_state],
                        transitions[*second_state][*first_state],
                    );
                }
            }
        }
        Ok(())
    }

    #[pyo3(signature = (*, search=None, assumptions=None, hints=None, branch_order=None, on_incumbent=None, verbose=false, time_limit=None, seed=0, threads=1, engine="auto", conflict_budget=None, list_hint=None, max_iterations=None, profile=false, memory_limit_mb=None, schedule_cdcl=false, routing_two_way=true, routing_nearest_neighbor=true, routing_warm_start=true, linear_backend="auto", lp_root_ms=50, lp_node_ms=0, lp_node_depth_interval=8, lp_max_variables=2000, lp_max_rows=1000, lp_max_nonzeros=100000, lp_min_coverage_percent=1, lp_phase_max_variables=1000))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &self,
        py: Python<'_>,
        search: Option<&Bound<'_, PyAny>>,
        assumptions: Option<&Bound<'_, PyAny>>,
        hints: Option<&Bound<'_, PyAny>>,
        branch_order: Option<&Bound<'_, PyAny>>,
        on_incumbent: Option<&Bound<'_, PyAny>>,
        verbose: bool,
        time_limit: Option<u64>,
        seed: u64,
        threads: usize,
        engine: &str,
        conflict_budget: Option<u64>,
        list_hint: Option<&Bound<'_, PyAny>>,
        max_iterations: Option<u64>,
        profile: bool,
        memory_limit_mb: Option<u64>,
        schedule_cdcl: bool,
        routing_two_way: bool,
        routing_nearest_neighbor: bool,
        routing_warm_start: bool,
        linear_backend: &str,
        lp_root_ms: u64,
        lp_node_ms: u64,
        lp_node_depth_interval: usize,
        lp_max_variables: usize,
        lp_max_rows: usize,
        lp_max_nonzeros: usize,
        lp_min_coverage_percent: usize,
        lp_phase_max_variables: usize,
    ) -> PyResult<PySolution> {
        if threads == 0 {
            return Err(PyValueError::new_err("threads must be a positive integer"));
        }
        if memory_limit_mb == Some(0) {
            return Err(PyValueError::new_err("memory_limit_mb must be a positive integer when provided"));
        }
        let engine = parse_engine(engine)?;
        let list_hint = list_hint.filter(|obj| !obj.is_none()).map(|obj| self.parse_list_hint(obj)).transpose()?;
        let objective_specs = objective_specs(&self.objective, &self.then_objectives);
        if let Some(callback) = on_incumbent {
            if !callback.is_callable() {
                return Err(PyTypeError::new_err("on_incumbent must be callable"));
            }
        }
        let has_collection = self.semantic.has_collection();
        if verbose && !has_collection {
            verbose_start(self.names.len(), self.semantic.model().constraints().len(), !objective_specs.is_empty());
            if let Some(objective) = objective_specs.first() {
                println!("  direction: {}", if objective.minimizing { "min" } else { "max" });
                println!("  expression: {}", objective.expr.text);
            }
            if engine == PythonEngine::Ls {
                println!("  engine: local-search");
            }
        }
        let assumptions = assumptions_from_py(self.id, assumptions)?;
        let hints = hint_pairs_from_py(self.id, hints)?;
        let mut guidance = branch_order_from_py(self.id, branch_order)?;
        if search.is_some_and(|search| !search.is_none()) {
            for variable in search_ids(self, search, None)? {
                if !guidance.contains(&variable.0) {
                    guidance.push(variable.0);
                }
            }
        }
        let package = self.solve_package();
        let request = SolveRequest {
            mode: engine.solve_mode(),
            seed,
            threads,
            limits: SolveLimits {
                time: time_limit.map(Duration::from_secs),
                memory_bytes: memory_limit_mb.map(|limit| limit.saturating_mul(1024 * 1024)),
                conflicts: conflict_budget,
                iterations: max_iterations,
            },
            profile,
            assumptions,
            hints,
            list_hint,
            branch_order: guidance,
            publish_incumbent_assignments: on_incumbent.is_some(),
            schedule_cdcl,
            routing: RoutingControls {
                two_way: routing_two_way,
                nearest_neighbor: routing_nearest_neighbor,
                warm_start: routing_warm_start,
            },
            linear: linear_controls(
                linear_backend,
                lp_root_ms,
                lp_node_ms,
                lp_node_depth_interval,
                lp_max_variables,
                lp_max_rows,
                lp_max_nonzeros,
                lp_min_coverage_percent,
                lp_phase_max_variables,
            )?,
            ..SolveRequest::default()
        };
        if verbose && has_collection {
            verbose_collection_start(&self.semantic, &request, time_limit);
        }
        let on_incumbent = on_incumbent.map(|cb| cb.clone().unbind());
        let num_vars = self.names.len();
        let primary_sense = if self.semantic.primary_list_sense().unwrap_or(true) { "min" } else { "max" };
        let run = with_interrupts(py, move || {
            let mut sink = PythonSolveEventSink::new(verbose, on_incumbent, primary_sense);
            let result = solve_model_with_external_stop(&package, &request, &SIGINT_TRIPPED, &mut sink);
            if let Some(error) = sink.integer.callback_error {
                return Err(error);
            }
            result
                .map(|result| {
                    let engine = collection_result_engine(&result);
                    CollectionRun { result, engine, events: sink.collection.summary }
                })
                .map_err(integer_solve_error)
        })??;
        let collection_engine = matches!(
            run.engine,
            EngineKind::RoutingExact
                | EngineKind::ListExact
                | EngineKind::ScheduleExact
                | EngineKind::ListLocalSearch
                | EngineKind::RoutingLocalSearch
                | EngineKind::ScheduleLocalSearch
        );
        if has_collection || collection_engine {
            let solution = self.collection_solution_from_result(&run, profile)?;
            if verbose {
                verbose_collection_finish(&solution, &run, profile);
            }
            Ok(solution)
        } else {
            let result_engine = if engine == PythonEngine::Ls { EngineKind::IntegerLocalSearch } else { EngineKind::IntegerExact };
            let mut solution = integer_solution_from_result(&run.result, &objective_specs, num_vars, result_engine, profile)?;
            attach_native_interval_solution(&mut solution, &self.native_intervals);
            if verbose {
                verbose_finish(&solution);
            }
            Ok(solution)
        }
    }

    /// A soft-constraint group for MUS extraction. Use as a context manager:
    /// every constraint posted inside the `with` block is guarded by one fresh
    /// selector, and [`mus`](PyModel::mus) reports which groups form an unsat
    /// core. `name` defaults to `c0, c1, …`. Constraints posted outside any
    /// `soft` block are hard (always active).
    ///
    /// ```python
    /// with model.soft("c1"):
    ///     model.sum([a], ">=", 1)
    /// core = model.mus()  # -> ["c1", ...] or None if satisfiable
    /// ```
    #[pyo3(signature = (name=None))]
    fn soft(slf: Py<Self>, name: Option<String>) -> PySoftGroup {
        PySoftGroup { model: slf, name }
    }

    /// Extract a minimal unsatisfiable subset of the [`soft`](PyModel::soft)
    /// groups (with all hard constraints active). Returns the list of group names
    /// in the MUS, or `None` if the model is satisfiable. Raises on time-out.
    #[pyo3(signature = (time_limit=None))]
    fn mus(&self, py: Python<'_>, time_limit: Option<u64>) -> PyResult<Option<Vec<String>>> {
        if self.semantic.has_collection() {
            return Err(PyValueError::new_err("mus() is not supported for list/interval models"));
        }
        let package = self.integer_package();
        let vars = self.decision_var_ids().into_iter().map(|variable| variable.0).collect::<Vec<_>>();
        let selectors = self.mus_selectors.iter().map(|(_, selector)| selector.0).collect::<Vec<_>>();
        let request = SolveRequest {
            limits: SolveLimits { time: time_limit.map(Duration::from_secs), ..SolveLimits::default() },
            ..SolveRequest::default()
        };
        let result =
            with_interrupts(py, move || extract_model_mus_with_external_stop(&package, &vars, &selectors, &request, &SIGINT_TRIPPED))?
                .map_err(integer_solve_error)?;
        match result {
            ModelMusResult::Sat(_) => Ok(None),
            ModelMusResult::Interrupted => Err(PyTimeoutError::new_err("mus() timed out")),
            ModelMusResult::Mus(core) => Ok(Some(core.into_iter().map(|selector| self.selector_name(IntVarRef(selector))).collect())),
        }
    }

    /// Enumerate the full infeasibility landscape of the [`soft`](PyModel::soft)
    /// groups (MARCO): returns `(muses, msses, complete)` where `muses` is every
    /// minimal unsatisfiable subset and `msses` every maximal satisfiable subset
    /// (each as a list of group names). `complete` is `False` if the search
    /// stopped early (time-out or `limit` reached). `limit` caps the total number
    /// of MUS + MSS returned (there can be exponentially many).
    #[pyo3(signature = (time_limit=None, limit=None))]
    fn enumerate_mus(&self, py: Python<'_>, time_limit: Option<u64>, limit: Option<usize>) -> PyResult<PyMusEnumeration> {
        if self.semantic.has_collection() {
            return Err(PyValueError::new_err("enumerate_mus() is not supported for list/interval models"));
        }
        let package = self.integer_package();
        let vars = self.decision_var_ids().into_iter().map(|variable| variable.0).collect::<Vec<_>>();
        let selectors = self.mus_selectors.iter().map(|(_, selector)| selector.0).collect::<Vec<_>>();
        let request = SolveRequest {
            limits: SolveLimits { time: time_limit.map(Duration::from_secs), ..SolveLimits::default() },
            ..SolveRequest::default()
        };
        let result: ModelMusEnumeration = with_interrupts(py, move || {
            enumerate_model_mus_with_external_stop(&package, &vars, &selectors, limit, &request, &SIGINT_TRIPPED)
        })?
        .map_err(integer_solve_error)?;
        let names = |group: &[usize]| group.iter().map(|&selector| self.selector_name(IntVarRef(selector))).collect::<Vec<_>>();
        Ok(PyMusEnumeration {
            muses: result.muses.iter().map(|m| names(m)).collect(),
            msses: result.msses.iter().map(|m| names(m)).collect(),
            complete: result.complete,
        })
    }

    /// Explain a MUS at sub-constraint granularity: for each core
    /// constraint that refutes by propagation, the *specific* atoms it reasoned
    /// about (`"x >= 4"`), not the whole global. Returns a `{name: [atom, ...]}`
    /// dict, or `None` when the model is satisfiable or the core needs search to
    /// refute (use [`mus`](PyModel::mus) for the constraint-level core then).
    #[pyo3(signature = (time_limit=None))]
    fn explain_mus(&self, py: Python<'_>, time_limit: Option<u64>) -> PyResult<Option<HashMap<String, Vec<String>>>> {
        if self.semantic.has_collection() {
            return Err(PyValueError::new_err("explain_mus() is not supported for list/interval models"));
        }
        let package = self.integer_package();
        let vars = self.decision_var_ids().into_iter().map(|variable| variable.0).collect::<Vec<_>>();
        let selectors = self.mus_selectors.iter().map(|(_, selector)| selector.0).collect::<Vec<_>>();
        let request = SolveRequest {
            limits: SolveLimits { time: time_limit.map(Duration::from_secs), ..SolveLimits::default() },
            ..SolveRequest::default()
        };
        let explained = with_interrupts(py, move || {
            let extracted = extract_model_mus_with_external_stop(&package, &vars, &selectors, &request, &SIGINT_TRIPPED)?;
            match extracted {
                ModelMusResult::Mus(core) if !core.is_empty() => {
                    explain_model_mus_with_external_stop(&package, &vars, &core, &request, &SIGINT_TRIPPED)
                }
                ModelMusResult::Sat(_) | ModelMusResult::Mus(_) | ModelMusResult::Interrupted => Ok(None),
            }
        })?
        .map_err(integer_solve_error)?;
        Ok(explained.map(|constraints| {
            constraints
                .into_iter()
                .map(|(selector, atoms)| {
                    (
                        self.selector_name(IntVarRef(selector)),
                        atoms.iter().map(|atom| self.atom_text(IntVarRef(atom.variable), atom.relation, atom.value)).collect(),
                    )
                })
                .collect()
        }))
    }

    #[pyo3(signature = (search=None))]
    fn count_solutions(&self, search: Option<&Bound<'_, PyAny>>) -> PyResult<u64> {
        let package = self.integer_package();
        let variables = search_ids(self, search, None)?.into_iter().map(|variable| variable.0).collect::<Vec<_>>();
        let request = SolveRequest::default();
        let stop = AtomicBool::new(false);
        count_model_solutions_with_external_stop(&package, &variables, &request, &stop).map_err(integer_solve_error)
    }

    /// Set the primary minimization objective.
    fn minimize(&mut self, objective: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Some(term) = collection_objective_term(objective) {
            self.push_collection_tier(&term, true, true)?;
            return Ok(());
        }
        self.set_integer_objective(objective, true)
    }

    /// Set the primary maximization objective.
    fn maximize(&mut self, objective: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Some(term) = collection_objective_term(objective) {
            self.push_collection_tier(&term, false, true)?;
            return Ok(());
        }
        self.set_integer_objective(objective, false)
    }

    /// Append a lower-priority lexicographic tier.
    fn then_minimize(&mut self, objective: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Some(term) = collection_objective_term(objective) {
            return self.push_collection_tier(&term, true, false);
        }
        self.push_integer_tier(objective, true)
    }

    fn then_maximize(&mut self, objective: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Some(term) = collection_objective_term(objective) {
            return self.push_collection_tier(&term, false, false);
        }
        self.push_integer_tier(objective, false)
    }

    fn session(&self) -> PyResult<PySolveSession> {
        if self.semantic.has_collection() {
            return Err(PyValueError::new_err("SolveSession is currently supported for integer exact models"));
        }
        let objectives = objective_specs(&self.objective, &self.then_objectives);
        let search = self.decision_var_ids().into_iter().map(|variable| variable.0).collect();
        let session = SemanticSolveSession::new(self.integer_package()).map_err(integer_solve_error)?;
        Ok(PySolveSession {
            id: self.id,
            names: self.names.clone(),
            objectives,
            search,
            native_intervals: self.native_intervals.clone(),
            session,
        })
    }

    fn __repr__(&self) -> String {
        format!("Model(num_vars={}, num_constraints={})", self.names.len(), self.semantic.model().constraints().len())
    }
}

impl PyModel {
    fn create_semantic_lists(
        &mut self,
        items: Vec<i32>,
        count: usize,
        optional: bool,
        ordering: shared_model::ListOrdering,
    ) -> PyResult<Vec<PyListVar>> {
        if items.is_empty() {
            return Err(PyValueError::new_err("list items cannot be empty"));
        }
        if count == 0 {
            return Err(PyValueError::new_err("need at least one list"));
        }
        let mut seen = HashSet::with_capacity(items.len());
        if let Some(duplicate) = items.iter().find(|value| !seen.insert(**value)) {
            return Err(PyValueError::new_err(format!("list items have a duplicate value {duplicate}; items must be distinct")));
        }

        let model = self.semantic.model_mut();
        model.clear_list_family();
        let mut lists = Vec::with_capacity(count + usize::from(optional));
        for _ in 0..count {
            lists.push(match ordering {
                shared_model::ListOrdering::Ordered => model.list(items.clone()),
                shared_model::ListOrdering::Unordered => model.unordered_list(items.clone()),
            });
        }
        if optional {
            lists.push(model.remainder_list(items.clone(), ordering));
        }
        model.add_constraint(shared_model::Constraint::ListPartition { lists, items });

        let list_generation = self.semantic.list_generation + 1;
        self.semantic.list_generation = list_generation;
        self.semantic.objectives.retain(|objective| !matches!(objective, shared_model::Objective::ListTerms { .. }));
        self.semantic.reified_values.clear();
        let unordered = ordering == shared_model::ListOrdering::Unordered;
        Ok((0..count).map(|index| PyListVar { model_id: self.id, gen: list_generation, index: index as u32, unordered }).collect())
    }

    /// Create or reuse a fixed semantic integer variable.
    fn const_var(&mut self, value: i32) -> IntVarRef {
        if let Some(&variable) = self.constants.get(&value) {
            return variable;
        }
        let id = self.semantic.model_mut().int_set(vec![value]);
        self.names.push(None);
        self.constants.insert(value, id);
        id
    }

    fn domain_min(&self, variable: IntVarRef) -> i32 {
        match &self.semantic.model().int_vars()[variable.0] {
            IntDomain::Bool => 0,
            IntDomain::Range { lo, .. } => *lo,
            IntDomain::Set(values) => values.iter().copied().min().expect("validated non-empty integer domain"),
        }
    }

    fn domain_max(&self, variable: IntVarRef) -> i32 {
        match &self.semantic.model().int_vars()[variable.0] {
            IntDomain::Bool => 1,
            IntDomain::Range { hi, .. } => *hi,
            IntDomain::Set(values) => values.iter().copied().max().expect("validated non-empty integer domain"),
        }
    }

    fn expression_bounds(&self, expression: &Expr) -> (i64, i64) {
        match expression {
            Expr::Constant(value) => (*value, *value),
            Expr::Variable(variable) => (i64::from(self.domain_min(*variable)), i64::from(self.domain_max(*variable))),
            Expr::Neg(value) => {
                let (lo, hi) = self.expression_bounds(value);
                (hi.saturating_neg(), lo.saturating_neg())
            }
            Expr::Abs(value) => {
                let (lo, hi) = self.expression_bounds(value);
                (0, lo.saturating_abs().max(hi.saturating_abs()))
            }
            Expr::Add(values) => values.iter().fold((0i64, 0i64), |(lo, hi), value| {
                let (value_lo, value_hi) = self.expression_bounds(value);
                (lo.saturating_add(value_lo), hi.saturating_add(value_hi))
            }),
            Expr::Sub(left, right) => {
                let (left_lo, left_hi) = self.expression_bounds(left);
                let (right_lo, right_hi) = self.expression_bounds(right);
                (left_lo.saturating_sub(right_hi), left_hi.saturating_sub(right_lo))
            }
            Expr::Mul(values) => values.iter().fold((1i64, 1i64), |(left_lo, left_hi), value| {
                let (right_lo, right_hi) = self.expression_bounds(value);
                let products = [
                    left_lo.saturating_mul(right_lo),
                    left_lo.saturating_mul(right_hi),
                    left_hi.saturating_mul(right_lo),
                    left_hi.saturating_mul(right_hi),
                ];
                (*products.iter().min().unwrap(), *products.iter().max().unwrap())
            }),
            Expr::Min(values) => {
                let bounds = values.iter().map(|value| self.expression_bounds(value)).collect::<Vec<_>>();
                (
                    bounds.iter().map(|bounds| bounds.0).min().unwrap_or(i32::MIN as i64),
                    bounds.iter().map(|bounds| bounds.1).min().unwrap_or(i32::MAX as i64),
                )
            }
            Expr::Max(values) => {
                let bounds = values.iter().map(|value| self.expression_bounds(value)).collect::<Vec<_>>();
                (
                    bounds.iter().map(|bounds| bounds.0).max().unwrap_or(i32::MIN as i64),
                    bounds.iter().map(|bounds| bounds.1).max().unwrap_or(i32::MAX as i64),
                )
            }
            Expr::IfThenElse(_, then_value, else_value) => {
                let then_bounds = self.expression_bounds(then_value);
                let else_bounds = self.expression_bounds(else_value);
                (then_bounds.0.min(else_bounds.0), then_bounds.1.max(else_bounds.1))
            }
            Expr::Div(_, _) | Expr::Mod(_, _) => (i32::MIN as i64, i32::MAX as i64),
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

    fn aux_for(&mut self, expression: Expr) -> IntVarRef {
        let (lo, hi) = self.expression_bounds(&expression);
        let clamp = |value: i64| value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        let variable = self.semantic.model_mut().int_range(clamp(lo), clamp(hi));
        self.names.push(None);
        self.add_integer_constraint(Constraint::Intension(expr::eq(expr::var(variable), expression)));
        variable
    }

    fn post_optional_no_overlap(&mut self, intervals: &[NativeIntervalSpec]) {
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::OptionalNoOverlap {
            starts: intervals.iter().map(|interval| interval.start).collect(),
            durations: intervals.iter().map(|interval| i64::from(interval.duration)).collect(),
            presences: intervals.iter().map(|interval| interval.presence).collect(),
        }));
    }

    fn register_native_backing_var(&mut self, var: IntVarRef, name: Option<String>) {
        while self.names.len() <= var.0 {
            self.names.push(None);
        }
        if name.is_some() || self.names[var.0].is_none() {
            self.names[var.0] = name;
        }
    }

    fn enter_native_interval_mode(&self) -> PyResult<()> {
        if self.semantic.has_lists() {
            return Err(PyValueError::new_err("model already has list variables; use one domain style per model"));
        }
        if self.semantic.has_intervals() {
            return Err(PyValueError::new_err("model already has schedule intervals; use native intervals or alternatives, not both"));
        }
        Ok(())
    }

    fn create_native_interval(&mut self, duration: i64, horizon: i64, optional: bool, name: Option<String>) -> PyResult<PyIntervalVar> {
        self.enter_native_interval_mode()?;
        let duration_i32 = checked_i32(duration, "interval duration")?;
        let start_max = checked_interval_start_max(horizon, duration)?;
        let interval = PythonIntervalRef(self.native_intervals.len());
        let start = self.semantic.model_mut().int_range(0, start_max);
        self.register_native_backing_var(start, name.as_ref().map(|name| format!("{name}.start")));
        let presence = optional.then(|| self.semantic.model_mut().bool_var());
        if let Some(var) = presence {
            self.register_native_backing_var(var, name.as_ref().map(|name| format!("{name}.presence")));
        }
        let index = self.native_intervals.len() as u32;
        self.native_intervals.push(NativeIntervalSpec { interval, start, presence, duration: duration_i32 });
        Ok(PyIntervalVar {
            model_id: self.id,
            gen: 0,
            index,
            kind: PyIntervalKind::Native,
            interval: Some(interval),
            start: Some(start),
            presence,
            duration: Some(duration),
            alternative: None,
        })
    }

    fn native_interval_spec(&self, iv: &PyIntervalVar) -> PyResult<NativeIntervalSpec> {
        if iv.model_id != self.id {
            return Err(PyValueError::new_err("this interval belongs to a different model"));
        }
        if iv.alternative.is_some() {
            return Err(PyValueError::new_err("an alternative master is not directly schedulable; schedule its members instead"));
        }
        if iv.kind != PyIntervalKind::Native {
            return Err(PyValueError::new_err("expected a native interval"));
        }
        let spec = self
            .native_intervals
            .get(iv.index as usize)
            .ok_or_else(|| PyValueError::new_err("this interval is stale; rebuild it from the current model"))?;
        if Some(spec.interval) != iv.interval {
            return Err(PyValueError::new_err("this interval is stale; rebuild it from the current model"));
        }
        Ok(spec.clone())
    }

    fn native_interval_specs(&self, intervals: &[PyIntervalVar]) -> PyResult<Vec<NativeIntervalSpec>> {
        intervals.iter().map(|iv| self.native_interval_spec(iv)).collect()
    }

    /// Expand a public interval to the concrete fixed-duration realizations
    /// consumed by integer scheduling globals. Plain intervals expand to one
    /// item, while an alternative expands to its optional members.
    fn native_interval_realizations(&self, interval: &PyIntervalVar) -> PyResult<Vec<NativeIntervalSpec>> {
        if interval.model_id != self.id {
            return Err(PyValueError::new_err("this interval belongs to a different model"));
        }
        if interval.kind != PyIntervalKind::Native {
            return Err(PyValueError::new_err("expected a native interval"));
        }
        if let Some(alternative) = &interval.alternative {
            return self.native_interval_specs(&alternative.members);
        }
        Ok(vec![self.native_interval_spec(interval)?])
    }

    fn flatten_native_intervals(&self, intervals: &[PyIntervalVar]) -> PyResult<(Vec<NativeIntervalSpec>, Vec<usize>)> {
        let mut realizations = Vec::new();
        let mut owners = Vec::new();
        for (owner, interval) in intervals.iter().enumerate() {
            let expanded = self.native_interval_realizations(interval)?;
            owners.extend(std::iter::repeat_n(owner, expanded.len()));
            realizations.extend(expanded);
        }
        Ok((realizations, owners))
    }

    fn post_native_precedence(&mut self, before: &NativeIntervalSpec, after: &NativeIntervalSpec, setup: i32) {
        let precedence = expr::le(
            expr::add(vec![expr::var(before.start), expr::int(i64::from(before.duration) + i64::from(setup))]),
            expr::var(after.start),
        );
        let mut active = Vec::new();
        if let Some(presence) = before.presence {
            active.push(expr::eq(expr::var(presence), expr::int(1)));
        }
        if let Some(presence) = after.presence {
            active.push(expr::eq(expr::var(presence), expr::int(1)));
        }
        let constraint = if active.is_empty() { precedence } else { expr::imp(expr::and(active), precedence) };
        self.add_integer_constraint(Constraint::Intension(constraint));
    }

    fn post_native_pair_order(&mut self, left: &NativeIntervalSpec, right: &NativeIntervalSpec, left_setup: i32, right_setup: i32) {
        let left_before = expr::le(
            expr::add(vec![expr::var(left.start), expr::int(i64::from(left.duration) + i64::from(left_setup))]),
            expr::var(right.start),
        );
        let right_before = expr::le(
            expr::add(vec![expr::var(right.start), expr::int(i64::from(right.duration) + i64::from(right_setup))]),
            expr::var(left.start),
        );
        let order = expr::or(vec![left_before, right_before]);
        let active = [left.presence, right.presence]
            .into_iter()
            .flatten()
            .map(|presence| expr::eq(expr::var(presence), expr::int(1)))
            .collect::<Vec<_>>();
        let constraint = if active.is_empty() { order } else { expr::imp(expr::and(active), order) };
        self.add_integer_constraint(Constraint::Intension(constraint));
    }

    fn post_native_cumulative(&mut self, intervals: Vec<(NativeIntervalSpec, i32)>, capacity: i32, blockers: &[(i32, i32, i32)]) {
        if intervals.is_empty() && blockers.is_empty() {
            return;
        }
        let mut starts = Vec::with_capacity(intervals.len() + blockers.len());
        let mut durations = Vec::with_capacity(intervals.len() + blockers.len());
        let mut demands = Vec::with_capacity(intervals.len() + blockers.len());
        for (interval, demand) in intervals {
            starts.push(interval.start);
            durations.push(self.const_var(interval.duration));
            demands.push(if let Some(presence) = interval.presence {
                self.aux_for(expr::mul(vec![expr::var(presence), expr::int(i64::from(demand))]))
            } else {
                self.const_var(demand)
            });
        }
        for &(start, duration, demand) in blockers {
            starts.push(self.const_var(start));
            durations.push(self.const_var(duration));
            demands.push(self.const_var(demand));
        }
        let capacity = self.const_var(capacity);
        self.add_integer_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::CumulativeVar { starts, durations, demands, capacity }));
    }

    fn set_native_makespan_objective(&mut self, intervals: &[NativeIntervalSpec]) -> PyResult<()> {
        if intervals.is_empty() {
            return Err(PyValueError::new_err("minimize_makespan needs at least one interval"));
        }
        let upper = intervals.iter().map(|spec| self.domain_max(spec.start).saturating_add(spec.duration)).max().unwrap_or(0);
        let makespan = self.semantic.model_mut().int_range(0, upper.max(0));
        self.register_native_backing_var(makespan, Some("makespan".to_string()));
        for spec in intervals {
            let end = expr::add(vec![expr::var(spec.start), expr::int(i64::from(spec.duration))]);
            let bound = expr::ge(expr::var(makespan), end);
            let constraint =
                if let Some(presence) = spec.presence { expr::imp(expr::eq(expr::var(presence), expr::int(1)), bound) } else { bound };
            self.add_integer_constraint(Constraint::Intension(constraint));
        }
        self.objective = Some(ObjectiveSpec {
            minimizing: true,
            expr: ExprLike { model_id: Some(self.id), expr: Expr::Variable(makespan), text: "makespan".to_string() },
        });
        self.then_objectives.clear();
        Ok(())
    }

    /// Reject a term that belongs to a different model or to a superseded
    /// `list_vars` generation.
    fn check_term_scope(&self, model_id: u64, gen: u64) -> PyResult<()> {
        if model_id != self.id {
            return Err(PyValueError::new_err("this list term belongs to a different model"));
        }
        if gen != self.semantic.list_generation {
            return Err(PyValueError::new_err("this list term is stale; rebuild it from the current list_vars()"));
        }
        Ok(())
    }

    /// Record a collection objective tier. `replace` clears prior tiers (a fresh
    /// primary objective); otherwise the term is appended as the next lower
    /// lexicographic tier.
    fn push_collection_tier(&mut self, term: &PyTerm, minimize: bool, replace: bool) -> PyResult<()> {
        self.check_term_scope(term.model_id, term.gen)?;
        if replace {
            self.semantic.objectives.clear();
        }
        let mut terms = term.reductions.clone();
        for &(value, coeff) in &term.values {
            let reduction = self
                .semantic
                .reified_values
                .get(value)
                .and_then(|slot| slot.as_ref())
                .ok_or_else(|| PyValueError::new_err("a list value in the objective was never bound with add(scan == value)"))?;
            terms.push(scale_reduction(reduction, coeff)?);
        }
        self.semantic.objectives.push(shared_model::Objective::ListTerms { minimize, terms, max_terms: term.max_terms.clone() });
        Ok(())
    }

    fn set_integer_objective(&mut self, objective: &Bound<'_, PyAny>, minimizing: bool) -> PyResult<()> {
        let expr = expr_from_py(objective)?;
        if let Some(model_id) = expr.model_id {
            if model_id != self.id {
                return Err(PyValueError::new_err("objective belongs to a different model"));
            }
        }
        let mut objective_vars = Vec::new();
        expr.expr.variables(&mut objective_vars);
        if objective_vars.is_empty() {
            return Err(PyValueError::new_err("objective must reference at least one model variable"));
        }
        self.objective = Some(ObjectiveSpec { minimizing, expr });
        self.then_objectives.clear();
        Ok(())
    }

    fn push_integer_tier(&mut self, objective: &Bound<'_, PyAny>, minimizing: bool) -> PyResult<()> {
        if self.objective.is_none() {
            return Err(PyValueError::new_err("then_minimize/then_maximize requires a primary integer objective first"));
        }
        let expr = expr_from_py(objective)?;
        if let Some(model_id) = expr.model_id {
            if model_id != self.id {
                return Err(PyValueError::new_err("objective belongs to a different model"));
            }
        }
        let mut objective_vars = Vec::new();
        expr.expr.variables(&mut objective_vars);
        if objective_vars.is_empty() {
            return Err(PyValueError::new_err("objective must reference at least one model variable"));
        }
        self.then_objectives.push(ObjectiveSpec { minimizing, expr });
        Ok(())
    }

    /// Begin (or restart) interval-schedule mode and invalidate prior compact
    /// schedule handles. Other independent semantic families are preserved.
    fn enter_schedule_mode(&mut self) -> PyResult<()> {
        self.semantic.schedule_generation += 1;
        Ok(())
    }

    /// Reject an interval handle from a different model or a superseded schedule
    /// generation, so a stale or cross-model interval cannot alias by index.
    fn check_interval_scope(&self, iv: &PyIntervalVar) -> PyResult<()> {
        if iv.model_id != self.id {
            return Err(PyValueError::new_err("this interval belongs to a different model"));
        }
        if iv.kind == PyIntervalKind::Native {
            self.native_interval_spec(iv)?;
            return Ok(());
        }
        if iv.gen != self.semantic.schedule_generation || iv.index as usize >= self.semantic.model().intervals().len() {
            return Err(PyValueError::new_err("this interval is stale; rebuild it from the current intervals()/alternatives()"));
        }
        Ok(())
    }

    /// Parse a list-shaped warm start. Semantic scope, coverage, and capability
    /// checks belong to the canonical compiler.
    fn parse_list_hint(&self, obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<i32>>> {
        obj.extract().map_err(|_| PyValueError::new_err("list_hint must be a list of lists of ints (one node sequence per list variable)"))
    }

    fn collection_solution_from_result(&self, run: &CollectionRun, profile: bool) -> PyResult<PySolution> {
        let result = &run.result;
        let primal = result.primal();
        let schedule_makespan = self.semantic.schedule_makespan();
        let has_list_objective = self.semantic.primary_list_sense().is_some();
        let primary_minimizing = self.semantic.primary_list_sense().unwrap_or(true);
        let primary_sense = if primary_minimizing { "min" } else { "max" };
        let local_search =
            matches!(run.engine, EngineKind::ListLocalSearch | EngineKind::RoutingLocalSearch | EngineKind::ScheduleLocalSearch);
        let objectives = primal.map_or_else(Vec::new, |candidate| candidate.objectives().to_vec());
        let objective = objectives.first().copied();
        let objective_sense =
            has_list_objective.then(|| primary_sense.to_string()).or_else(|| schedule_makespan.then(|| "min".to_string()));
        let objective_expr = match run.engine {
            EngineKind::RoutingExact => Some("integer routing edge-sum".to_string()),
            _ if has_list_objective => Some("list objective".to_string()),
            _ if schedule_makespan => Some("makespan".to_string()),
            _ => None,
        };
        let report = result_engine_report(result, run.engine);
        let orchestration_report = result.reports().first();
        let stats = result.aggregate_search_stats();
        let lists = if self.semantic.has_lists() {
            let unordered = self
                .semantic
                .model()
                .lists()
                .iter()
                .enumerate()
                .filter_map(|(index, declaration)| (declaration.ordering == shared_model::ListOrdering::Unordered).then_some(index))
                .collect();
            canonicalize_unordered_lists(primal.map(|candidate| candidate.assignment().lists.clone()), &unordered)
        } else {
            None
        };
        let intervals = primal
            .filter(|_| self.semantic.has_intervals())
            .map(|candidate| candidate.assignment().intervals.as_slice())
            .unwrap_or_default();
        let starts = intervals.iter().map(|interval| interval.start).collect();
        let presences = intervals.iter().map(|interval| interval.present).collect();
        let machines =
            intervals.iter().map(|interval| interval.machine.and_then(|machine| i64::try_from(machine).ok()).unwrap_or(-1)).collect();
        let (dual_bound, absolute_gap, relative_gap, bound_method) =
            collection_bound_fields(result, objectives.first().copied(), primary_minimizing);

        let expose_search_profile = profile && local_search;
        let alns_iterations = expose_search_profile.then(|| parse_report_metadata(report, "alns_iterations").unwrap_or_default());
        let candidates_evaluated = expose_search_profile.then(|| parse_report_metadata(report, "candidates_evaluated").unwrap_or_default());
        let candidates_per_second =
            expose_search_profile.then(|| parse_report_metadata(report, "candidates_per_second").unwrap_or_default());
        let full_recompute_percentage =
            expose_search_profile.then(|| parse_report_metadata(report, "full_recompute_percentage").unwrap_or_default());
        let construction_seconds = expose_search_profile.then(|| parse_report_metadata(report, "construction_seconds").unwrap_or_default());
        let construction_candidates =
            expose_search_profile.then(|| parse_report_metadata(report, "construction_candidates").unwrap_or_default());
        let constructor = expose_search_profile.then(|| report_metadata(report, "constructor").map(str::to_string)).flatten();
        let constructor_fleet = expose_search_profile.then(|| parse_report_metadata(report, "constructor_fleet")).flatten();
        let constructor_cost = expose_search_profile.then(|| parse_report_metadata(report, "constructor_cost")).flatten();
        let anytime_checkpoints = expose_search_profile.then(|| parse_anytime_checkpoints(report)).flatten();
        let neighborhood_profile = expose_search_profile.then(|| parse_neighborhood_profile(report)).flatten();
        let routing_counters = expose_search_profile.then(|| parse_routing_counters(report)).flatten();
        let backend_build_seconds = parse_report_metadata(orchestration_report, "backend_build_seconds");
        let estimated_backend_bytes = parse_report_metadata(orchestration_report, "estimated_backend_bytes");
        let records_first_feasible = matches!(
            run.engine,
            EngineKind::ScheduleExact | EngineKind::ListLocalSearch | EngineKind::RoutingLocalSearch | EngineKind::ScheduleLocalSearch
        );
        let time_to_first_feasible = (profile && records_first_feasible).then_some(run.events.first_feasible_at).flatten().or_else(|| {
            expose_search_profile
                .then(|| parse_report_metadata::<f64>(report, "time_to_first_feasible"))
                .flatten()
                .map(|time| backend_build_seconds.unwrap_or_default() + time)
        });

        Ok(PySolution {
            status: result.status().as_str().to_string(),
            objective,
            objective_sense,
            objective_expr,
            values: visible_integer_values(result, self.names.len())?,
            stats: stats.into(),
            lists,
            objectives,
            starts,
            presences,
            machines,
            dual_bound,
            absolute_gap,
            relative_gap,
            bound_method,
            alns_iterations,
            candidates_evaluated,
            candidates_per_second,
            full_recompute_percentage,
            backend_build_seconds: profile.then_some(backend_build_seconds).flatten(),
            construction_seconds,
            time_to_first_feasible,
            construction_candidates,
            estimated_backend_bytes: profile.then_some(estimated_backend_bytes).flatten(),
            constructor,
            constructor_fleet,
            constructor_cost,
            anytime_checkpoints,
            neighborhood_profile,
            routing_counters,
            ls_moves: None,
            ls_constraints: None,
            ls_functionals: None,
            ls_unsupported: None,
            ls_rejected_incumbents: None,
            ls_checkpoint_replays: None,
        })
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
fn max_of(items: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = items.py();
    let raw = items.try_iter()?.map(|item| item.map(|item| item.unbind())).collect::<PyResult<Vec<_>>>()?;
    if raw.is_empty() {
        return Err(PyValueError::new_err("max_of requires at least one expression"));
    }

    let mut terms = Vec::with_capacity(raw.len());
    let mut saw_non_term = false;
    for item in &raw {
        match item.bind(py).extract::<PyRef<'_, PyTerm>>() {
            Ok(term) => terms.push(term.clone()),
            Err(_) => {
                saw_non_term = true;
                break;
            }
        }
    }
    if !terms.is_empty() {
        if saw_non_term || terms.len() != raw.len() {
            return Err(PyTypeError::new_err("max_of(iterable) cannot mix list-domain terms with arithmetic expressions"));
        }
        let term = max_term(terms)?;
        return Ok(term.into_pyobject(py)?.into_any().unbind());
    }

    let mut model_id = None;
    let mut exprs = Vec::with_capacity(raw.len());
    let mut texts = Vec::with_capacity(raw.len());
    for item in raw {
        let item = item.bind(py);
        let item = expr_from_py(item)?;
        model_id = merge_model_ids(model_id, item.model_id)?;
        exprs.push(item.expr);
        texts.push(item.text);
    }
    let expr = PyExpr { inner: ExprLike { model_id, expr: Expr::Max(exprs), text: format!("max({})", texts.join(", ")) } };
    Ok(expr.into_pyobject(py)?.into_any().unbind())
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
/// to a [`list::ExprArena`] when the term joins the model.
enum PyNode {
    Const(i64),
    Arg(u8),
    Array(Arc<Vec<i64>>, Arc<PyNode>),
    Matrix(Arc<Vec<Vec<i64>>>, Arc<PyNode>, Arc<PyNode>),
    Add(Arc<PyNode>, Arc<PyNode>),
    Sub(Arc<PyNode>, Arc<PyNode>),
    Mul(Arc<PyNode>, Arc<PyNode>),
    Mod(Arc<PyNode>, Arc<PyNode>),
    Pow(Arc<PyNode>, u32),
    MulScaled(Arc<PyNode>, Arc<PyNode>, i64),
    DivScaled(Arc<PyNode>, Arc<PyNode>, i64),
    Min(Arc<PyNode>, Arc<PyNode>),
    Max(Arc<PyNode>, Arc<PyNode>),
    Div(Arc<PyNode>, Arc<PyNode>),
    Abs(Arc<PyNode>),
    Lt(Arc<PyNode>, Arc<PyNode>),
    Le(Arc<PyNode>, Arc<PyNode>),
    Eq(Arc<PyNode>, Arc<PyNode>),
    Ne(Arc<PyNode>, Arc<PyNode>),
    IfThenElse(Arc<PyNode>, Arc<PyNode>, Arc<PyNode>),
    PiecewiseLinear(Arc<PyNode>, Arc<Vec<(i64, i64)>>),
    External(Arc<str>, Arc<Vec<Arc<PyNode>>>),
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
    fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Mod(self.node.clone(), coerce_node(other)?)))
    }
    fn __rmod__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Mod(coerce_node(other)?, self.node.clone())))
    }
    fn __pow__(&self, exponent: u32, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<PyLambdaExpr> {
        if modulo.is_some() {
            return Err(PyValueError::new_err("modular power is not supported; use (x ** n) % m"));
        }
        Ok(node(PyNode::Pow(self.node.clone(), exponent)))
    }
    fn __neg__(&self) -> PyLambdaExpr {
        node(PyNode::Sub(Arc::new(PyNode::Const(0)), self.node.clone()))
    }
    fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Div(self.node.clone(), coerce_node(other)?)))
    }
    fn __rfloordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Div(coerce_node(other)?, self.node.clone())))
    }
    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Lt(self.node.clone(), coerce_node(other)?)))
    }
    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Le(self.node.clone(), coerce_node(other)?)))
    }
    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        // a > b  ==  b < a
        Ok(node(PyNode::Lt(coerce_node(other)?, self.node.clone())))
    }
    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
        Ok(node(PyNode::Le(coerce_node(other)?, self.node.clone())))
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
    reductions: Vec<list::Reduction>,
    max_terms: Option<Vec<list::MaxTerm>>,
    /// `(value_id, coeff)` for each reified `list_value()` added into this term;
    /// resolved to its bound scan reduction when the objective tier is pushed.
    values: Vec<(usize, i64)>,
}

/// A constraint `term <op> rhs` over a single list reduction.
#[pyclass(name = "ListConstraint", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyListConstraint {
    model_id: u64,
    gen: u64,
    reduction: list::Reduction,
    op: list::Op,
    rhs: i64,
}

/// Coerce a collection objective argument: a `Term`, or a bare `list_value()`
/// handle (`minimize(v)`). Returns `None` for an integer expression objective.
fn collection_objective_term(objective: &Bound<'_, PyAny>) -> Option<PyTerm> {
    if let Ok(term) = objective.extract::<PyRef<'_, PyTerm>>() {
        return Some(term.clone());
    }
    objective.extract::<PyRef<'_, ListValue>>().ok().map(|value| value.as_term())
}

/// Sum two terms, rejecting a mix of models or `list_vars` generations.
fn combine_terms(a: &PyTerm, b: &PyTerm) -> PyResult<PyTerm> {
    if a.model_id != b.model_id || a.gen != b.gen {
        return Err(PyValueError::new_err("cannot combine terms from different models or list_vars generations"));
    }
    let mut reductions = a.reductions.clone();
    reductions.extend(b.reductions.iter().cloned());
    let mut max_terms = a.max_terms.clone().unwrap_or_default();
    max_terms.extend(b.max_terms.iter().flatten().cloned());
    let max_terms = (!max_terms.is_empty()).then_some(max_terms);
    let mut values = a.values.clone();
    values.extend(b.values.iter().copied());
    Ok(PyTerm { model_id: a.model_id, gen: a.gen, reductions, max_terms, values })
}

fn scale_reduction(reduction: &list::Reduction, coeff: i64) -> PyResult<list::Reduction> {
    let mut reduction = reduction.clone();
    reduction.coeff = reduction.coeff.checked_mul(coeff).ok_or_else(|| PyValueError::new_err("term coefficient overflows i64"))?;
    Ok(reduction)
}

fn scale_term(term: &PyTerm, coeff: i64) -> PyResult<PyTerm> {
    let reductions = term.reductions.iter().map(|reduction| scale_reduction(reduction, coeff)).collect::<PyResult<Vec<_>>>()?;
    let max_terms = term
        .max_terms
        .as_ref()
        .map(|terms| {
            terms
                .iter()
                .map(|term| {
                    Ok(list::MaxTerm {
                        groups: term.groups.clone(),
                        coeff: term.coeff.checked_mul(coeff).ok_or_else(|| PyValueError::new_err("term coefficient overflows i64"))?,
                    })
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .transpose()?;
    let values = term
        .values
        .iter()
        .map(|&(value, c)| Ok((value, c.checked_mul(coeff).ok_or_else(|| PyValueError::new_err("term coefficient overflows i64"))?)))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTerm { model_id: term.model_id, gen: term.gen, reductions, max_terms, values })
}

fn max_term(terms: Vec<PyTerm>) -> PyResult<PyTerm> {
    let first = terms.first().ok_or_else(|| PyValueError::new_err("max_of requires at least one term"))?;
    let model_id = first.model_id;
    let gen = first.gen;
    let mut groups = Vec::with_capacity(terms.len());
    for term in terms {
        if term.model_id != model_id || term.gen != gen {
            return Err(PyValueError::new_err("cannot combine terms from different models or list_vars generations"));
        }
        if term.max_terms.is_some() {
            return Err(PyTypeError::new_err("nested max_of terms are not supported"));
        }
        if !term.values.is_empty() {
            return Err(PyTypeError::new_err("reified list values are not supported inside max_of terms"));
        }
        groups.push(term.reductions);
    }
    Ok(PyTerm { model_id, gen, reductions: Vec::new(), max_terms: Some(vec![list::MaxTerm { groups, coeff: 1 }]), values: Vec::new() })
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

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
        scale_term(self, other.extract::<i64>().map_err(|_| PyTypeError::new_err("a term can only be multiplied by an integer"))?)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
        self.__mul__(other)
    }

    fn __neg__(&self) -> PyResult<PyTerm> {
        scale_term(self, -1)
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        if self.max_terms.as_ref().is_some_and(|terms| !terms.is_empty()) {
            return Err(PyValueError::new_err("max_of list terms are supported as objectives, not constraints"));
        }
        let op = match op {
            CompareOp::Le => list::Op::Le,
            CompareOp::Ge => list::Op::Ge,
            CompareOp::Eq => list::Op::Eq,
            _ => return Err(PyValueError::new_err("a term supports only <=, >=, ==")),
        };
        if self.reductions.len() != 1 || !self.values.is_empty() {
            return Err(PyValueError::new_err("a constraint must be a single reduction over one list, not a sum of terms"));
        }
        let py = other.py();
        // `scan_sum(r, ...) == v` reifies the scan total to a `list_value()` handle,
        // so the value can then enter a linear objective.
        if let Ok(value) = other.extract::<PyRef<'_, ListValue>>() {
            if value.model_id != self.model_id || value.gen != self.gen {
                return Err(PyValueError::new_err("this list value belongs to a different model or list_vars generation"));
            }
            // Reification binds the value AS a definition (`value := reduction`),
            // which is exactly `==`. An inequality (`<=`/`>=`) would make the value
            // a bounded optimisation variable the enumeration does not yet range
            // over, so it is rejected rather than silently treated as `==`.
            if !matches!(op, list::Op::Eq) {
                return Err(PyValueError::new_err(
                    "reify a scan into a list value with `scan == value` (inequality reification is not supported)",
                ));
            }
            let reified = PyReified { model_id: self.model_id, gen: self.gen, value: value.value, reduction: self.reductions[0].clone() };
            return Ok(reified.into_pyobject(py)?.into_any().unbind());
        }
        let rhs =
            other.extract::<i64>().map_err(|_| PyTypeError::new_err("a term can only be compared to an integer bound or a list value"))?;
        let constraint = PyListConstraint { model_id: self.model_id, gen: self.gen, reduction: self.reductions[0].clone(), op, rhs };
        Ok(constraint.into_pyobject(py)?.into_any().unbind())
    }
}

/// A collection-scoped scalar minted by `Model.list_value()`. A scan total is
/// bound to it with `add(scan == v)`; it then composes into a linear objective
/// (`sum_edges + v`). Collection models forbid real integer variables, so this is
/// not an `IntVar` -- it is resolved to its bound reduction at solve time.
#[pyclass(name = "ListValue", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct ListValue {
    model_id: u64,
    gen: u64,
    value: usize,
}

impl ListValue {
    fn as_term(&self) -> PyTerm {
        PyTerm { model_id: self.model_id, gen: self.gen, reductions: Vec::new(), max_terms: None, values: vec![(self.value, 1)] }
    }
}

#[pymethods]
impl ListValue {
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
        let this = self.as_term();
        if let Ok(term) = other.extract::<PyRef<'_, PyTerm>>() {
            return combine_terms(&this, &term);
        }
        if let Ok(value) = other.extract::<PyRef<'_, ListValue>>() {
            return combine_terms(&this, &value.as_term());
        }
        if other.extract::<i64>().is_ok_and(|v| v == 0) {
            return Ok(this);
        }
        Err(PyTypeError::new_err("a list value can only be added to a term or another list value"))
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
        self.__add__(other)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
        let coeff = other.extract::<i64>().map_err(|_| PyTypeError::new_err("a list value can only be multiplied by an integer"))?;
        Ok(PyTerm { model_id: self.model_id, gen: self.gen, reductions: Vec::new(), max_terms: None, values: vec![(self.value, coeff)] })
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
        self.__mul__(other)
    }

    fn __neg__(&self) -> PyTerm {
        PyTerm { model_id: self.model_id, gen: self.gen, reductions: Vec::new(), max_terms: None, values: vec![(self.value, -1)] }
    }
}

/// A reified binding `reduction <op> value`, produced by comparing a scan total
/// to a `list_value()` handle and recorded by `Model.add`.
#[pyclass(name = "Reified", module = "qayd", skip_from_py_object)]
#[derive(Clone)]
struct PyReified {
    model_id: u64,
    gen: u64,
    value: usize,
    reduction: list::Reduction,
}

/// Lower a Python lambda-body tree into a reduction's flat expression arena.
fn lower(n: &PyNode, arena: &mut list::ExprArena) -> list::ExprId {
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
        PyNode::Mod(a, b) => {
            let x = lower(a, arena);
            let y = lower(b, arena);
            arena.modulo(x, y)
        }
        PyNode::Pow(base, exponent) => {
            let base = lower(base, arena);
            arena.pow(base, *exponent)
        }
        PyNode::MulScaled(a, b, scale) => {
            let x = lower(a, arena);
            let y = lower(b, arena);
            arena.mul_scaled(x, y, *scale)
        }
        PyNode::DivScaled(a, b, scale) => {
            let x = lower(a, arena);
            let y = lower(b, arena);
            arena.div_scaled(x, y, *scale)
        }
        PyNode::Min(a, b) => {
            let x = lower(a, arena);
            let y = lower(b, arena);
            arena.min(x, y)
        }
        PyNode::Max(a, b) => {
            let x = lower(a, arena);
            let y = lower(b, arena);
            arena.max(x, y)
        }
        PyNode::Div(a, b) => {
            let x = lower(a, arena);
            let y = lower(b, arena);
            arena.div(x, y)
        }
        PyNode::Abs(a) => {
            let x = lower(a, arena);
            arena.abs(x)
        }
        PyNode::Lt(a, b) => {
            let x = lower(a, arena);
            let y = lower(b, arena);
            arena.lt(x, y)
        }
        PyNode::Le(a, b) => {
            let x = lower(a, arena);
            let y = lower(b, arena);
            arena.le(x, y)
        }
        PyNode::Eq(a, b) => {
            let x = lower(a, arena);
            let y = lower(b, arena);
            arena.eq(x, y)
        }
        PyNode::Ne(a, b) => {
            let x = lower(a, arena);
            let y = lower(b, arena);
            arena.ne(x, y)
        }
        PyNode::IfThenElse(c, a, b) => {
            let cc = lower(c, arena);
            let x = lower(a, arena);
            let y = lower(b, arena);
            arena.if_then_else(cc, x, y)
        }
        PyNode::PiecewiseLinear(input, points) => {
            let input = lower(input, arena);
            arena.piecewise_linear(input, points.clone())
        }
        PyNode::External(name, args) => {
            let args = args.iter().map(|arg| lower(arg, arena)).collect();
            arena.external(name.clone(), args)
        }
    }
}

fn single_term(route: &PyListVar, reduction: list::Reduction) -> PyTerm {
    PyTerm { model_id: route.model_id, gen: route.gen, reductions: vec![reduction], max_terms: None, values: Vec::new() }
}

fn item_iterable(route: &PyListVar) -> list::Iterable {
    if route.unordered {
        list::Iterable::SetItems(route.index as usize)
    } else {
        list::Iterable::Items(route.index as usize)
    }
}

fn require_ordered(route: &PyListVar, operation: &str) -> PyResult<()> {
    if route.unordered {
        Err(PyValueError::new_err(format!("{operation} requires an ordered ListVar, not a SetVar")))
    } else {
        Ok(())
    }
}

/// Build a per-item reduction `op(route, i => body)` from a Python lambda.
fn build_items_reduction(route: &PyListVar, op: list::ReduceOp, func: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
    let body = coerce_node(&func.call1((node(PyNode::Arg(0)),))?)?;
    let mut arena = list::ExprArena::default();
    let body_id = lower(&body, &mut arena);
    Ok(single_term(route, list::Reduction { op, iterable: item_iterable(route), arena, body: body_id, coeff: 1 }))
}

/// `sum(route, i => body)`, or `sum(terms)` to add a collection of terms.
#[pyfunction]
#[pyo3(signature = (arg, func=None))]
fn sum(py: Python<'_>, arg: &Bound<'_, PyAny>, func: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    if let Some(f) = func {
        let route = arg
            .extract::<PyRef<'_, PyListVar>>()
            .map_err(|_| PyTypeError::new_err("sum(route, lambda): the first argument must be a list variable"))?;
        return Ok(build_items_reduction(&route, list::ReduceOp::Sum, f)?.into_pyobject(py)?.into_any().unbind());
    }
    // An iterable of list-domain terms folds to a Term; an iterable of
    // arithmetic operands (Expr/IntVar/int) folds to an Expr. Mixing the two
    // domains has no meaning.
    let mut terms: Option<PyTerm> = None;
    let mut exprs: Option<(Vec<Expr>, Option<u64>, Vec<String>)> = None;
    for item in arg.try_iter()? {
        let item = item?;
        if let Ok(t) = item.extract::<PyRef<'_, PyTerm>>() {
            if exprs.is_some() {
                return Err(PyTypeError::new_err("sum(iterable) cannot mix list-domain terms with arithmetic expressions"));
            }
            terms = Some(match terms {
                None => t.clone(),
                Some(a) => combine_terms(&a, &t)?,
            });
            continue;
        }
        let e = expr_from_py(&item)
            .map_err(|_| PyTypeError::new_err("sum(iterable) expects list-domain terms or arithmetic operands (Expr, IntVar, int)"))?;
        if terms.is_some() {
            return Err(PyTypeError::new_err("sum(iterable) cannot mix list-domain terms with arithmetic expressions"));
        }
        let (parts, model_id, texts) = exprs.get_or_insert_with(|| (Vec::new(), None, Vec::new()));
        *model_id = merge_model_ids(*model_id, e.model_id)?;
        parts.push(e.expr);
        texts.push(e.text);
    }
    match (terms, exprs) {
        (Some(t), None) => Ok(t.into_pyobject(py)?.into_any().unbind()),
        (None, Some((parts, model_id, texts))) => {
            let expr = PyExpr { inner: ExprLike { model_id, expr: expr::add(parts), text: format!("({})", texts.join(" + ")) } };
            Ok(expr.into_pyobject(py)?.into_any().unbind())
        }
        _ => Err(PyValueError::new_err("sum got no terms to add")),
    }
}

/// `min(route, i => body)` over a route's items (undefined, hence infeasible,
/// for an empty route).
#[pyfunction]
fn minimum(route: &PyListVar, func: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
    build_items_reduction(route, list::ReduceOp::Min, func)
}

/// `max(route, i => body)` over a route's items.
#[pyfunction]
fn maximum(route: &PyListVar, func: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
    build_items_reduction(route, list::ReduceOp::Max, func)
}

/// `count(route, i => predicate)`: items whose body is non-zero. With no lambda,
/// the route's length.
#[pyfunction(name = "count")]
#[pyo3(signature = (route, func=None))]
fn count_reduction(route: &PyListVar, func: Option<&Bound<'_, PyAny>>) -> PyResult<PyTerm> {
    match func {
        Some(f) => build_items_reduction(route, list::ReduceOp::Count, f),
        None => {
            let mut arena = list::ExprArena::default();
            let body = arena.constant(1);
            Ok(single_term(route, list::Reduction { op: list::ReduceOp::Count, iterable: item_iterable(route), arena, body, coeff: 1 }))
        }
    }
}

/// `sum_edges(route, (i, j) => body, start=, end=)`: sum the body over the edges
/// of the closed tour `[start, items.., end]`.
#[pyfunction]
#[pyo3(signature = (route, func, *, start=0, end=0))]
fn sum_edges(route: &PyListVar, func: &Bound<'_, PyAny>, start: i32, end: i32) -> PyResult<PyTerm> {
    require_ordered(route, "sum_edges")?;
    let body = coerce_node(&func.call1((node(PyNode::Arg(0)), node(PyNode::Arg(1))))?)?;
    let mut arena = list::ExprArena::default();
    let body_id = lower(&body, &mut arena);
    let iterable = list::Iterable::Edges { list: route.index as usize, start, end };
    Ok(single_term(route, list::Reduction { op: list::ReduceOp::Sum, iterable, arena, body: body_id, coeff: 1 }))
}

fn pairs_term(route: &PyListVar, body: Arc<PyNode>) -> PyTerm {
    let mut arena = list::ExprArena::default();
    let body_id = lower(&body, &mut arena);
    let iterable = list::Iterable::Pairs(route.index as usize);
    single_term(route, list::Reduction { op: list::ReduceOp::Sum, iterable, arena, body: body_id, coeff: 1 })
}

/// `item_pairs(route, (a, b) => body)`: sum the body over every ordered pair of
/// items `(a, b)` in the route. Use for pairwise conflict counts (bin packing
/// with conflicts), e.g. `item_pairs(bin, lambda a, b: conflict[a][b]) == 0`.
#[pyfunction]
fn item_pairs(route: &PyListVar, func: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
    require_ordered(route, "item_pairs")?;
    let body = coerce_node(&func.call1((node(PyNode::Arg(0)), node(PyNode::Arg(1))))?)?;
    Ok(pairs_term(route, body))
}

/// `pos_pairs(route, (a, b, i, j) => body)`: sum the body over every ordered
/// pair of positions, with the items `a`/`b` at positions `i`/`j`. Use for
/// quadratic objectives (QAP), e.g.
/// `pos_pairs(p, lambda a, b, i, j: A[i][j] * B[a][b])`.
#[pyfunction]
fn pos_pairs(route: &PyListVar, func: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
    require_ordered(route, "pos_pairs")?;
    let args = (node(PyNode::Arg(0)), node(PyNode::Arg(1)), node(PyNode::Arg(2)), node(PyNode::Arg(3)));
    let body = coerce_node(&func.call1(args)?)?;
    Ok(pairs_term(route, body))
}

/// `min(a, b)` / `max(a, b)` inside a lambda body (each operand an expression or
/// integer).
#[pyfunction(name = "min")]
fn min_expr(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
    Ok(node(PyNode::Min(coerce_node(a)?, coerce_node(b)?)))
}

#[pyfunction(name = "max")]
fn max_expr(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
    Ok(node(PyNode::Max(coerce_node(a)?, coerce_node(b)?)))
}

/// `abs(x)` inside a lambda body.
#[pyfunction(name = "abs")]
fn abs_expr(a: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
    Ok(node(PyNode::Abs(coerce_node(a)?)))
}

/// `cond != 0 ? a : b` inside a lambda body.
#[pyfunction(name = "if_")]
fn if_expr(cond: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
    Ok(node(PyNode::IfThenElse(coerce_node(cond)?, coerce_node(a)?, coerce_node(b)?)))
}

/// `a == b` / `a != b` (1 or 0) inside a lambda body.
#[pyfunction(name = "eq")]
fn eq_expr(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
    Ok(node(PyNode::Eq(coerce_node(a)?, coerce_node(b)?)))
}

#[pyfunction(name = "ne")]
fn ne_expr(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyLambdaExpr> {
    Ok(node(PyNode::Ne(coerce_node(a)?, coerce_node(b)?)))
}

/// Convert a finite Python float into a deterministic fixed-point raw value.
#[pyfunction]
#[pyo3(signature = (value, *, scale=1_000_000))]
fn fixed(value: f64, scale: i64) -> PyResult<i64> {
    list::FixedPoint::from_f64(value, scale).map(|value| value.raw).map_err(PyValueError::new_err)
}

/// Fixed-point multiplication with nearest-integer rounding.
#[pyfunction]
fn mul_scaled(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, scale: i64) -> PyResult<PyLambdaExpr> {
    if scale <= 0 {
        return Err(PyValueError::new_err("fixed-point scale must be positive"));
    }
    Ok(node(PyNode::MulScaled(coerce_node(a)?, coerce_node(b)?, scale)))
}

/// Fixed-point division with nearest-integer rounding.
#[pyfunction]
fn div_scaled(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, scale: i64) -> PyResult<PyLambdaExpr> {
    if scale <= 0 {
        return Err(PyValueError::new_err("fixed-point scale must be positive"));
    }
    Ok(node(PyNode::DivScaled(coerce_node(a)?, coerce_node(b)?, scale)))
}

/// Continuous piecewise-linear interpolation over fixed-point knots.
#[pyfunction]
fn piecewise(input: &Bound<'_, PyAny>, points: Vec<(i64, i64)>) -> PyResult<PyLambdaExpr> {
    if points.is_empty() || points.windows(2).any(|window| window[0].0 >= window[1].0) {
        return Err(PyValueError::new_err("piecewise points need strictly increasing x coordinates"));
    }
    Ok(node(PyNode::PiecewiseLinear(coerce_node(input)?, Arc::new(points))))
}

/// Register a deterministic Python callback callable from list expressions.
#[pyfunction]
fn register_external(name: String, function: Py<PyAny>) -> PyResult<()> {
    let callback = function;
    list::register_external_function(name, move |args| {
        Python::attach(|py| {
            let tuple = PyTuple::new(py, args).map_err(|error| error.to_string())?;
            callback.bind(py).call1(tuple).and_then(|value| value.extract::<i64>()).map_err(|error| error.to_string())
        })
    })
    .map_err(PyValueError::new_err)
}

/// Build a call to a previously registered external function.
#[pyfunction]
#[pyo3(signature = (name, *args))]
fn external(name: String, args: &Bound<'_, PyTuple>) -> PyResult<PyLambdaExpr> {
    if !list::external_function_registered(&name) {
        return Err(PyValueError::new_err(format!("external function '{name}' is not registered")));
    }
    let args = args.iter().map(|arg| coerce_node(&arg)).collect::<PyResult<Vec<_>>>()?;
    Ok(node(PyNode::External(name.into(), Arc::new(args))))
}

/// `scan_sum(route, step, emit, init=, boundary=)`: fold an accumulator along
/// the route and sum a per-step value. `step(cur, acc, prev) -> new_acc` and
/// `emit(cur, acc, prev) -> value` (where `acc` in `emit` is the new
/// accumulator, `prev` is the previous item or `boundary` at the first step).
/// Used for cumulative time/load, e.g. time-window lateness.
///
/// With `end` set, the scan performs ONE more transition for the closing edge
/// after the last item -- `step(end, acc_last, last_item)` then its `emit` -- so a
/// threaded resource is folded over the CLOSED tour, including the return arc to
/// `end`. Mirrors how `sum_edges(..., end=)` closes a tour. `end=None` (default)
/// stops at the last item (unchanged behaviour).
#[pyfunction]
#[pyo3(signature = (route, step, emit, *, init=0, boundary=0, end=None))]
fn scan_sum(
    route: &PyListVar,
    step: &Bound<'_, PyAny>,
    emit: &Bound<'_, PyAny>,
    init: i64,
    boundary: i32,
    end: Option<i32>,
) -> PyResult<PyTerm> {
    require_ordered(route, "scan_sum")?;
    let step_body = coerce_node(&step.call1((node(PyNode::Arg(0)), node(PyNode::Arg(1)), node(PyNode::Arg(2))))?)?;
    let emit_body = coerce_node(&emit.call1((node(PyNode::Arg(0)), node(PyNode::Arg(1)), node(PyNode::Arg(2))))?)?;
    let mut arena = list::ExprArena::default();
    let step_id = lower(&step_body, &mut arena);
    let emit_id = lower(&emit_body, &mut arena);
    let iterable = list::Iterable::Scan { list: route.index as usize, init, boundary, step: step_id, end };
    Ok(single_term(route, list::Reduction { op: list::ReduceOp::Sum, iterable, arena, body: emit_id, coeff: 1 }))
}

/// `select_kth(route, k, step, emit, init=, boundary=)`: the `k`-th smallest
/// (0-indexed) of the per-step `emit` values along the route, i.e. an order
/// statistic / quantile of a scanned series. `step(cur, acc, prev) -> new_acc`
/// threads the accumulator (use `acc + 1` for a position counter); `emit(cur,
/// acc, prev) -> value` is the value sorted over. Used for risk quantiles such
/// as value-at-risk; undefined (infeasible) if the route has fewer than `k + 1`
/// items. Maximise it for a risk-averse objective.
#[pyfunction]
#[pyo3(signature = (route, k, step, emit, *, init=0, boundary=0))]
fn select_kth(route: &PyListVar, k: usize, step: &Bound<'_, PyAny>, emit: &Bound<'_, PyAny>, init: i64, boundary: i32) -> PyResult<PyTerm> {
    require_ordered(route, "select_kth")?;
    let step_body = coerce_node(&step.call1((node(PyNode::Arg(0)), node(PyNode::Arg(1)), node(PyNode::Arg(2))))?)?;
    let emit_body = coerce_node(&emit.call1((node(PyNode::Arg(0)), node(PyNode::Arg(1)), node(PyNode::Arg(2))))?)?;
    let mut arena = list::ExprArena::default();
    let step_id = lower(&step_body, &mut arena);
    let emit_id = lower(&emit_body, &mut arena);
    let iterable = list::Iterable::Scan { list: route.index as usize, init, boundary, step: step_id, end: None };
    Ok(single_term(route, list::Reduction { op: list::ReduceOp::SelectKth(k), iterable, arena, body: emit_id, coeff: 1 }))
}

/// `windows(route, size, inner, emit)`: for each window of `size` consecutive
/// items, sum `inner(item)` to a window total, then sum `emit(total)` over
/// windows. Used for sliding-window counts, e.g. car-sequencing option limits:
/// `windows(seq, q, inner=lambda c: opt[c], emit=lambda t: cp.max(0, t - p))`.
#[pyfunction]
fn windows(route: &PyListVar, size: usize, inner: &Bound<'_, PyAny>, emit: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
    require_ordered(route, "windows")?;
    let inner_body = coerce_node(&inner.call1((node(PyNode::Arg(0)),))?)?;
    let emit_body = coerce_node(&emit.call1((node(PyNode::Arg(1)),))?)?;
    let mut arena = list::ExprArena::default();
    let inner_id = lower(&inner_body, &mut arena);
    let emit_id = lower(&emit_body, &mut arena);
    let iterable = list::Iterable::Windows { list: route.index as usize, size, inner: inner_id };
    Ok(single_term(route, list::Reduction { op: list::ReduceOp::Sum, iterable, arena, body: emit_id, coeff: 1 }))
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

/// `1` if `route` has any item, else `0`. Summed over routes (e.g.
/// `sum(cp.used(r) for r in routes)`) this is the number of used routes/bins,
/// for a "minimise the fleet" or "minimise open bins" objective tier.
#[pyfunction]
fn used(route: &PyListVar) -> PyTerm {
    let mut arena = list::ExprArena::default();
    let body = arena.constant(0);
    single_term(route, list::Reduction { op: list::ReduceOp::Used, iterable: item_iterable(route), arena, body, coeff: 1 })
}

/// Deliberate panic, for tests only: proves the extension is built with
/// unwinding panics (profile `pyext`), so a Rust panic surfaces as a Python
/// exception instead of aborting the interpreter.
#[pyfunction]
#[doc(hidden)]
fn _rust_panic() {
    panic!("qayd _rust_panic(): intentional test panic");
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyIntVar>()?;
    m.add_class::<PyListVar>()?;
    m.add_class::<PyIntervalVar>()?;
    m.add_class::<PySequenceVar>()?;
    m.add_class::<PyTerm>()?;
    m.add_class::<PyListConstraint>()?;
    m.add_class::<ListValue>()?;
    m.add_class::<PyReified>()?;
    m.add_class::<PyExpr>()?;
    m.add_class::<PyLambdaExpr>()?;
    m.add_class::<PyArray>()?;
    m.add_class::<PyMatrix>()?;
    m.add_class::<PyMatrixRow>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<PySoftGroup>()?;
    m.add_class::<PySolution>()?;
    m.add_class::<PySolveStats>()?;
    m.add_class::<PyMusEnumeration>()?;
    m.add_class::<PySolveSession>()?;
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
    m.add_function(wrap_pyfunction!(item_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(pos_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(min_expr, m)?)?;
    m.add_function(wrap_pyfunction!(max_expr, m)?)?;
    m.add_function(wrap_pyfunction!(abs_expr, m)?)?;
    m.add_function(wrap_pyfunction!(if_expr, m)?)?;
    m.add_function(wrap_pyfunction!(eq_expr, m)?)?;
    m.add_function(wrap_pyfunction!(ne_expr, m)?)?;
    m.add_function(wrap_pyfunction!(fixed, m)?)?;
    m.add_function(wrap_pyfunction!(mul_scaled, m)?)?;
    m.add_function(wrap_pyfunction!(div_scaled, m)?)?;
    m.add_function(wrap_pyfunction!(piecewise, m)?)?;
    m.add_function(wrap_pyfunction!(register_external, m)?)?;
    m.add_function(wrap_pyfunction!(external, m)?)?;
    m.add_function(wrap_pyfunction!(scan_sum, m)?)?;
    m.add_function(wrap_pyfunction!(select_kth, m)?)?;
    m.add_function(wrap_pyfunction!(windows, m)?)?;
    m.add_function(wrap_pyfunction!(used, m)?)?;
    m.add_function(wrap_pyfunction!(_rust_panic, m)?)?;
    m.add("STAR", i32::MIN)?;
    Ok(())
}
