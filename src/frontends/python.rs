use std::collections::HashMap;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pyo3::class::basic::CompareOp;
use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule};

use crate::constraints::count;
use crate::constraints::graph;
use crate::constraints::intension;
use crate::constraints::interval as interval_constraints;
use crate::constraints::lex;
use crate::constraints::linear::{self, Relation};
use crate::constraints::primitives;
use crate::constraints::scheduling;
use crate::constraints::table;
use crate::engines::ls::cop::{solve_ls, LocalRhs, LocalSearchSpec, LsConfig};
use crate::engines::ls::lists;
use crate::engines::{list_exact as list_exact_engine, routing as routing_engine, schedule as schedule_engine};
use crate::expr::{self, Expr};
use crate::ids::{IntervalId, VarId};
use crate::lcg::clause::{ClauseSharing, SharedClausePool};
use crate::lcg::lit::{AtomKind, AtomTable, Lit};
use crate::model as shared_model;
use crate::model::list;
use crate::problem::{Objective as ProblemObjective, Problem};
use crate::search::{self, Assumption, AssumptionOp, Objective as SearchObjective, SearchControl, SolveStats};
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
    interval: Option<IntervalId>,
    start: Option<VarId>,
    presence: Option<VarId>,
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
    presences: Vec<VarId>,
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
    interval: IntervalId,
    start: VarId,
    presence: Option<VarId>,
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
        self.start.map(|var| PyIntVar { model_id: self.model_id, index: var.0, name: None })
    }

    #[getter]
    fn presence(&self) -> Option<PyIntVar> {
        self.presence.map(|var| PyIntVar { model_id: self.model_id, index: var.0, name: None })
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
    then_objectives: Vec<ObjectiveSpec>,
    native_intervals: Vec<NativeIntervalSpec>,
    /// Local-search model built in parallel with the CP posts, so `--ls`-style
    /// LS (`solve(engine="ls")`) can run on the same model.
    local: LocalSearchSpec,
    /// Set once `list_vars` is called: the universe of items partitioned among
    /// the list variables. Its presence makes `solve()` dispatch to the
    /// list-domain engine instead of the integer CP/LS path.
    col_universe: Option<Vec<i32>>,
    col_lists: usize,
    /// Bumped on each `list_vars` call so route handles from an earlier call (or
    /// another model) are rejected instead of silently aliasing the wrong list.
    col_gen: u64,
    col_objectives: Vec<list::ObjectiveTier>,
    col_constraints: Vec<list::Constraint>,
    col_globals: Vec<list::GlobalConstraint>,
    col_schedule: Option<list::Schedule>,
    col_sched_gen: u64,
}

#[pyclass(name = "SolveSession", module = "qayd", unsendable)]
struct PySolveSession {
    id: u64,
    solver: Solver,
    names: Vec<Option<String>>,
    objectives: Vec<MaterializedObjective>,
    search: Vec<VarId>,
    native_intervals: Vec<NativeIntervalSpec>,
    clauses: Arc<SharedClausePool>,
    next_worker: usize,
}

#[derive(Clone)]
struct MaterializedObjective {
    minimizing: bool,
    var: VarId,
    text: String,
    support: Vec<VarId>,
}

type PyRawNogood = (u32, Vec<u32>);
type PyNogoodLit = (u32, String, i32);
type PyNogood = (u32, Vec<PyNogoodLit>);

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

fn interval_from_py(obj: &Bound<'_, PyAny>) -> PyResult<PyIntervalVar> {
    obj.extract::<PyRef<'_, PyIntervalVar>>().map(|iv| iv.clone()).map_err(|_| PyTypeError::new_err("expected an IntervalVar"))
}

fn interval_list_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Vec<PyIntervalVar>> {
    let iter = PyIterator::from_object(obj)?;
    iter.map(|item| interval_from_py(&item?)).collect()
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

fn search_ids_for(model_id: u64, num_vars: usize, search: Option<&Bound<'_, PyAny>>, extra: Option<VarId>) -> PyResult<Vec<VarId>> {
    let mut vars = match search {
        Some(obj) if !obj.is_none() => ids_for(model_id, &var_list_from_py(obj)?)?,
        _ => (0..num_vars).map(|i| VarId(i as u32)).collect(),
    };
    if let Some(var) = extra {
        if !vars.contains(&var) {
            vars.push(var);
        }
    }
    Ok(vars)
}

fn search_ids(model: &PyModel, search: Option<&Bound<'_, PyAny>>, extra: Option<VarId>) -> PyResult<Vec<VarId>> {
    search_ids_for(model.id, model.names.len(), search, extra)
}

fn append_expr_vars(vars: &mut Vec<VarId>, expr: &Expr) {
    let mut objective_vars = Vec::new();
    expr.collect_vars(&mut objective_vars);
    for var in objective_vars {
        append_var(vars, var);
    }
}

fn append_var(vars: &mut Vec<VarId>, var: VarId) {
    if !vars.contains(&var) {
        vars.push(var);
    }
}

fn append_interval_domain_vars(vars: &mut Vec<VarId>, solver: &Solver) {
    for i in 0..solver.store.num_intervals() {
        let interval = IntervalId(i as u32);
        append_var(vars, solver.store.interval_start_var(interval));
        if let Some(presence) = solver.store.interval_presence_var(interval) {
            append_var(vars, presence);
        }
    }
    for i in 0..solver.store.disjunctive_pair_count() {
        append_var(vars, solver.store.disjunctive_order_var(i));
    }
}

fn attach_native_interval_solution(solution: &mut PySolution, intervals: &[NativeIntervalSpec]) {
    if intervals.is_empty() || solution.values.iter().all(Option::is_none) {
        return;
    }
    let mut starts = Vec::with_capacity(intervals.len());
    let mut presences = Vec::with_capacity(intervals.len());
    for interval in intervals {
        let present = interval.presence.and_then(|var| solution.values.get(var.index()).copied().flatten()) != Some(0);
        presences.push(present);
        starts.push(if present { solution.values.get(interval.start.index()).copied().flatten().map(i64::from) } else { None });
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
            if let Some(slot) = values.get_mut(var.index()) {
                *slot = Some(value);
            }
        }
    }
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
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PythonEngine {
    Auto,
    Exact,
    Ls,
}

fn parse_engine(engine: &str) -> PyResult<PythonEngine> {
    match engine {
        "auto" => Ok(PythonEngine::Auto),
        "exact" => Ok(PythonEngine::Exact),
        "ls" => Ok(PythonEngine::Ls),
        _ => Err(PyValueError::new_err("engine must be 'auto', 'exact', or 'ls'")),
    }
}

fn collection_has_max_terms(model: &list::CollectionModel) -> bool {
    model.objectives.iter().any(list::ObjectiveTier::has_max_terms)
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

fn stop_after(limit: u64) -> Arc<AtomicBool> {
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(limit));
            stop.store(true, Ordering::SeqCst);
        });
    }
    stop
}

/// Deadline flag for a solve: armed by a background timer when `time_limit` is
/// set, otherwise never fires on its own (the search runs until it completes or
/// is interrupted).
fn deadline(time_limit: Option<u64>) -> Arc<AtomicBool> {
    match time_limit {
        Some(limit) => stop_after(limit),
        None => Arc::new(AtomicBool::new(false)),
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
/// Python background threads keep running. A watcher thread bridges a Ctrl-C
/// (caught by our OS SIGINT handler) into the shared `stop`, so the stop-aware
/// search unwinds; a `KeyboardInterrupt` is then raised on return.
fn with_interrupts<T, F>(py: Python<'_>, stop: &Arc<AtomicBool>, compute: F) -> PyResult<T>
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    let _sigint = SigintGuard::arm();
    let done = Arc::new(AtomicBool::new(false));
    let interrupted = Arc::new(AtomicBool::new(false));
    let watcher = {
        let stop = Arc::clone(stop);
        let done = Arc::clone(&done);
        let interrupted = Arc::clone(&interrupted);
        std::thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                if SIGINT_TRIPPED.load(Ordering::Relaxed) {
                    interrupted.store(true, Ordering::SeqCst);
                    stop.store(true, Ordering::SeqCst);
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        })
    };
    let result = py.detach(compute);
    done.store(true, Ordering::SeqCst);
    let _ = watcher.join();
    if interrupted.load(Ordering::SeqCst) {
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

fn relation_to_assumption_op(relation: &str) -> PyResult<AssumptionOp> {
    match relation {
        "==" | "=" | "eq" | "Eq" => Ok(AssumptionOp::Eq),
        "!=" | "<>" | "ne" | "Ne" => Ok(AssumptionOp::Ne),
        "<=" | "le" | "Le" => Ok(AssumptionOp::Le),
        "<" | "lt" | "Lt" => Ok(AssumptionOp::Lt),
        ">=" | "ge" | "Ge" => Ok(AssumptionOp::Ge),
        ">" | "gt" | "Gt" => Ok(AssumptionOp::Gt),
        _ => Err(PyValueError::new_err("assumption relation must be one of == != <= < >= >")),
    }
}

fn flipped_op(op: AssumptionOp) -> AssumptionOp {
    match op {
        AssumptionOp::Eq => AssumptionOp::Eq,
        AssumptionOp::Ne => AssumptionOp::Ne,
        AssumptionOp::Le => AssumptionOp::Ge,
        AssumptionOp::Lt => AssumptionOp::Gt,
        AssumptionOp::Ge => AssumptionOp::Le,
        AssumptionOp::Gt => AssumptionOp::Lt,
    }
}

fn var_const_assumption(lhs: &Expr, rhs: &Expr, op: AssumptionOp) -> Option<Assumption> {
    match (lhs, rhs) {
        (Expr::Var(var), Expr::Const(value)) => i32::try_from(*value).ok().map(|value| Assumption { var: *var, op, value }),
        (Expr::Const(value), Expr::Var(var)) => i32::try_from(*value).ok().map(|value| Assumption { var: *var, op: flipped_op(op), value }),
        _ => None,
    }
}

fn simple_assumption_expr(expr: &Expr) -> Option<Assumption> {
    match expr {
        Expr::Eq(lhs, rhs) => var_const_assumption(lhs, rhs, AssumptionOp::Eq),
        Expr::Ne(lhs, rhs) => var_const_assumption(lhs, rhs, AssumptionOp::Ne),
        Expr::Le(lhs, rhs) => var_const_assumption(lhs, rhs, AssumptionOp::Le),
        Expr::Lt(lhs, rhs) => var_const_assumption(lhs, rhs, AssumptionOp::Lt),
        Expr::Ge(lhs, rhs) => var_const_assumption(lhs, rhs, AssumptionOp::Ge),
        Expr::Gt(lhs, rhs) => var_const_assumption(lhs, rhs, AssumptionOp::Gt),
        Expr::Not(inner) => simple_assumption_expr(inner).map(|assumption| {
            let op = match assumption.op {
                AssumptionOp::Eq => AssumptionOp::Ne,
                AssumptionOp::Ne => AssumptionOp::Eq,
                AssumptionOp::Le => AssumptionOp::Gt,
                AssumptionOp::Lt => AssumptionOp::Ge,
                AssumptionOp::Ge => AssumptionOp::Lt,
                AssumptionOp::Gt => AssumptionOp::Le,
            };
            Assumption { op, ..assumption }
        }),
        Expr::Var(var) => Some(Assumption::eq(*var, 1)),
        _ => None,
    }
}

fn assumption_from_py(model_id: u64, item: &Bound<'_, PyAny>) -> PyResult<Assumption> {
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
        return Ok(Assumption::eq(one_id_for(model_id, &var)?, 1));
    }
    if let Ok((var, value)) = item.extract::<(PyRef<'_, PyIntVar>, i32)>() {
        return Ok(Assumption::eq(one_id_for(model_id, &var)?, value));
    }
    if let Ok((var, relation, value)) = item.extract::<(PyRef<'_, PyIntVar>, String, i32)>() {
        return Ok(Assumption { var: one_id_for(model_id, &var)?, op: relation_to_assumption_op(&relation)?, value });
    }
    Err(PyTypeError::new_err("assumption must be a simple Constraint, an IntVar, (IntVar, value), or (IntVar, relation, value)"))
}

fn assumptions_from_py(model_id: u64, assumptions: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Assumption>> {
    let Some(assumptions) = assumptions else {
        return Ok(Vec::new());
    };
    if assumptions.is_none() {
        return Ok(Vec::new());
    }
    let iter = PyIterator::from_object(assumptions)?;
    iter.map(|item| assumption_from_py(model_id, &item?)).collect()
}

fn hint_pairs_from_py(model_id: u64, hints: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<(VarId, i32)>> {
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
            out.push((one_id_for(model_id, &var)?, value));
        } else {
            return Err(PyTypeError::new_err("hints must be an iterable of (IntVar, value) pairs"));
        }
    }
    Ok(out)
}

fn branch_order_from_py(model_id: u64, branch_order: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<VarId>> {
    let Some(branch_order) = branch_order else {
        return Ok(Vec::new());
    };
    if branch_order.is_none() {
        return Ok(Vec::new());
    }
    ids_for(model_id, &var_list_from_py(branch_order)?)
}

fn phase_from_hints(num_vars: usize, hints: &[(VarId, i32)]) -> Vec<Option<i32>> {
    let mut phase = vec![None; num_vars];
    for &(var, value) in hints {
        if let Some(slot) = phase.get_mut(var.index()) {
            *slot = Some(value);
        }
    }
    phase
}

fn objective_specs(primary: &Option<ObjectiveSpec>, tiers: &[ObjectiveSpec]) -> Vec<ObjectiveSpec> {
    let mut out = Vec::new();
    if let Some(objective) = primary {
        out.push(objective.clone());
    }
    out.extend_from_slice(tiers);
    out
}

fn expr_bounds_i32(solver: &Solver, expr: &Expr) -> PyResult<(i32, i32)> {
    let (lo, hi) = expr.bounds(&|var| (i64::from(solver.store.min(var)), i64::from(solver.store.max(var))));
    Ok((checked_i32(lo, "objective lower bound")?, checked_i32(hi, "objective upper bound")?))
}

fn materialize_objectives(solver: &mut Solver, objectives: &[ObjectiveSpec]) -> PyResult<Vec<MaterializedObjective>> {
    let mut out = Vec::with_capacity(objectives.len());
    for objective in objectives {
        let mut support = Vec::new();
        objective.expr.expr.collect_vars(&mut support);
        support.sort_unstable();
        support.dedup();
        let var = match &objective.expr.expr {
            Expr::Var(var) => *var,
            expr => {
                let (lo, hi) = expr_bounds_i32(solver, expr)?;
                let obj = solver.new_var_range(lo, hi);
                intension::intension(solver, expr::eq(expr::var(obj), expr.clone()));
                obj
            }
        };
        out.push(MaterializedObjective { minimizing: objective.minimizing, var, text: objective.expr.text.clone(), support });
    }
    Ok(out)
}

fn call_incumbent(
    py: Python<'_>,
    callback: Option<&Bound<'_, PyAny>>,
    value: i64,
    vars: &[VarId],
    assignment: &[i32],
    num_vars: usize,
) -> PyResult<()> {
    let Some(callback) = callback else {
        return Ok(());
    };
    let values = PyDict::new(py);
    for (&var, &value) in vars.iter().zip(assignment) {
        if var.index() < num_vars {
            values.set_item(var.0, value)?;
        }
    }
    callback.call1((value, values))?;
    Ok(())
}

fn add_stats(total: &mut SolveStats, stats: SolveStats) {
    total.solutions += stats.solutions;
    total.nodes += stats.nodes;
    total.failures += stats.failures;
    total.learned_lits += stats.learned_lits;
    total.vivified_clauses += stats.vivified_clauses;
    total.vivified_lits += stats.vivified_lits;
    total.binary_clause_visits += stats.binary_clause_visits;
    total.watched_clause_visits += stats.watched_clause_visits;
    total.watched_literal_scans += stats.watched_literal_scans;
    total.binary_implications += stats.binary_implications;
}

fn next_clause_sharing(clauses: Option<&Arc<SharedClausePool>>, next_worker: &mut usize) -> Option<ClauseSharing> {
    clauses.map(|clauses| {
        let worker = *next_worker;
        *next_worker = next_worker.saturating_add(1);
        ClauseSharing::new(Arc::clone(clauses), worker)
    })
}

fn atom_table_for_solver(solver: &Solver, vars: &[VarId], clauses: &SharedClausePool) -> AtomTable {
    let nvars = solver.store.num_vars();
    let mut active = (0..nvars).map(|i| solver.store.is_relevant(VarId(i as u32))).collect::<Vec<_>>();
    for &var in vars {
        if var.index() < active.len() {
            active[var.index()] = true;
        }
    }
    AtomTable::build_active_sparse_with_registry(
        nvars,
        |v: VarId| active[v.index()],
        |v: VarId| solver.store.size(v) == 2 && solver.store.contains(v, -1) && solver.store.contains(v, 1),
        |v: VarId| solver.store.sparse_values(v),
        |v: VarId| (solver.store.min(v), solver.store.max(v)),
        clauses.lazy_atoms(),
    )
}

fn decode_nogood_lit(atoms: &AtomTable, lit: Lit) -> (u32, String, i32) {
    match atoms.decode(lit.atom()) {
        AtomKind::Ge { var, k } if lit.is_positive() => (var.0, ">=".to_string(), k),
        AtomKind::Ge { var, k } => (var.0, "<".to_string(), k),
        AtomKind::Eq { var, v } if lit.is_positive() => (var.0, "==".to_string(), v),
        AtomKind::Eq { var, v } => (var.0, "!=".to_string(), v),
    }
}

fn append_objective_search_vars(vars: &mut Vec<VarId>, objectives: &[MaterializedObjective]) {
    for objective in objectives {
        for &var in &objective.support {
            append_var(vars, var);
        }
        append_var(vars, objective.var);
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_integer_template(
    solver_template: &Solver,
    model_id: u64,
    num_vars: usize,
    search_vars: &[VarId],
    objectives: &[MaterializedObjective],
    assumptions: &[Assumption],
    hints: &[(VarId, i32)],
    stop: &Arc<AtomicBool>,
    seed: u64,
    clauses: Option<&Arc<SharedClausePool>>,
    next_worker: &mut usize,
    conflict_budget: Option<u64>,
    branch_order: &[VarId],
    verbose: bool,
    on_incumbent: Option<Py<PyAny>>,
) -> PyResult<PySolution> {
    let mut vars = search_vars.to_vec();
    for &var in branch_order {
        append_var(&mut vars, var);
    }
    append_interval_domain_vars(&mut vars, solver_template);
    append_objective_search_vars(&mut vars, objectives);
    let mut remaining_conflicts = conflict_budget;

    if objectives.is_empty() {
        let mut solver = solver_template.clone();
        let phase = phase_from_hints(solver.store.num_vars(), hints);
        let sharing = next_clause_sharing(clauses, next_worker);
        let (assignment, stats, complete) = search::decide_sat_assuming_seeded(
            &mut solver,
            &vars,
            assumptions,
            stop,
            seed,
            sharing,
            remaining_conflicts,
            phase,
            branch_order.to_vec(),
        );
        let status = match (assignment.is_some(), complete) {
            (true, _) => "SATISFIABLE",
            (false, true) => "UNSATISFIABLE",
            (false, false) => "UNKNOWN",
        };
        return Ok(make_solution(status, &vars, assignment.as_deref(), None, None, None, stats, num_vars));
    }

    let mut active_assumptions = assumptions.to_vec();
    let mut total_stats = SolveStats::default();
    let mut best_assignment: Option<Vec<i32>> = None;
    let mut objective_values = Vec::with_capacity(objectives.len());
    let mut complete = true;

    for (tier, objective) in objectives.iter().enumerate() {
        if remaining_conflicts == Some(0) || stop.load(Ordering::Relaxed) {
            complete = false;
            break;
        }
        let mut solver = solver_template.clone();
        let phase = phase_from_hints(solver.store.num_vars(), hints);
        let sharing = next_clause_sharing(clauses, next_worker);
        let mut callback_error: Option<PyErr> = None;
        let (best, stats, tier_complete) = search::optimize_assuming_seeded(
            &mut solver,
            &vars,
            &active_assumptions,
            SearchObjective::Var(objective.var),
            objective.minimizing,
            stop,
            seed.wrapping_add(tier as u64),
            None,
            sharing,
            remaining_conflicts,
            phase,
            branch_order.to_vec(),
            |value, assignment| {
                if verbose {
                    println!("  incumbent tier {tier}: {value}");
                }
                if callback_error.is_none() {
                    if let Some(cb) = &on_incumbent {
                        // The solve runs with the GIL released; reacquire it
                        // only for the callback invocation itself.
                        let res = Python::attach(|py| call_incumbent(py, Some(cb.bind(py)), value, &vars, assignment, num_vars));
                        if let Err(err) = res {
                            callback_error = Some(err);
                            stop.store(true, Ordering::SeqCst);
                        }
                    }
                }
            },
        );
        if let Some(err) = callback_error {
            return Err(err);
        }
        if let Some(limit) = remaining_conflicts.as_mut() {
            *limit = limit.saturating_sub(stats.failures);
        }
        add_stats(&mut total_stats, stats);
        let Some((assignment, value)) = best else {
            let status = if tier == 0 && tier_complete { "UNSATISFIABLE" } else { "UNKNOWN" };
            return Ok(make_solution(
                status,
                &vars,
                best_assignment.as_deref(),
                objective_values.first().copied(),
                Some(if objectives[0].minimizing { "min" } else { "max" }),
                Some(&objectives[0].text),
                total_stats,
                num_vars,
            ));
        };
        best_assignment = Some(assignment);
        objective_values.push(value);
        let value_i32 = checked_i32(value, "objective value")?;
        active_assumptions.push(Assumption::eq(objective.var, value_i32));
        if !tier_complete {
            complete = false;
            break;
        }
    }

    let mut solution = make_solution(
        if complete && objective_values.len() == objectives.len() { "OPTIMAL" } else { "SATISFIABLE" },
        &vars,
        best_assignment.as_deref(),
        objective_values.first().copied(),
        Some(if objectives[0].minimizing { "min" } else { "max" }),
        Some(&objectives[0].text),
        total_stats,
        num_vars,
    );
    solution.objectives = objective_values;
    let _ = model_id;
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
impl PySolveSession {
    #[getter]
    fn learned_nogoods(&self) -> usize {
        self.clauses.len()
    }

    fn clear_nogoods(&mut self) {
        self.clauses = Arc::new(SharedClausePool::default());
        self.next_worker = 0;
    }

    #[pyo3(signature = (limit=None))]
    fn raw_nogoods(&self, limit: Option<usize>) -> Vec<PyRawNogood> {
        self.clauses
            .snapshot(limit.unwrap_or(0))
            .into_iter()
            .map(|(lbd, lits)| (lbd, lits.iter().map(|lit| lit.code()).collect()))
            .collect()
    }

    #[pyo3(signature = (limit=None))]
    fn nogoods(&self, limit: Option<usize>) -> Vec<PyNogood> {
        let atoms = atom_table_for_solver(&self.solver, &self.search, &self.clauses);
        self.clauses
            .snapshot(limit.unwrap_or(0))
            .into_iter()
            .map(|(lbd, lits)| (lbd, lits.iter().copied().map(|lit| decode_nogood_lit(&atoms, lit)).collect()))
            .collect()
    }

    #[pyo3(signature = (*, search=None, assumptions=None, hints=None, branch_order=None, on_incumbent=None, verbose=false, time_limit=None, seed=0, conflict_budget=None))]
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
    ) -> PyResult<PySolution> {
        if let Some(callback) = on_incumbent {
            if !callback.is_callable() {
                return Err(PyTypeError::new_err("on_incumbent must be callable"));
            }
        }
        if verbose {
            verbose_start(self.names.len(), self.solver.num_propagators(), !self.objectives.is_empty());
        }
        let assumptions = assumptions_from_py(self.id, assumptions)?;
        let hints = hint_pairs_from_py(self.id, hints)?;
        let branch_order = branch_order_from_py(self.id, branch_order)?;
        let search_vars = match search {
            Some(obj) if !obj.is_none() => search_ids_for(self.id, self.names.len(), Some(obj), None)?,
            _ => self.search.clone(),
        };
        let stop = deadline(time_limit);
        let clauses = Arc::clone(&self.clauses);
        // Owned/plain captures so the compute closure is Send and the solve
        // runs with the GIL released (the callback reacquires it on its own).
        let on_incumbent = on_incumbent.map(|cb| cb.clone().unbind());
        let solver_template = self.solver.clone();
        let (model_id, num_vars, objectives) = (self.id, self.names.len(), self.objectives.clone());
        let worker0 = self.next_worker;
        let stop_inner = Arc::clone(&stop);
        let (solution, worker) = with_interrupts(py, &stop, move || {
            let mut worker = worker0;
            let solution = solve_integer_template(
                &solver_template,
                model_id,
                num_vars,
                &search_vars,
                &objectives,
                &assumptions,
                &hints,
                &stop_inner,
                seed,
                Some(&clauses),
                &mut worker,
                conflict_budget,
                &branch_order,
                verbose,
                on_incumbent,
            );
            (solution, worker)
        })?;
        self.next_worker = worker;
        let mut solution = solution?;
        attach_native_interval_solution(&mut solution, &self.native_intervals);
        if verbose {
            verbose_finish(&solution);
        }
        Ok(solution)
    }

    fn __repr__(&self) -> String {
        format!("SolveSession(num_vars={}, objectives={}, learned_nogoods={})", self.names.len(), self.objectives.len(), self.clauses.len())
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
            then_objectives: Vec::new(),
            native_intervals: Vec::new(),
            local: LocalSearchSpec::default(),
            col_universe: None,
            col_lists: 0,
            col_gen: 0,
            col_objectives: Vec::new(),
            col_constraints: Vec::new(),
            col_globals: Vec::new(),
            col_schedule: None,
            col_sched_gen: 0,
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
            self.col_constraints.push(list::Constraint { reduction: lc.reduction.clone(), op: lc.op, rhs: lc.rhs });
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

    #[pyo3(signature = (items, durations=None))]
    fn no_overlap(&mut self, items: &Bound<'_, PyAny>, durations: Option<Vec<i64>>) -> PyResult<()> {
        if let Some(durations) = durations {
            let starts = ids_for(self.id, &var_list_from_py(items)?)?;
            if starts.len() != durations.len() {
                return Err(PyValueError::new_err("starts and durations must have the same length"));
            }
            let origins: Vec<Vec<VarId>> = starts.iter().map(|&s| vec![s]).collect();
            let lengths: Vec<Vec<Expr>> = durations.iter().map(|&d| vec![expr::int(d)]).collect();
            self.local.add_no_overlap(origins, lengths, false);
            scheduling::no_overlap(&mut self.solver, &starts, &durations);
            return Ok(());
        }

        let intervals = interval_list_from_py(items)?;
        if intervals.iter().all(|iv| iv.kind == PyIntervalKind::Native) {
            let ids = self.native_interval_specs(&intervals)?.into_iter().map(|spec| spec.interval).collect::<Vec<_>>();
            interval_constraints::no_overlap(&mut self.solver, &ids);
        } else if intervals.iter().all(|iv| iv.kind == PyIntervalKind::Schedule) {
            for iv in &intervals {
                self.check_interval_scope(iv)?;
            }
            let idx = intervals.iter().map(|iv| iv.index as usize).collect();
            let sched = self.col_schedule.as_mut().ok_or_else(|| PyValueError::new_err("create intervals before no_overlap"))?;
            sched.resources.push(list::Resource::NoOverlap(idx));
        } else {
            return Err(PyValueError::new_err("cannot mix native intervals and schedule intervals in one no_overlap"));
        }
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

    /// Create `count` ordered list variables over `items`.
    ///
    /// With `optional=True`, an extra hidden pool list is added that no returned
    /// handle references, so items may remain unassigned to the visible lists.
    #[pyo3(signature = (items, count, *, optional=false))]
    fn list_vars(&mut self, items: Vec<i32>, count: usize, optional: bool) -> PyResult<Vec<PyListVar>> {
        if items.is_empty() {
            return Err(PyValueError::new_err("list items cannot be empty"));
        }
        if count == 0 {
            return Err(PyValueError::new_err("need at least one list"));
        }
        if !self.names.is_empty() || self.objective.is_some() || !self.then_objectives.is_empty() {
            return Err(PyValueError::new_err("cannot mix integer variables with list_vars; use one modeling style per model"));
        }
        if self.col_schedule.is_some() {
            return Err(PyValueError::new_err("model already has interval variables; use one domain style per model (list or interval)"));
        }
        // The item set is partitioned among the lists; duplicate ids would
        // be independent positions that share a value, which silently diverges
        // from set semantics (and from how reductions aggregate by value).
        let mut seen = std::collections::HashSet::with_capacity(items.len());
        if let Some(dup) = items.iter().find(|v| !seen.insert(**v)) {
            return Err(PyValueError::new_err(format!("list items have a duplicate value {dup}; items must be distinct")));
        }
        self.col_universe = Some(items);
        self.col_lists = if optional { count + 1 } else { count };
        self.col_gen += 1;
        // Reset any reductions recorded against a previous `list_vars` generation.
        self.col_objectives.clear();
        self.col_constraints.clear();
        self.col_globals.clear();
        let gen = self.col_gen;
        Ok((0..count).map(|i| PyListVar { model_id: self.id, gen, index: i as u32 }).collect())
    }

    /// Precedence over list items or interval variables.
    fn precedence(&mut self, before: &Bound<'_, PyAny>, after: &Bound<'_, PyAny>) -> PyResult<()> {
        if let (Ok(a), Ok(b)) = (before.extract::<PyRef<'_, PyIntervalVar>>(), after.extract::<PyRef<'_, PyIntervalVar>>()) {
            if a.kind != b.kind {
                return Err(PyValueError::new_err("cannot mix native intervals and schedule intervals in one precedence"));
            }
            if a.kind == PyIntervalKind::Native {
                let before = self.native_interval_spec(&a)?.interval;
                let after = self.native_interval_spec(&b)?.interval;
                interval_constraints::interval_precedence(&mut self.solver, before, after);
            } else {
                self.check_interval_scope(&a)?;
                self.check_interval_scope(&b)?;
                let sched = self.col_schedule.as_mut().ok_or_else(|| PyValueError::new_err("create intervals before precedence"))?;
                sched.precedences.push((a.index as usize, b.index as usize));
            }
            return Ok(());
        }
        let before =
            before.extract::<i32>().map_err(|_| PyTypeError::new_err("precedence expects two item ids or two IntervalVar handles"))?;
        let after =
            after.extract::<i32>().map_err(|_| PyTypeError::new_err("precedence expects two item ids or two IntervalVar handles"))?;
        self.col_globals.push(list::GlobalConstraint::ListLe { before, after });
        Ok(())
    }

    /// Require two items to share a list (same vehicle, same bin).
    fn same_list(&mut self, a: i32, b: i32) {
        self.col_globals.push(list::GlobalConstraint::SameList { a, b });
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
    /// exactly one member executes, and the returned master synchronises with
    /// it (`.start` is the shared start, `.end`/`.realized_duration` follow the
    /// chosen member). A dedicated bounds channel confines the master's start
    /// to the union of the still-capable members' windows and rules out members
    /// whose window it can no longer reach.
    #[pyo3(signature = (members, *, name=None))]
    fn alternative(&mut self, members: &Bound<'_, PyAny>, name: Option<String>) -> PyResult<PyIntervalVar> {
        let members = interval_list_from_py(members)?;
        if members.is_empty() {
            return Err(PyValueError::new_err("alternative needs at least one member interval"));
        }
        let mut ids: Vec<IntervalId> = Vec::with_capacity(members.len());
        let mut presences: Vec<VarId> = Vec::with_capacity(members.len());
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
            start_lo = start_lo.min(self.solver.store.interval_start_min(spec.interval));
            start_hi = start_hi.max(self.solver.store.interval_start_max(spec.interval));
            ids.push(spec.interval);
            presences.push(presence);
            durations.push(i64::from(spec.duration));
        }

        let shared_start = self.solver.new_var_range(start_lo, start_hi);
        self.register_native_backing_var(shared_start, name.as_ref().map(|name| format!("{name}.start")));

        interval_constraints::exactly_one_mode(&mut self.solver, &ids);
        interval_constraints::alternative_channel(&mut self.solver, shared_start, &ids);

        let name = name.unwrap_or_else(|| format!("alternative{}", shared_start.0));
        Ok(PyIntervalVar {
            model_id: self.id,
            gen: 0,
            index: u32::MAX,
            kind: PyIntervalKind::Native,
            interval: None,
            start: Some(shared_start),
            presence: None,
            duration: None,
            alternative: Some(AlternativeInterval { name, members, presences, durations }),
        })
    }

    /// Create intervals whose mode is selected from `(machine, duration)` pairs.
    fn alternatives(&mut self, modes: Vec<Vec<(usize, i64)>>, horizon: i64) -> PyResult<Vec<PyIntervalVar>> {
        if modes.iter().any(|m| m.is_empty()) {
            return Err(PyValueError::new_err("each interval needs at least one (machine, duration) mode"));
        }
        self.enter_schedule_mode()?;
        let intervals = modes
            .iter()
            .map(|opts| list::IntervalVar {
                duration: 0,
                horizon,
                modes: opts.iter().map(|&(machine, duration)| list::Mode { machine, duration }).collect(),
            })
            .collect();
        self.col_schedule = Some(list::Schedule { intervals, precedences: Vec::new(), resources: Vec::new(), minimize_makespan: true });
        let gen = self.col_sched_gen;
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

    /// Moded intervals that choose the same machine never overlap.
    fn no_overlap_by_machine(&mut self) -> PyResult<()> {
        let sched = self.col_schedule.as_mut().ok_or_else(|| PyValueError::new_err("create alternatives before no_overlap_by_machine"))?;
        sched.resources.push(list::Resource::MachineNoOverlap);
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
                self.native_interval_specs(&intervals)?
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
        let sched = self.col_schedule.as_mut().ok_or_else(|| PyValueError::new_err("create intervals before minimize_makespan"))?;
        sched.minimize_makespan = true;
        Ok(())
    }

    /// A renewable resource of `capacity`: `demands` are `(interval, amount)`
    /// pairs whose total over any instant may not exceed the capacity.
    fn resource(&mut self, demands: Vec<(PyIntervalVar, i64)>, capacity: i64) -> PyResult<()> {
        if demands.iter().all(|(iv, _)| iv.kind == PyIntervalKind::Native) {
            let mut intervals = Vec::with_capacity(demands.len());
            let mut heights = Vec::with_capacity(demands.len());
            for (iv, amount) in &demands {
                intervals.push(self.native_interval_spec(iv)?.interval);
                heights.push(checked_i32(*amount, "cumulative demand")?);
            }
            interval_constraints::cumulative(&mut self.solver, &intervals, &heights, checked_i32(capacity, "cumulative capacity")?);
        } else if demands.iter().all(|(iv, _)| iv.kind == PyIntervalKind::Schedule) {
            for (iv, _) in &demands {
                self.check_interval_scope(iv)?;
            }
            let demands = demands.iter().map(|(iv, amount)| (iv.index as usize, *amount)).collect();
            let sched = self.col_schedule.as_mut().ok_or_else(|| PyValueError::new_err("create intervals before resource"))?;
            sched.resources.push(list::Resource::Cumulative { demands, capacity });
        } else {
            return Err(PyValueError::new_err("cannot mix native intervals and schedule intervals in one resource"));
        }
        Ok(())
    }

    #[pyo3(signature = (*, search=None, assumptions=None, hints=None, branch_order=None, on_incumbent=None, verbose=false, time_limit=None, seed=0, engine="auto", conflict_budget=None))]
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
        engine: &str,
        conflict_budget: Option<u64>,
    ) -> PyResult<PySolution> {
        let engine = parse_engine(engine)?;
        let exact_hooks = assumptions.is_some_and(|obj| !obj.is_none())
            || hints.is_some_and(|obj| !obj.is_none())
            || branch_order.is_some_and(|obj| !obj.is_none())
            || on_incumbent.is_some()
            || conflict_budget.is_some();
        if self.col_universe.is_some() || self.col_schedule.is_some() {
            if exact_hooks {
                return Err(PyValueError::new_err(
                    "assumptions, hints, branch_order, callbacks, and conflict_budget are only supported by the integer exact engine",
                ));
            }
            if !self.names.is_empty() || self.objective.is_some() || !self.then_objectives.is_empty() {
                return Err(PyValueError::new_err(
                    "model mixes integer variables with list/interval variables; use one modeling style per model",
                ));
            }
            return self.solve_collection(py, time_limit, seed, verbose, engine);
        }
        if engine == PythonEngine::Ls {
            if !self.native_intervals.is_empty() {
                return Err(PyValueError::new_err("engine='ls' does not support native interval variables"));
            }
            if exact_hooks {
                return Err(PyValueError::new_err(
                    "assumptions, hints, branch_order, callbacks, and conflict_budget are only supported by the integer exact engine",
                ));
            }
            let Some(objective) = &self.objective else {
                return Err(PyValueError::new_err("engine='ls' requires an objective"));
            };
            return self.solve_ls(py, objective, search, verbose, time_limit, seed);
        }
        if !exact_hooks && self.then_objectives.is_empty() {
            if let Some(objective) = &self.objective {
                return self.solve_single_optimization(py, objective, search, &[], verbose, time_limit, seed);
            }
        }
        let objective_specs = objective_specs(&self.objective, &self.then_objectives);
        if !objective_specs.is_empty() || exact_hooks {
            if let Some(callback) = on_incumbent {
                if !callback.is_callable() {
                    return Err(PyTypeError::new_err("on_incumbent must be callable"));
                }
            }
            if verbose {
                verbose_start(self.names.len(), self.solver.num_propagators(), !objective_specs.is_empty());
            }
            let assumptions = assumptions_from_py(self.id, assumptions)?;
            let hints = hint_pairs_from_py(self.id, hints)?;
            let branch_order = branch_order_from_py(self.id, branch_order)?;
            let mut solver = self.solver.clone();
            let objectives = materialize_objectives(&mut solver, &objective_specs)?;
            let mut search_vars = search_ids_for(self.id, self.names.len(), search, None)?;
            append_interval_domain_vars(&mut search_vars, &solver);
            let stop = deadline(time_limit);
            // Owned/plain captures: the solve detaches from the GIL, and only
            // the incumbent callback reattaches per invocation.
            let on_incumbent = on_incumbent.map(|cb| cb.clone().unbind());
            let (model_id, num_vars) = (self.id, self.names.len());
            let stop_inner = Arc::clone(&stop);
            let mut solution = with_interrupts(py, &stop, move || {
                let mut worker = 0usize;
                solve_integer_template(
                    &solver,
                    model_id,
                    num_vars,
                    &search_vars,
                    &objectives,
                    &assumptions,
                    &hints,
                    &stop_inner,
                    seed,
                    None,
                    &mut worker,
                    conflict_budget,
                    &branch_order,
                    verbose,
                    on_incumbent,
                )
            })??;
            attach_native_interval_solution(&mut solution, &self.native_intervals);
            if verbose {
                verbose_finish(&solution);
            }
            return Ok(solution);
        }
        if verbose {
            verbose_start(self.names.len(), self.solver.num_propagators(), false);
        }
        let mut vars = search_ids(self, search, None)?;
        let mut solver = self.solver.clone();
        append_interval_domain_vars(&mut vars, &solver);
        let stop = deadline(time_limit);
        let mut assignment = None;
        let stats = with_interrupts(py, &stop, || {
            search::solve_interruptible_seeded(
                &mut solver,
                &vars,
                |solver| {
                    assignment = Some(vars.iter().map(|&var| solver.store.value(var)).collect::<Vec<_>>());
                    SearchControl::Stop
                },
                &stop,
                seed,
            )
        })?;
        let status = if assignment.is_some() { "SATISFIABLE" } else { "UNSATISFIABLE" };
        let mut solution = make_solution(status, &vars, assignment.as_deref(), None, None, None, stats, self.names.len());
        attach_native_interval_solution(&mut solution, &self.native_intervals);
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

    /// Set the primary minimization objective.
    fn minimize(&mut self, objective: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(term) = objective.extract::<PyRef<'_, PyTerm>>() {
            self.push_collection_tier(&term, true, true)?;
            return Ok(());
        }
        self.set_integer_objective(objective, true)
    }

    /// Set the primary maximization objective.
    fn maximize(&mut self, objective: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(term) = objective.extract::<PyRef<'_, PyTerm>>() {
            self.push_collection_tier(&term, false, true)?;
            return Ok(());
        }
        self.set_integer_objective(objective, false)
    }

    /// Append a lower-priority lexicographic tier.
    fn then_minimize(&mut self, objective: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(term) = objective.extract::<PyRef<'_, PyTerm>>() {
            return self.push_collection_tier(&term, true, false);
        }
        self.push_integer_tier(objective, true)
    }

    fn then_maximize(&mut self, objective: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(term) = objective.extract::<PyRef<'_, PyTerm>>() {
            return self.push_collection_tier(&term, false, false);
        }
        self.push_integer_tier(objective, false)
    }

    fn session(&self) -> PyResult<PySolveSession> {
        if self.col_universe.is_some() || self.col_schedule.is_some() {
            return Err(PyValueError::new_err("SolveSession is currently supported for integer exact models"));
        }
        let objective_specs = objective_specs(&self.objective, &self.then_objectives);
        let mut solver = self.solver.clone();
        let objectives = materialize_objectives(&mut solver, &objective_specs)?;
        let mut search = (0..self.names.len()).map(|i| VarId(i as u32)).collect::<Vec<_>>();
        append_interval_domain_vars(&mut search, &solver);
        append_objective_search_vars(&mut search, &objectives);
        // Freeze the atom layout for the whole session: the eager LCG atom ids
        // depend on the ACTIVE set (relevant ∪ search vars), and the session's
        // kept nogoods are re-injected by raw atom id across epochs. Marking the
        // full session universe relevant makes the layout invariant, so a
        // per-epoch `search` only guides branching and can never reinterpret a
        // stored nogood (nor trip `set_base`'s layout assert).
        for &v in &search {
            solver.store.mark_relevant(v);
        }
        Ok(PySolveSession {
            id: self.id,
            solver,
            names: self.names.clone(),
            objectives,
            search,
            native_intervals: self.native_intervals.clone(),
            clauses: Arc::new(SharedClausePool::default()),
            next_worker: 0,
        })
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

    fn register_native_backing_var(&mut self, var: VarId, name: Option<String>) {
        while self.names.len() <= var.index() {
            self.names.push(None);
        }
        if name.is_some() || self.names[var.index()].is_none() {
            self.names[var.index()] = name;
        }
    }

    fn enter_native_interval_mode(&self) -> PyResult<()> {
        if self.col_universe.is_some() {
            return Err(PyValueError::new_err("model already has list variables; use one domain style per model"));
        }
        if self.col_schedule.is_some() {
            return Err(PyValueError::new_err("model already has schedule intervals; use native intervals or alternatives, not both"));
        }
        Ok(())
    }

    fn create_native_interval(&mut self, duration: i64, horizon: i64, optional: bool, name: Option<String>) -> PyResult<PyIntervalVar> {
        self.enter_native_interval_mode()?;
        let duration_i32 = checked_i32(duration, "interval duration")?;
        let start_max = checked_interval_start_max(horizon, duration)?;
        let interval = if optional {
            self.solver.store.new_optional_interval(0, start_max, duration_i32)
        } else {
            self.solver.store.new_interval(0, start_max, duration_i32)
        };
        let start = self.solver.store.interval_start_var(interval);
        self.register_native_backing_var(start, name.as_ref().map(|name| format!("{name}.start")));
        let presence = self.solver.store.interval_presence_var(interval);
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

    fn set_native_makespan_objective(&mut self, intervals: &[NativeIntervalSpec]) -> PyResult<()> {
        if intervals.is_empty() {
            return Err(PyValueError::new_err("minimize_makespan needs at least one interval"));
        }
        let upper = intervals.iter().map(|spec| self.solver.store.interval_end_max(spec.interval)).max().unwrap_or(0);
        let makespan = self.solver.new_var_range(0, upper.max(0));
        self.register_native_backing_var(makespan, Some("makespan".to_string()));
        for spec in intervals {
            let end = expr::add(vec![expr::var(spec.start), expr::int(i64::from(spec.duration))]);
            let bound = expr::ge(expr::var(makespan), end);
            let constraint =
                if let Some(presence) = spec.presence { expr::imp(expr::eq(expr::var(presence), expr::int(1)), bound) } else { bound };
            intension::intension(&mut self.solver, constraint);
        }
        self.objective = Some(ObjectiveSpec {
            minimizing: true,
            expr: ExprLike { model_id: Some(self.id), expr: Expr::Var(makespan), text: "makespan".to_string() },
        });
        self.then_objectives.clear();
        Ok(())
    }

    /// Solve the recorded list-domain model (list variables + reductions) with the
    /// collection local-search engine, time-limited (default 5s).
    /// Reject a term that belongs to a different model or to a superseded
    /// `list_vars` generation.
    fn check_term_scope(&self, model_id: u64, gen: u64) -> PyResult<()> {
        if model_id != self.id {
            return Err(PyValueError::new_err("this list term belongs to a different model"));
        }
        if gen != self.col_gen {
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
            self.col_objectives.clear();
        } else if self.col_objectives.len() >= list::MAX_TIERS {
            return Err(PyValueError::new_err(format!("at most {} objective tiers are supported", list::MAX_TIERS)));
        }
        self.col_objectives.push(list::ObjectiveTier { minimize, terms: term.reductions.clone(), max_terms: term.max_terms.clone() });
        Ok(())
    }

    fn set_integer_objective(&mut self, objective: &Bound<'_, PyAny>, minimizing: bool) -> PyResult<()> {
        if self.col_universe.is_some() || self.col_schedule.is_some() {
            return Err(PyValueError::new_err("integer objectives cannot be mixed with list or interval models"));
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
        self.objective = Some(ObjectiveSpec { minimizing, expr });
        self.then_objectives.clear();
        Ok(())
    }

    fn push_integer_tier(&mut self, objective: &Bound<'_, PyAny>, minimizing: bool) -> PyResult<()> {
        if self.objective.is_none() {
            return Err(PyValueError::new_err("then_minimize/then_maximize requires a primary integer objective first"));
        }
        if self.col_universe.is_some() || self.col_schedule.is_some() {
            return Err(PyValueError::new_err("integer objective tiers cannot be mixed with list or interval models"));
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
        self.then_objectives.push(ObjectiveSpec { minimizing, expr });
        Ok(())
    }

    /// Begin (or restart) interval-schedule mode. Rejects mixing with integer or
    /// list variables, and bumps the schedule generation so handles from an
    /// earlier `intervals`/`alternatives` call become stale.
    fn enter_schedule_mode(&mut self) -> PyResult<()> {
        if !self.names.is_empty() || !self.native_intervals.is_empty() || self.objective.is_some() || !self.then_objectives.is_empty() {
            return Err(PyValueError::new_err("cannot mix integer variables with interval variables; use one modeling style per model"));
        }
        if self.col_universe.is_some() {
            return Err(PyValueError::new_err("model already has list variables; use one domain style per model (list or interval)"));
        }
        self.col_sched_gen += 1;
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
        if iv.gen != self.col_sched_gen {
            return Err(PyValueError::new_err("this interval is stale; rebuild it from the current intervals()/alternatives()"));
        }
        Ok(())
    }

    fn try_solve_domain_collection(
        &self,
        py: Python<'_>,
        model: &list::CollectionModel,
        selection: &shared_model::BackendSelection,
        time_limit: Option<u64>,
        seed: u64,
        verbose: bool,
    ) -> PyResult<Option<PySolution>> {
        if selection.backend != shared_model::Backend::DomainExact {
            return Ok(None);
        }
        if let Some(schedule) = &model.schedule {
            return self.try_solve_domain_schedule(py, schedule, model, selection, time_limit, seed, verbose);
        }
        self.try_solve_domain_lists(py, model, selection, time_limit, verbose)
    }

    fn try_solve_routing_integer(
        &self,
        py: Python<'_>,
        model: &list::CollectionModel,
        selection: &shared_model::BackendSelection,
        time_limit: Option<u64>,
        seed: u64,
        verbose: bool,
    ) -> PyResult<Option<PySolution>> {
        if selection.class != shared_model::ModelClass::Routing || selection.backend != shared_model::Backend::IntegerExact {
            return Ok(None);
        }

        let limit = time_limit.unwrap_or(5);
        let primary_sense = model.objectives.first().map_or("min", |tier| if tier.minimize { "min" } else { "max" });
        if verbose {
            println!("qayd solve (integer routing)");
            println!("  class: {}", selection.class.name());
            println!("  backend: {}", selection.backend.name());
            println!("  reason: {}", selection.reason);
            println!("  items: {}", model.items.len());
            println!("  lists: {}", model.lists);
            println!("  constraints: {}", model.constraints.len());
            println!("  objective tiers: {}", model.objectives.len());
            println!("  time limit: {limit}s");
        }

        let stop = stop_after(limit);
        let start = Instant::now();
        let mut report = |objective: i64| {
            if verbose {
                println!("  o {objective}  ({primary_sense}, {:.2}s)", start.elapsed().as_secs_f64());
            }
        };
        let Some(outcome) = with_interrupts(py, &stop, || routing_engine::solve_collection(model, seed, &stop, &mut report))? else {
            return Ok(None);
        };
        let sol = outcome.solution;
        let status = if sol.feasible {
            if outcome.complete {
                "OPTIMAL"
            } else {
                "SATISFIABLE"
            }
        } else if outcome.complete {
            "UNSATISFIABLE"
        } else {
            "UNKNOWN"
        };
        if verbose {
            println!("qayd result (integer routing)");
            println!("  status: {status}");
            if sol.feasible {
                println!("  objectives: {:?}", sol.objectives);
            }
            println!("  improvements: {}", outcome.improvements);
            println!("  solutions: {}", outcome.stats.solutions);
            println!("  nodes: {}", outcome.stats.nodes);
            println!("  failures: {}", outcome.stats.failures);
        }

        let objectives = if sol.feasible { sol.objectives.clone() } else { Vec::new() };
        Ok(Some(PySolution {
            status: status.to_string(),
            objective: sol.feasible.then(|| sol.objectives.first().copied().unwrap_or(0)),
            objective_sense: Some(primary_sense.to_string()),
            objective_expr: Some("integer routing edge-sum".to_string()),
            values: Vec::new(),
            stats: outcome.stats.into(),
            lists: sol.feasible.then(|| sol.lists.clone()),
            objectives,
            starts: Vec::new(),
            presences: Vec::new(),
            machines: Vec::new(),
        }))
    }

    fn try_solve_domain_lists(
        &self,
        py: Python<'_>,
        model: &list::CollectionModel,
        selection: &shared_model::BackendSelection,
        time_limit: Option<u64>,
        verbose: bool,
    ) -> PyResult<Option<PySolution>> {
        if selection.backend != shared_model::Backend::DomainExact {
            return Ok(None);
        }
        let Some(objective_tiers) = shared_model::list_objective_tiers(&model.objectives, &model.items) else {
            return Ok(None);
        };
        let has_objective = !objective_tiers.is_empty();
        let minimize = objective_tiers.first().is_none_or(|tier| tier.minimize);

        let limit = time_limit.unwrap_or(5);
        let constraint_count = 1 + model.constraints.len() + model.globals.len();
        if verbose {
            println!("qayd solve (domain exact)");
            println!("  class: {}", selection.class.name());
            println!("  backend: {}", selection.backend.name());
            println!("  reason: {}", selection.reason);
            println!("  kind: lists");
            println!("  items: {}", model.items.len());
            println!("  lists: {}", model.lists);
            println!("  constraints: {constraint_count}");
            println!("  objective tiers: {}", model.objectives.len());
            println!("  time limit: {limit}s");
        }

        let stop = stop_after(limit);
        let solved = with_interrupts(py, &stop, || {
            list_exact_engine::solve(model, &objective_tiers, &stop, |candidate| {
                if verbose {
                    println!("  o {}  ({})", candidate[0], if minimize { "min" } else { "max" });
                }
            })
        })?;
        let outcome = match solved.map_err(PyValueError::new_err)? {
            Some(outcome) => outcome,
            None => return Ok(None),
        };

        let status = outcome.status.as_str();
        if verbose {
            println!("qayd result (domain exact)");
            println!("  status: {status}");
            if has_objective && !outcome.objectives.is_empty() {
                println!("  objectives: {:?}", outcome.objectives);
            }
            println!("  solutions: {}", outcome.stats.solutions);
            println!("  nodes: {}", outcome.stats.nodes);
            println!("  failures: {}", outcome.stats.failures);
        }

        let objective = outcome.objectives.first().copied();
        Ok(Some(PySolution {
            status: status.to_string(),
            objective,
            objective_sense: has_objective.then(|| if minimize { "min" } else { "max" }.to_string()),
            objective_expr: has_objective.then(|| "list objective".to_string()),
            values: Vec::new(),
            stats: outcome.stats.into(),
            lists: outcome.solution,
            objectives: outcome.objectives,
            starts: Vec::new(),
            presences: Vec::new(),
            machines: Vec::new(),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_solve_domain_schedule(
        &self,
        py: Python<'_>,
        schedule: &list::Schedule,
        model: &list::CollectionModel,
        selection: &shared_model::BackendSelection,
        time_limit: Option<u64>,
        seed: u64,
        verbose: bool,
    ) -> PyResult<Option<PySolution>> {
        if !model.objectives.is_empty() || !model.constraints.is_empty() || !model.globals.is_empty() || !model.items.is_empty() {
            return Ok(None);
        }

        let limit = time_limit.unwrap_or(5);
        let moded = schedule.intervals.iter().any(|iv| !iv.modes.is_empty());
        if verbose {
            println!("qayd solve (domain exact)");
            println!("  class: {}", selection.class.name());
            println!("  backend: {}", selection.backend.name());
            println!("  reason: {}", selection.reason);
            println!("  kind: {}", if moded { "intervals (machine choice)" } else { "intervals" });
            println!("  operations: {}", schedule.intervals.len());
            println!("  precedences: {}", schedule.precedences.len());
            println!("  resources: {}", schedule.resources.len());
            println!("  objective: {}", if schedule.minimize_makespan { "makespan" } else { "no" });
            println!("  time limit: {limit}s");
        }

        let stop = stop_after(limit);
        let options = schedule_engine::Options {
            seed,
            optional_modes_cdcl: schedule.minimize_makespan && std::env::var_os("QAYD_SCHEDULE_CDCL").is_some(),
        };
        let outcome = with_interrupts(py, &stop, || {
            schedule_engine::solve(schedule, &stop, options, |value| {
                if verbose {
                    println!("  o {value}  (min)");
                }
            })
        })?
        .map_err(PyValueError::new_err)?;
        let Some(outcome) = outcome else {
            return Ok(None);
        };

        if verbose {
            println!("qayd result (domain exact)");
            println!("  status: {}", outcome.status.as_str());
            if let Some(objective) = outcome.objective {
                println!("  objective: {objective}");
            }
            println!("  solutions: {}", outcome.stats.solutions);
            println!("  nodes: {}", outcome.stats.nodes);
            println!("  failures: {}", outcome.stats.failures);
        }

        Ok(Some(PySolution {
            status: outcome.status.as_str().to_string(),
            objective: outcome.objective,
            objective_sense: schedule.minimize_makespan.then(|| "min".to_string()),
            objective_expr: schedule.minimize_makespan.then(|| "makespan".to_string()),
            values: Vec::new(),
            stats: outcome.stats.into(),
            lists: None,
            objectives: outcome.objective.map(|value| vec![value]).unwrap_or_default(),
            starts: outcome.starts.into_iter().map(Some).collect(),
            presences: Vec::new(),
            machines: outcome.machines,
        }))
    }

    fn solve_collection(
        &self,
        py: Python<'_>,
        time_limit: Option<u64>,
        seed: u64,
        verbose: bool,
        engine: PythonEngine,
    ) -> PyResult<PySolution> {
        let model = list::CollectionModel {
            items: self.col_universe.clone().unwrap_or_default(),
            lists: self.col_lists,
            objectives: self.col_objectives.clone(),
            constraints: self.col_constraints.clone(),
            globals: self.col_globals.clone(),
            schedule: self.col_schedule.clone(),
        };
        model.validate().map_err(PyValueError::new_err)?;
        let shared = shared_model::Model::from_collection(&model);
        let selection = shared_model::BackendSelection::for_model(&shared);
        if engine != PythonEngine::Ls {
            if let Some(solution) = self.try_solve_routing_integer(py, &model, &selection, time_limit, seed, verbose)? {
                return Ok(solution);
            }
            if let Some(solution) = self.try_solve_domain_collection(py, &model, &selection, time_limit, seed, verbose)? {
                return Ok(solution);
            }
            if engine == PythonEngine::Exact {
                return Err(PyValueError::new_err(format!("model is not supported by an exact engine: {}", selection.reason)));
            }
        }
        if collection_has_max_terms(&model) {
            return Err(PyValueError::new_err("max_of list terms is supported by exact list backends only for this model shape"));
        }
        let limit = time_limit.unwrap_or(5);
        // The first tier drives the progress line; report its sense.
        let primary_sense = self.col_objectives.first().map_or("min", |t| if t.minimize { "min" } else { "max" });
        if verbose {
            println!("qayd solve (collection)");
            println!("  class: {}", selection.class.name());
            println!("  backend: {}", selection.backend.name());
            println!("  reason: {}", selection.reason);
            println!("  items: {}", model.items.len());
            println!("  lists: {}", model.lists);
            println!("  constraints: {}", model.constraints.len());
            println!("  objective tiers: {}", model.objectives.len());
            println!("  time limit: {limit}s");
        }
        let stop = stop_after(limit);
        let start = Instant::now();
        let mut improvements = 0u64;
        let mut report = |objective: i64| {
            if verbose {
                improvements += 1;
                println!("  o {objective}  ({primary_sense}, {:.2}s)", start.elapsed().as_secs_f64());
            }
        };
        let sol = with_interrupts(py, &stop, || lists::solve_collection(&model, seed, &stop, &mut report))?;
        if verbose {
            println!("qayd result (collection)");
            println!("  status: {}", if sol.feasible { "SATISFIABLE" } else { "UNKNOWN" });
            if sol.feasible {
                println!("  objectives: {:?}", sol.objectives);
            }
            println!("  improvements: {improvements}");
        }
        // Only expose objectives and lists for a feasible incumbent. An
        // infeasible best-effort partition may violate constraints, so hiding
        // it stops a caller from accidentally consuming an invalid solution.
        let objectives = if sol.feasible { sol.objectives.clone() } else { Vec::new() };
        Ok(PySolution {
            status: if sol.feasible { "SATISFIABLE".to_string() } else { "UNKNOWN".to_string() },
            objective: sol.feasible.then(|| sol.objectives.first().copied().unwrap_or(0)),
            objective_sense: Some(primary_sense.to_string()),
            objective_expr: None,
            values: Vec::new(),
            stats: SolveStats::default().into(),
            lists: sol.feasible.then(|| sol.lists.clone()),
            objectives,
            starts: if sol.feasible { sol.starts.iter().copied().map(Some).collect() } else { Vec::new() },
            presences: Vec::new(),
            machines: if sol.feasible { sol.machines.clone() } else { Vec::new() },
        })
    }

    /// Solve a COP with the local-search engine (`solve_ls`) - the same
    /// incumbent-only LS that powers `--ls`. Requires an objective and a time
    /// limit (defaults to 10s, since LS never terminates on its own).
    fn solve_ls(
        &self,
        py: Python<'_>,
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
        let stop = stop_after(time_limit.unwrap_or(10));
        let config = LsConfig { gls: true, min_conflicts: true, kick_bandit: false };
        let local = self.local.clone();
        let outcome = with_interrupts(py, &stop, || {
            solve_ls(problem, local, &stop, seed, config, |value, _solution, _source| {
                if verbose {
                    println!("  incumbent: {value}");
                }
            })
        })?;
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
            None => make_solution(
                "UNKNOWN",
                &vars,
                None,
                None,
                Some(sense),
                Some(&objective.expr.text),
                SolveStats::default(),
                self.names.len(),
            ),
        };
        if verbose {
            verbose_finish(&solution);
        }
        Ok(solution)
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_single_optimization(
        &self,
        py: Python<'_>,
        objective: &ObjectiveSpec,
        search: Option<&Bound<'_, PyAny>>,
        branch_order: &[VarId],
        verbose: bool,
        time_limit: Option<u64>,
        seed: u64,
    ) -> PyResult<PySolution> {
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
        for &var in branch_order {
            append_var(&mut vars, var);
        }
        append_expr_vars(&mut vars, &objective.expr.expr);
        let mut solver = self.solver.clone();
        append_interval_domain_vars(&mut vars, &solver);
        let stop = deadline(time_limit);
        let minimizing = objective.minimizing;
        let obj_expr = objective.expr.expr.clone();
        let (best, stats, complete) = with_interrupts(py, &stop, || {
            let search_objective = match &obj_expr {
                Expr::Var(var) => SearchObjective::Var(*var),
                expr => SearchObjective::Expr(expr),
            };
            search::optimize_seeded(
                &mut solver,
                &vars,
                search_objective,
                minimizing,
                &stop,
                seed,
                None,
                None,
                &[],
                None,
                Vec::new(),
                branch_order.to_vec(),
                |value, _| {
                    if verbose {
                        println!("  incumbent: {value}");
                    }
                },
            )
        })?;
        let Some((assignment, objective_value)) = best else {
            let mut solution = make_solution(
                if complete { "UNSATISFIABLE" } else { "UNKNOWN" },
                &vars,
                None,
                None,
                Some(if objective.minimizing { "min" } else { "max" }),
                Some(&objective.expr.text),
                stats,
                self.names.len(),
            );
            attach_native_interval_solution(&mut solution, &self.native_intervals);
            if verbose {
                verbose_finish(&solution);
            }
            return Ok(solution);
        };
        let mut solution = make_solution(
            if complete { "OPTIMAL" } else { "SATISFIABLE" },
            &vars,
            Some(&assignment),
            Some(objective_value),
            Some(if objective.minimizing { "min" } else { "max" }),
            Some(&objective.expr.text),
            stats,
            self.names.len(),
        );
        attach_native_interval_solution(&mut solution, &self.native_intervals);
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
    Min(Arc<PyNode>, Arc<PyNode>),
    Max(Arc<PyNode>, Arc<PyNode>),
    Div(Arc<PyNode>, Arc<PyNode>),
    Abs(Arc<PyNode>),
    Lt(Arc<PyNode>, Arc<PyNode>),
    Le(Arc<PyNode>, Arc<PyNode>),
    Eq(Arc<PyNode>, Arc<PyNode>),
    Ne(Arc<PyNode>, Arc<PyNode>),
    IfThenElse(Arc<PyNode>, Arc<PyNode>, Arc<PyNode>),
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
    Ok(PyTerm { model_id: a.model_id, gen: a.gen, reductions, max_terms })
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
    Ok(PyTerm { model_id: term.model_id, gen: term.gen, reductions, max_terms })
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
        groups.push(term.reductions);
    }
    Ok(PyTerm { model_id, gen, reductions: Vec::new(), max_terms: Some(vec![list::MaxTerm { groups, coeff: 1 }]) })
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

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<PyListConstraint> {
        if self.max_terms.as_ref().is_some_and(|terms| !terms.is_empty()) {
            return Err(PyValueError::new_err("max_of list terms are supported as objectives, not constraints"));
        }
        let rhs = other.extract::<i64>().map_err(|_| PyTypeError::new_err("a term can only be compared to an integer bound"))?;
        let op = match op {
            CompareOp::Le => list::Op::Le,
            CompareOp::Ge => list::Op::Ge,
            CompareOp::Eq => list::Op::Eq,
            _ => return Err(PyValueError::new_err("a term supports only <=, >=, ==")),
        };
        if self.reductions.len() != 1 {
            return Err(PyValueError::new_err("a constraint must be a single reduction over one list, not a sum of terms"));
        }
        Ok(PyListConstraint { model_id: self.model_id, gen: self.gen, reduction: self.reductions[0].clone(), op, rhs })
    }
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
    }
}

fn single_term(route: &PyListVar, reduction: list::Reduction) -> PyTerm {
    PyTerm { model_id: route.model_id, gen: route.gen, reductions: vec![reduction], max_terms: None }
}

/// Build a per-item reduction `op(route, i => body)` from a Python lambda.
fn build_items_reduction(route: &PyListVar, op: list::ReduceOp, func: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
    let body = coerce_node(&func.call1((node(PyNode::Arg(0)),))?)?;
    let mut arena = list::ExprArena::default();
    let body_id = lower(&body, &mut arena);
    Ok(single_term(route, list::Reduction { op, iterable: list::Iterable::Items(route.index as usize), arena, body: body_id, coeff: 1 }))
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
            Ok(single_term(
                route,
                list::Reduction { op: list::ReduceOp::Count, iterable: list::Iterable::Items(route.index as usize), arena, body, coeff: 1 },
            ))
        }
    }
}

/// `sum_edges(route, (i, j) => body, start=, end=)`: sum the body over the edges
/// of the closed tour `[start, items.., end]`.
#[pyfunction]
#[pyo3(signature = (route, func, *, start=0, end=0))]
fn sum_edges(route: &PyListVar, func: &Bound<'_, PyAny>, start: i32, end: i32) -> PyResult<PyTerm> {
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
    let body = coerce_node(&func.call1((node(PyNode::Arg(0)), node(PyNode::Arg(1))))?)?;
    Ok(pairs_term(route, body))
}

/// `pos_pairs(route, (a, b, i, j) => body)`: sum the body over every ordered
/// pair of positions, with the items `a`/`b` at positions `i`/`j`. Use for
/// quadratic objectives (QAP), e.g.
/// `pos_pairs(p, lambda a, b, i, j: A[i][j] * B[a][b])`.
#[pyfunction]
fn pos_pairs(route: &PyListVar, func: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
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

/// `scan_sum(route, step, emit, init=, boundary=)`: fold an accumulator along
/// the route and sum a per-step value. `step(cur, acc, prev) -> new_acc` and
/// `emit(cur, acc, prev) -> value` (where `acc` in `emit` is the new
/// accumulator, `prev` is the previous item or `boundary` at the first step).
/// Used for cumulative time/load, e.g. time-window lateness.
#[pyfunction]
#[pyo3(signature = (route, step, emit, *, init=0, boundary=0))]
fn scan_sum(route: &PyListVar, step: &Bound<'_, PyAny>, emit: &Bound<'_, PyAny>, init: i64, boundary: i32) -> PyResult<PyTerm> {
    let step_body = coerce_node(&step.call1((node(PyNode::Arg(0)), node(PyNode::Arg(1)), node(PyNode::Arg(2))))?)?;
    let emit_body = coerce_node(&emit.call1((node(PyNode::Arg(0)), node(PyNode::Arg(1)), node(PyNode::Arg(2))))?)?;
    let mut arena = list::ExprArena::default();
    let step_id = lower(&step_body, &mut arena);
    let emit_id = lower(&emit_body, &mut arena);
    let iterable = list::Iterable::Scan { list: route.index as usize, init, boundary, step: step_id };
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
    let step_body = coerce_node(&step.call1((node(PyNode::Arg(0)), node(PyNode::Arg(1)), node(PyNode::Arg(2))))?)?;
    let emit_body = coerce_node(&emit.call1((node(PyNode::Arg(0)), node(PyNode::Arg(1)), node(PyNode::Arg(2))))?)?;
    let mut arena = list::ExprArena::default();
    let step_id = lower(&step_body, &mut arena);
    let emit_id = lower(&emit_body, &mut arena);
    let iterable = list::Iterable::Scan { list: route.index as usize, init, boundary, step: step_id };
    Ok(single_term(route, list::Reduction { op: list::ReduceOp::SelectKth(k), iterable, arena, body: emit_id, coeff: 1 }))
}

/// `windows(route, size, inner, emit)`: for each window of `size` consecutive
/// items, sum `inner(item)` to a window total, then sum `emit(total)` over
/// windows. Used for sliding-window counts, e.g. car-sequencing option limits:
/// `windows(seq, q, inner=lambda c: opt[c], emit=lambda t: cp.max(0, t - p))`.
#[pyfunction]
fn windows(route: &PyListVar, size: usize, inner: &Bound<'_, PyAny>, emit: &Bound<'_, PyAny>) -> PyResult<PyTerm> {
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
    single_term(
        route,
        list::Reduction { op: list::ReduceOp::Used, iterable: list::Iterable::Items(route.index as usize), arena, body, coeff: 1 },
    )
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
    m.add_function(wrap_pyfunction!(scan_sum, m)?)?;
    m.add_function(wrap_pyfunction!(select_kth, m)?)?;
    m.add_function(wrap_pyfunction!(windows, m)?)?;
    m.add_function(wrap_pyfunction!(used, m)?)?;
    m.add_function(wrap_pyfunction!(_rust_panic, m)?)?;
    m.add("STAR", table::STAR)?;
    Ok(())
}
