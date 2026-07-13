//! Optional LP relaxation backed by `amthal`.
//!
//! Amthal supplies floating-point primal and dual candidates. Value phases may
//! use the primal candidate directly. Pruning is permitted only after Qayd has
//! reconstructed a Lagrangian lower bound in exact rational arithmetic from the
//! dual candidate and the current integer bounds.

use std::sync::atomic::AtomicBool;

use crate::ids::VarId;
use crate::search::Objective;
use crate::store::Solver;

#[derive(Clone, Debug, Default)]
pub struct AdvisoryLpHint {
    pub phase: Vec<Option<i32>>,
    pub objective: Option<f64>,
    pub certified_bound: Option<i64>,
    pub rows: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LpRuntimeStats {
    pub rows: u64,
    pub solves: u64,
    pub certified: u64,
    pub prunes: u64,
    pub node_prunes: u64,
    pub global_prunes: u64,
    pub timeouts: u64,
    pub skipped: u64,
    pub refactorizations: u64,
    pub micros: u64,
    pub root_bound: Option<i64>,
}

#[derive(Default)]
pub(crate) struct LpRootResult {
    pub phase: Vec<Option<i32>>,
}

#[cfg_attr(not(feature = "lp-relaxation"), allow(dead_code))]
pub(crate) enum NodeLpResult {
    NotRun,
    Continue(Vec<Option<i32>>),
    Prune(Vec<(VarId, i32, i32)>),
}

#[cfg(not(feature = "lp-relaxation"))]
pub(crate) struct IncrementalLp;

#[cfg(not(feature = "lp-relaxation"))]
impl IncrementalLp {
    pub(crate) fn new(_solver: &Solver, _objective: Objective<'_>, _minimizing: bool, _enabled: bool, _stop: &AtomicBool) -> Self {
        Self
    }

    pub(crate) fn solve_root(&mut self, _solver: &Solver, _search: &[VarId], _stop: &AtomicBool) -> LpRootResult {
        LpRootResult::default()
    }

    pub(crate) fn check_node(&mut self, _solver: &Solver, _incumbent: i64, _node: u64, _at_root: bool, _stop: &AtomicBool) -> NodeLpResult {
        NodeLpResult::NotRun
    }

    pub(crate) fn stats(&self) -> LpRuntimeStats {
        LpRuntimeStats::default()
    }

    pub(crate) fn certifies_global(&mut self, _incumbent: i64) -> bool {
        false
    }
}

#[cfg(not(feature = "lp-relaxation"))]
pub(crate) fn hint_for_direction(
    _solver: &Solver,
    _search: &[VarId],
    _objective: Objective<'_>,
    _minimizing: bool,
    _stop: &AtomicBool,
) -> AdvisoryLpHint {
    AdvisoryLpHint::default()
}

#[cfg(feature = "lp-relaxation")]
mod enabled {
    use std::collections::{BTreeMap, HashSet};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    use amthal::{LpForm, LpSession, LpSolution, LpStatus, Model, Sense};
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::{Signed, ToPrimitive};

    use super::{AdvisoryLpHint, LpRootResult, LpRuntimeStats, NodeLpResult};
    use crate::ids::VarId;
    use crate::search::Objective;
    use crate::store::Solver;

    const MAX_EXACT_F64_INT: u64 = 1u64 << 53;

    struct Config {
        max_vars: usize,
        max_rows: usize,
        max_nonzeros: usize,
        min_coverage_percent: usize,
        min_gain: i64,
        phase_max_vars: usize,
        root_budget: Duration,
        node_budget: Duration,
        total_budget: Duration,
        node_every: u64,
    }

    impl Config {
        fn read() -> Self {
            Self {
                max_vars: env_usize("QAYD_LP_MAX_VARS", 2_000),
                max_rows: env_usize("QAYD_LP_MAX_ROWS", 1_000),
                max_nonzeros: env_usize("QAYD_LP_MAX_NONZEROS", 100_000),
                min_coverage_percent: env_usize("QAYD_LP_MIN_COVERAGE", 1).min(100),
                min_gain: env_i64("QAYD_LP_MIN_GAIN", 2).max(0),
                phase_max_vars: env_usize("QAYD_LP_PHASE_MAX_VARS", 1_000),
                root_budget: Duration::from_millis(env_u64("QAYD_LP_ROOT_MS", 50)),
                node_budget: Duration::from_millis(env_u64("QAYD_LP_NODE_MS", 1)),
                total_budget: Duration::from_millis(env_u64("QAYD_LP_TOTAL_MS", 100)),
                node_every: env_u64("QAYD_LP_NODE_EVERY", 64).max(1),
            }
        }
    }

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
    }

    fn env_u64(name: &str, default: u64) -> u64 {
        std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
    }

    fn env_i64(name: &str, default: i64) -> i64 {
        std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
    }

    #[derive(Clone)]
    struct ExactRow {
        terms: Vec<(usize, i128)>,
        lower: Option<i128>,
        upper: Option<i128>,
    }

    struct ProblemData {
        objective_constant: i128,
        objective: Vec<i128>,
        rows: Vec<ExactRow>,
        form: LpForm,
        root_bounds: Vec<(i32, i32)>,
        objective_var: Option<usize>,
    }

    impl ProblemData {
        fn build(solver: &Solver, objective: Objective<'_>, minimizing: bool, config: &Config, stop: &AtomicBool) -> Option<Self> {
            let num_vars = solver.store.num_vars();
            let source_rows = solver.linear_relaxation();
            if stop.load(Ordering::Relaxed)
                || num_vars == 0
                || num_vars > config.max_vars
                || source_rows.is_empty()
                || source_rows.len() > config.max_rows
            {
                return None;
            }

            let scale = if minimizing { 1i128 } else { -1i128 };
            let objective_var = match objective {
                Objective::Var(var) => Some(var.index()),
                Objective::Linear { .. } | Objective::Expr(_) => None,
            };
            let (constant, terms) = match objective {
                Objective::Var(var) => (0, vec![(var, 1i128)]),
                Objective::Linear { coeffs, vars } => {
                    if coeffs.len() != vars.len() {
                        return None;
                    }
                    (0, vars.iter().copied().zip(coeffs.iter().map(|&coefficient| i128::from(coefficient))).collect())
                }
                Objective::Expr(expr) => {
                    let form = expr.linear_form()?;
                    (form.constant, form.terms)
                }
            };
            let objective_constant = constant.checked_mul(scale)?;
            let mut objective_coefficients = vec![0i128; num_vars];
            for (var, coefficient) in terms {
                let scaled = coefficient.checked_mul(scale)?;
                objective_coefficients[var.index()] = objective_coefficients[var.index()].checked_add(scaled)?;
            }
            if !exact_i128(objective_constant) || objective_coefficients.iter().any(|&coefficient| !exact_i128(coefficient)) {
                return None;
            }

            let mut rows = Vec::with_capacity(source_rows.len());
            let mut nonzeros = 0usize;
            let mut covered = HashSet::new();
            for row in source_rows {
                if stop.load(Ordering::Relaxed) {
                    return None;
                }
                let mut normalized = BTreeMap::<usize, i128>::new();
                for (&coefficient, &var) in row.coeffs.iter().zip(&row.vars) {
                    let entry = normalized.entry(var.index()).or_default();
                    *entry = entry.checked_add(i128::from(coefficient))?;
                }
                normalized.retain(|_, coefficient| *coefficient != 0);
                if normalized.values().any(|&coefficient| !exact_i128(coefficient))
                    || row.lower.is_some_and(|bound| !exact_i128(i128::from(bound)))
                    || row.upper.is_some_and(|bound| !exact_i128(i128::from(bound)))
                {
                    return None;
                }
                nonzeros = nonzeros.checked_add(normalized.len())?;
                if nonzeros > config.max_nonzeros {
                    return None;
                }
                covered.extend(normalized.keys().copied());
                rows.push(ExactRow {
                    terms: normalized.into_iter().collect(),
                    lower: row.lower.map(i128::from),
                    upper: row.upper.map(i128::from),
                });
            }
            if covered.len().saturating_mul(100) < num_vars.saturating_mul(config.min_coverage_percent) {
                return None;
            }

            let root_bounds: Vec<_> = (0..num_vars)
                .map(|index| {
                    let var = VarId(index as u32);
                    (solver.store.min(var), solver.store.max(var))
                })
                .collect();
            let mut model = Model::new(Sense::Minimize);
            model.add_obj_offset(objective_constant as f64);
            let amthal_vars: Vec<_> = root_bounds
                .iter()
                .enumerate()
                .map(|(index, &(lower, upper))| {
                    model.add_continuous(f64::from(lower), f64::from(upper), objective_coefficients[index] as f64)
                })
                .collect();
            for row in &rows {
                let terms: Vec<_> = row.terms.iter().map(|&(index, coefficient)| (amthal_vars[index], coefficient as f64)).collect();
                model.add_constraint(
                    row.lower.map_or(f64::NEG_INFINITY, |bound| bound as f64),
                    row.upper.map_or(f64::INFINITY, |bound| bound as f64),
                    &terms,
                );
            }
            Some(Self {
                objective_constant,
                objective: objective_coefficients,
                rows,
                form: LpForm::from_model(&model),
                root_bounds,
                objective_var,
            })
        }

        fn current_bounds(&self, solver: &Solver) -> Vec<(i32, i32)> {
            let mut bounds = current_bounds(solver);
            if let Some(index) = self.objective_var {
                bounds[index] = self.root_bounds[index];
            }
            bounds
        }

        fn box_bound_ceil(&self, bounds: &[(i32, i32)]) -> i64 {
            let mut value = BigInt::from(self.objective_constant);
            for (&coefficient, &(lower, upper)) in self.objective.iter().zip(bounds) {
                let endpoint = if coefficient >= 0 { lower } else { upper };
                value += BigInt::from(coefficient) * BigInt::from(endpoint);
            }
            clamp_bigint(&value)
        }

        fn certify(&self, duals: &[f64], bounds: &[(i32, i32)]) -> Option<i64> {
            if duals.len() != self.rows.len() || bounds.len() != self.objective.len() {
                return None;
            }
            // Amthal solves [A | -I] z = 0. For any finite multiplier y,
            // c'z = (c - A'y)'x + y's on feasible points. Minimizing that
            // identity independently over the exact x and row-slack bounds is
            // a valid lower bound even when y is not dual feasible. Converting
            // every f64 to its exact dyadic rational makes the final ceil a
            // proof object independent of simplex tolerances.
            let mut residual: Vec<BigRational> =
                self.objective.iter().map(|&coefficient| BigRational::from_integer(BigInt::from(coefficient))).collect();
            let mut bound = BigRational::from_integer(BigInt::from(self.objective_constant));
            for (row, &candidate) in self.rows.iter().zip(duals) {
                if !candidate.is_finite() {
                    return None;
                }
                let y = BigRational::from_float(candidate)?;
                if y.is_positive() {
                    bound += &y * BigInt::from(row.lower?);
                } else if y.is_negative() {
                    bound += &y * BigInt::from(row.upper?);
                }
                for &(index, coefficient) in &row.terms {
                    residual[index] -= &y * BigInt::from(coefficient);
                }
            }
            for (coefficient, &(lower, upper)) in residual.into_iter().zip(bounds) {
                let endpoint = if coefficient.is_negative() { upper } else { lower };
                bound += coefficient * BigInt::from(endpoint);
            }
            let ceil = bound.ceil().to_integer();
            Some(clamp_bigint(&ceil))
        }
    }

    fn exact_i128(value: i128) -> bool {
        value.unsigned_abs() <= u128::from(MAX_EXACT_F64_INT)
    }

    fn clamp_bigint(value: &BigInt) -> i64 {
        value.to_i64().unwrap_or_else(|| if value.is_negative() { i64::MIN } else { i64::MAX })
    }

    fn current_bounds(solver: &Solver) -> Vec<(i32, i32)> {
        (0..solver.store.num_vars())
            .map(|index| {
                let var = VarId(index as u32);
                (solver.store.min(var), solver.store.max(var))
            })
            .collect()
    }

    fn phase_from_solution(
        data: &ProblemData,
        solver: &Solver,
        search: &[VarId],
        solution: &LpSolution,
        stop: &AtomicBool,
        max_vars: usize,
    ) -> Vec<Option<i32>> {
        if solution.x.len() != solver.store.num_vars() || solution.x.len() > max_vars || stop.load(Ordering::Relaxed) {
            return Vec::new();
        }
        let mut rounded = Vec::with_capacity(solution.x.len());
        for (index, &value) in solution.x.iter().enumerate() {
            if !value.is_finite() {
                return Vec::new();
            }
            let var = VarId(index as u32);
            let candidate = value.round().clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32;
            rounded.push(if solver.store.contains(var, candidate) {
                candidate
            } else if (value - f64::from(solver.store.min(var))).abs() <= (value - f64::from(solver.store.max(var))).abs() {
                solver.store.min(var)
            } else {
                solver.store.max(var)
            });
        }
        for row in &data.rows {
            let Some(activity) = row
                .terms
                .iter()
                .try_fold(0i128, |sum, &(index, coefficient)| sum.checked_add(coefficient.checked_mul(i128::from(rounded[index]))?))
            else {
                return Vec::new();
            };
            if row.lower.is_some_and(|lower| activity < lower) || row.upper.is_some_and(|upper| activity > upper) {
                return Vec::new();
            }
        }
        let mut checker = solver.clone();
        for (index, &value) in rounded.iter().enumerate() {
            if stop.load(Ordering::Relaxed) || checker.store.fix(VarId(index as u32), value).is_err() {
                return Vec::new();
            }
        }
        if checker.propagate_until(|| stop.load(Ordering::Relaxed)).is_err() || stop.load(Ordering::Relaxed) {
            return Vec::new();
        }
        let mut phase = vec![None; solver.store.num_vars()];
        for &var in search {
            phase[var.index()] = Some(rounded[var.index()]);
        }
        phase
    }

    pub(crate) struct IncrementalLp {
        config: Config,
        data: Option<ProblemData>,
        session: Option<LpSession>,
        last_bounds: Vec<(i32, i32)>,
        last_certificate: Option<i64>,
        global_certificate: Option<i64>,
        quality: bool,
        minimizing: bool,
        last_objective: Option<f64>,
        spent: Duration,
        stats: LpRuntimeStats,
    }

    impl IncrementalLp {
        pub(crate) fn new(solver: &Solver, objective: Objective<'_>, minimizing: bool, enabled: bool, stop: &AtomicBool) -> Self {
            let config = Config::read();
            let data = enabled.then(|| ProblemData::build(solver, objective, minimizing, &config, stop)).flatten();
            let session = data.as_ref().map(|data| LpSession::new(data.form.clone()));
            let rows = data.as_ref().map_or(0, |data| data.rows.len() as u64);
            Self {
                config,
                data,
                session,
                last_bounds: Vec::new(),
                last_certificate: None,
                global_certificate: None,
                quality: false,
                minimizing,
                last_objective: None,
                spent: Duration::ZERO,
                stats: LpRuntimeStats { rows, skipped: u64::from(rows == 0), ..LpRuntimeStats::default() },
            }
        }

        pub(crate) fn solve_root(&mut self, solver: &Solver, search: &[VarId], stop: &AtomicBool) -> LpRootResult {
            if self.data.is_none() {
                return LpRootResult::default();
            }
            let bounds = self.data.as_ref().expect("LP data checked above").current_bounds(solver);
            let root_budget = self.config.root_budget;
            let Some(solution) = self.solve(&bounds, root_budget, stop) else {
                return LpRootResult::default();
            };
            if solution.status != LpStatus::Optimal {
                return LpRootResult::default();
            }
            self.last_objective = Some(if self.minimizing { solution.obj } else { -solution.obj });
            let data = self.data.as_ref().expect("LP data checked above");
            let certificate = data.certify(&solution.duals, &bounds);
            if let Some(certificate) = certificate {
                self.stats.certified += 1;
                self.stats.root_bound = Some(if self.minimizing { certificate } else { certificate.saturating_neg() });
                let box_bound = data.box_bound_ceil(&bounds);
                self.quality = certificate.saturating_sub(box_bound) >= self.config.min_gain;
                self.last_certificate = Some(certificate);
                self.global_certificate = Some(certificate);
            }
            self.last_bounds = bounds;
            LpRootResult { phase: phase_from_solution(data, solver, search, &solution, stop, self.config.phase_max_vars) }
        }

        pub(crate) fn check_node(&mut self, solver: &Solver, incumbent: i64, node: u64, at_root: bool, stop: &AtomicBool) -> NodeLpResult {
            if !self.quality || stop.load(Ordering::Relaxed) {
                return NodeLpResult::NotRun;
            }
            let normalized_incumbent = if self.minimizing { incumbent } else { incumbent.saturating_neg() };
            let Some(data) = self.data.as_ref() else {
                return NodeLpResult::NotRun;
            };
            let bounds = data.current_bounds(solver);
            let within_last = bounds.len() == self.last_bounds.len()
                && bounds
                    .iter()
                    .zip(&self.last_bounds)
                    .all(|(&(lower, upper), &(old_lower, old_upper))| lower >= old_lower && upper <= old_upper);
            if within_last && self.last_certificate.is_some_and(|certificate| certificate >= normalized_incumbent) {
                let certified_bounds = self.last_bounds.clone();
                self.stats.prunes += 1;
                self.stats.node_prunes += 1;
                return NodeLpResult::Prune(self.changed_bounds(&certified_bounds));
            }
            if bounds == self.last_bounds {
                return NodeLpResult::NotRun;
            }
            if !at_root && !node.is_multiple_of(self.config.node_every) {
                return NodeLpResult::NotRun;
            }
            let Some(solution) = self.solve(&bounds, self.config.node_budget, stop) else {
                return NodeLpResult::NotRun;
            };
            self.last_bounds = bounds.clone();
            self.last_certificate = None;
            if solution.status != LpStatus::Optimal {
                return NodeLpResult::NotRun;
            }
            let Some(data) = self.data.as_ref() else {
                return NodeLpResult::NotRun;
            };
            let Some(certificate) = data.certify(&solution.duals, &bounds) else {
                return NodeLpResult::NotRun;
            };
            self.stats.certified += 1;
            self.last_certificate = Some(certificate);
            if at_root {
                self.global_certificate = Some(self.global_certificate.map_or(certificate, |old| old.max(certificate)));
            }
            if certificate >= normalized_incumbent {
                self.stats.prunes += 1;
                self.stats.node_prunes += 1;
                NodeLpResult::Prune(self.changed_bounds(&bounds))
            } else {
                NodeLpResult::Continue(Vec::new())
            }
        }

        pub(crate) fn stats(&self) -> LpRuntimeStats {
            let mut stats = self.stats;
            if let Some(session) = &self.session {
                stats.refactorizations = session.refactorizations() as u64;
            }
            stats
        }

        pub(crate) fn certifies_global(&mut self, incumbent: i64) -> bool {
            let normalized = if self.minimizing { incumbent } else { incumbent.saturating_neg() };
            let certified = self.global_certificate.is_some_and(|bound| bound >= normalized);
            if certified {
                self.stats.prunes += 1;
                self.stats.global_prunes += 1;
            }
            certified
        }

        fn changed_bounds(&self, bounds: &[(i32, i32)]) -> Vec<(VarId, i32, i32)> {
            let Some(data) = self.data.as_ref() else {
                return Vec::new();
            };
            bounds
                .iter()
                .zip(&data.root_bounds)
                .enumerate()
                .filter_map(|(index, (&current, &root))| (current != root).then_some((VarId(index as u32), current.0, current.1)))
                .collect()
        }

        fn solve(&mut self, bounds: &[(i32, i32)], requested: Duration, stop: &AtomicBool) -> Option<LpSolution> {
            if stop.load(Ordering::Relaxed) || self.spent >= self.config.total_budget {
                self.stats.skipped += 1;
                return None;
            }
            let budget = requested.min(self.config.total_budget - self.spent);
            if budget.is_zero() {
                self.stats.skipped += 1;
                return None;
            }
            let session = self.session.as_mut()?;
            for (index, &(lower, upper)) in bounds.iter().enumerate() {
                if stop.load(Ordering::Relaxed) {
                    self.stats.skipped += 1;
                    return None;
                }
                if session.lower(index) != f64::from(lower) || session.upper(index) != f64::from(upper) {
                    session.set_bounds(index, f64::from(lower), f64::from(upper));
                }
            }
            let start = Instant::now();
            session.deadline = Some(start + budget);
            let solution = session.solve();
            session.deadline = None;
            let elapsed = start.elapsed();
            self.spent += elapsed;
            self.stats.solves += 1;
            self.stats.micros = self.stats.micros.saturating_add(elapsed.as_micros().min(u128::from(u64::MAX)) as u64);
            if solution.status == LpStatus::TimeLimit {
                self.stats.timeouts += 1;
            }
            Some(solution)
        }
    }

    pub(crate) fn hint_for_direction(
        solver: &Solver,
        search: &[VarId],
        objective: Objective<'_>,
        minimizing: bool,
        stop: &AtomicBool,
    ) -> AdvisoryLpHint {
        let mut lp = IncrementalLp::new(solver, objective, minimizing, true, stop);
        let root = lp.solve_root(solver, search, stop);
        let stats = lp.stats();
        AdvisoryLpHint { phase: root.phase, objective: lp.last_objective, certified_bound: stats.root_bound, rows: stats.rows as usize }
    }
}

#[cfg(feature = "lp-relaxation")]
pub(crate) use enabled::{hint_for_direction, IncrementalLp};

/// Public diagnostic entry point for a materialized objective variable. The
/// returned bound is exact when `certified_bound` is present.
pub fn advisory_var_hint(solver: &Solver, search: &[VarId], objective: VarId, minimizing: bool) -> AdvisoryLpHint {
    static NEVER_STOP: AtomicBool = AtomicBool::new(false);
    hint_for_direction(solver, search, Objective::Var(objective), minimizing, &NEVER_STOP)
}
