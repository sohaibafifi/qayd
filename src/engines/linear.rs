//! Backend-neutral linear relaxation for root guidance and exact search pruning.
//!
//! A backend may suggest primal values and row multipliers in floating point.
//! Primal values are only search guidance. A bound is exposed only after qayd
//! reconstructs the corresponding Lagrangian bound in exact rational
//! arithmetic from the original integer coefficients and domains.

#[cfg(not(feature = "lp-relaxation"))]
use std::sync::atomic::AtomicBool;

#[cfg(not(feature = "lp-relaxation"))]
use crate::ids::VarId;
#[cfg(not(feature = "lp-relaxation"))]
use crate::orchestrator::LinearControls;
#[cfg(not(feature = "lp-relaxation"))]
use crate::search::Objective;
use crate::search::SolveStats;
#[cfg(not(feature = "lp-relaxation"))]
use crate::store::Solver;

#[derive(Default)]
pub(crate) struct RootRelaxation {
    pub(crate) phase: Vec<Option<i32>>,
    pub(crate) bound: Option<i64>,
    pub(crate) backend: Option<&'static str>,
    pub(crate) stats: SolveStats,
    pub(crate) search: Option<SearchRelaxationTemplate>,
}

#[cfg(not(feature = "lp-relaxation"))]
#[derive(Clone)]
pub(crate) struct SearchRelaxationTemplate;

#[cfg(not(feature = "lp-relaxation"))]
pub(crate) struct SearchRelaxation;

#[cfg(not(feature = "lp-relaxation"))]
#[allow(dead_code)]
pub(crate) enum SearchCheck {
    Continue,
    Prune(Vec<crate::store::Premise>),
}

#[cfg(not(feature = "lp-relaxation"))]
impl SearchRelaxationTemplate {
    pub(crate) fn start(&self) -> SearchRelaxation {
        SearchRelaxation
    }
}

#[cfg(not(feature = "lp-relaxation"))]
impl SearchRelaxation {
    pub(crate) fn check(&mut self, _store: &crate::store::Store, _incumbent: Option<i64>) -> SearchCheck {
        SearchCheck::Continue
    }

    pub(crate) fn record_prune(&mut self) {}

    pub(crate) fn stats(&self) -> SolveStats {
        SolveStats::default()
    }
}

#[cfg(not(feature = "lp-relaxation"))]
pub(crate) fn solve_root(
    _solver: &Solver,
    _search: &[VarId],
    _objective: Objective<'_>,
    _minimizing: bool,
    _controls: LinearControls,
    _stop: &AtomicBool,
) -> RootRelaxation {
    RootRelaxation::default()
}

#[cfg(feature = "lp-relaxation")]
mod enabled {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use amthal::{DualSolver, LpForm, LpSession, LpSolver, LpStatus, Model, Sense};
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::{Signed, ToPrimitive, Zero};

    use super::RootRelaxation;
    use crate::ids::VarId;
    use crate::orchestrator::{LinearBackendMode, LinearControls};
    use crate::search::{Objective, SolveStats};
    use crate::store::{Premise, Solver, Store};

    const MAX_EXACT_F64_INTEGER: u64 = 1u64 << 53;

    #[derive(Clone)]
    struct LinearRow {
        terms: Vec<(usize, i128)>,
        lower: Option<i128>,
        upper: Option<i128>,
    }

    /// Exact backend-neutral model. Backends receive floating projections of
    /// this data, while certification always uses these original integers.
    struct LinearModel {
        objective_constant: i128,
        objective: Vec<i128>,
        rows: Vec<LinearRow>,
        bounds: Vec<(i32, i32)>,
    }

    #[derive(Clone)]
    struct CertifiedLpBound {
        value: BigInt,
        premises: Vec<Premise>,
    }

    #[derive(Clone)]
    pub(crate) struct SearchRelaxationTemplate {
        model: Arc<LinearModel>,
        form: Arc<LpForm>,
        minimizing: bool,
        root_primal: Arc<[f64]>,
        root_certificate: Option<CertifiedLpBound>,
        node_time: Duration,
        depth_interval: usize,
    }

    pub(crate) struct SearchRelaxation {
        template: SearchRelaxationTemplate,
        session: LpSession,
        solved_bounds: Vec<(i32, i32)>,
        primal: Vec<f64>,
        certificate: Option<CertifiedLpBound>,
        next_depth: usize,
        last_attempt_depth: usize,
        refactorizations: usize,
        stats: SolveStats,
    }

    pub(crate) enum SearchCheck {
        Continue,
        Prune(Vec<Premise>),
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum LinearStatus {
        Optimal,
        TimeLimit,
        Other,
    }

    struct LinearSolution {
        status: LinearStatus,
        primal: Vec<f64>,
        row_duals: Vec<f64>,
        refactorizations: u64,
    }

    /// Small internal boundary shared by Amthal today and HiGHS later.
    trait LinearBackend {
        fn name(&self) -> &'static str;
        fn solve(&self, model: &LinearModel, time_limit: Duration, stop: &AtomicBool) -> Option<LinearSolution>;
    }

    struct AmthalBackend;

    impl LinearBackend for AmthalBackend {
        fn name(&self) -> &'static str {
            "amthal"
        }

        fn solve(&self, model: &LinearModel, time_limit: Duration, stop: &AtomicBool) -> Option<LinearSolution> {
            if time_limit.is_zero() || stop.load(Ordering::Acquire) {
                return None;
            }
            let form = amthal_form(model);
            let mut solver = DualSolver { deadline: Some(Instant::now() + time_limit), ..DualSolver::default() };
            let solution = solver.solve(&form, None);
            let status = match solution.status {
                LpStatus::Optimal => LinearStatus::Optimal,
                LpStatus::TimeLimit => LinearStatus::TimeLimit,
                LpStatus::Infeasible | LpStatus::Unbounded | LpStatus::IterationLimit | LpStatus::NumericalError => LinearStatus::Other,
            };
            Some(LinearSolution { status, primal: solution.x, row_duals: solution.duals, refactorizations: 0 })
        }
    }

    fn amthal_form(model: &LinearModel) -> LpForm {
        let mut source = Model::new(Sense::Minimize);
        source.add_obj_offset(model.objective_constant as f64);
        let variables: Vec<_> = model
            .bounds
            .iter()
            .zip(&model.objective)
            .map(|(&(lower, upper), &coefficient)| source.add_continuous(f64::from(lower), f64::from(upper), coefficient as f64))
            .collect();
        for row in &model.rows {
            let terms: Vec<_> = row.terms.iter().map(|&(index, coefficient)| (variables[index], coefficient as f64)).collect();
            source.add_constraint(
                row.lower.map_or(f64::NEG_INFINITY, |bound| bound as f64),
                row.upper.map_or(f64::INFINITY, |bound| bound as f64),
                &terms,
            );
        }
        LpForm::from_model(&source)
    }

    impl LinearModel {
        fn build(solver: &Solver, objective: Objective<'_>, minimizing: bool, controls: LinearControls, stop: &AtomicBool) -> Option<Self> {
            let variable_count = solver.store.num_vars();
            let source_rows = solver.linear_relaxation();
            if stop.load(Ordering::Acquire)
                || variable_count == 0
                || variable_count > controls.max_variables
                || source_rows.is_empty()
                || source_rows.len() > controls.max_rows
            {
                return None;
            }

            let scale = if minimizing { 1i128 } else { -1i128 };
            let (constant, coefficients, variables) = objective_affine(objective, stop)?;
            let objective_constant = i128::from(constant).checked_mul(scale)?;
            let mut objective_coefficients = vec![0i128; variable_count];
            for (coefficient, variable) in coefficients.into_iter().zip(variables) {
                let scaled = i128::from(coefficient).checked_mul(scale)?;
                let slot = objective_coefficients.get_mut(variable.index())?;
                *slot = slot.checked_add(scaled)?;
            }
            if !exact_as_f64(objective_constant) || objective_coefficients.iter().any(|&coefficient| !exact_as_f64(coefficient)) {
                return None;
            }

            let mut rows = Vec::with_capacity(source_rows.len());
            let mut covered = BTreeSet::new();
            let mut nonzeros = 0usize;
            for source in source_rows {
                if stop.load(Ordering::Acquire) || source.coefficients.len() != source.variables.len() {
                    return None;
                }
                let mut normalized = BTreeMap::<usize, i128>::new();
                for (&coefficient, &variable) in source.coefficients.iter().zip(&source.variables) {
                    let slot = normalized.entry(variable.index()).or_default();
                    *slot = slot.checked_add(i128::from(coefficient))?;
                }
                normalized.retain(|_, coefficient| *coefficient != 0);
                if normalized.keys().any(|&index| index >= variable_count)
                    || normalized.values().any(|&coefficient| !exact_as_f64(coefficient))
                    || source.lower.is_some_and(|bound| !exact_as_f64(i128::from(bound)))
                    || source.upper.is_some_and(|bound| !exact_as_f64(i128::from(bound)))
                {
                    return None;
                }
                nonzeros = nonzeros.checked_add(normalized.len())?;
                if nonzeros > controls.max_nonzeros {
                    return None;
                }
                covered.extend(normalized.keys().copied());
                rows.push(LinearRow {
                    terms: normalized.into_iter().collect(),
                    lower: source.lower.map(i128::from),
                    upper: source.upper.map(i128::from),
                });
            }
            if covered.len().saturating_mul(100) < variable_count.saturating_mul(controls.min_coverage_percent) {
                return None;
            }
            let bounds = (0..variable_count)
                .map(|index| {
                    let variable = VarId(u32::try_from(index).ok()?);
                    Some((solver.store.min(variable), solver.store.max(variable)))
                })
                .collect::<Option<Vec<_>>>()?;
            Some(Self { objective_constant, objective: objective_coefficients, rows, bounds })
        }

        fn certify(&self, row_duals: &[f64], bounds: &[(i32, i32)]) -> Option<CertifiedLpBound> {
            if row_duals.len() != self.rows.len() || bounds.len() != self.objective.len() {
                return None;
            }
            // Amthal solves [A | -I] z = 0. For every finite multiplier y,
            // c'x = (c - A'y)'x + y's on feasible points. Minimizing each
            // residual term over the exact variable and row-activity boxes is
            // therefore a valid lower bound even if y is not dual feasible.
            let mut residual: Vec<BigRational> =
                self.objective.iter().map(|&coefficient| BigRational::from_integer(BigInt::from(coefficient))).collect();
            let mut bound = BigRational::from_integer(BigInt::from(self.objective_constant));
            for (row, &candidate) in self.rows.iter().zip(row_duals) {
                if !candidate.is_finite() {
                    return None;
                }
                let multiplier = BigRational::from_float(candidate)?;
                if multiplier.is_positive() {
                    bound += &multiplier * BigInt::from(row.lower?);
                } else if multiplier.is_negative() {
                    bound += &multiplier * BigInt::from(row.upper?);
                }
                for &(index, coefficient) in &row.terms {
                    residual[index] -= &multiplier * BigInt::from(coefficient);
                }
            }
            let mut premises = Vec::new();
            for (index, (coefficient, &(lower, upper))) in residual.into_iter().zip(bounds).enumerate() {
                let endpoint = if coefficient.is_negative() { upper } else { lower };
                if !coefficient.is_zero() {
                    let variable = VarId(u32::try_from(index).ok()?);
                    premises.push(if coefficient.is_negative() {
                        Premise::Le { var: variable, bound: upper }
                    } else {
                        Premise::Ge { var: variable, bound: lower }
                    });
                }
                bound += coefficient * BigInt::from(endpoint);
            }
            Some(CertifiedLpBound { value: bound.ceil().to_integer(), premises })
        }
    }

    fn objective_affine(objective: Objective<'_>, stop: &AtomicBool) -> Option<(i64, Vec<i64>, Vec<VarId>)> {
        match objective {
            Objective::Var(variable) => Some((0, vec![1], vec![variable])),
            Objective::VarWithAffine { objective, .. } | Objective::BoundedDiveVarWithAffine { objective, .. } => {
                Some((0, vec![1], vec![objective]))
            }
            Objective::Linear { coeffs, vars } | Objective::BoundedDiveLinear { coeffs, vars } => {
                (coeffs.len() == vars.len()).then(|| (0, coeffs.to_vec(), vars.to_vec()))
            }
            Objective::Expr(expression) | Objective::BoundedDiveExpr(expression) => expression.affine_form_interruptible(stop),
        }
    }

    fn exact_as_f64(value: i128) -> bool {
        value.unsigned_abs() <= u128::from(MAX_EXACT_F64_INTEGER)
    }

    fn clamp_bigint(value: &BigInt) -> i64 {
        value.to_i64().unwrap_or_else(|| if value.is_negative() { i64::MIN } else { i64::MAX })
    }

    fn phase_from_primal(solver: &Solver, search: &[VarId], primal: &[f64], max_variables: usize, stop: &AtomicBool) -> Vec<Option<i32>> {
        if primal.len() != solver.store.num_vars() || primal.len() > max_variables || stop.load(Ordering::Acquire) {
            return Vec::new();
        }
        let mut phase = vec![None; primal.len()];
        for &variable in search {
            let value = primal[variable.index()];
            if !value.is_finite() {
                return Vec::new();
            }
            let rounded = value.round().clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32;
            let candidate = if solver.store.contains(variable, rounded) {
                rounded
            } else {
                let lower = solver.store.min(variable);
                let upper = solver.store.max(variable);
                if (value - f64::from(lower)).abs() <= (value - f64::from(upper)).abs() {
                    lower
                } else {
                    upper
                }
            };
            phase[variable.index()] = Some(candidate);
        }
        phase
    }

    impl SearchRelaxationTemplate {
        fn new(
            model: Arc<LinearModel>,
            minimizing: bool,
            primal: Vec<f64>,
            certificate: Option<CertifiedLpBound>,
            controls: LinearControls,
        ) -> Self {
            let form = Arc::new(amthal_form(&model));
            Self {
                model,
                form,
                minimizing,
                root_primal: Arc::from(primal),
                root_certificate: certificate,
                node_time: controls.node_time,
                depth_interval: controls.node_depth_interval.max(1),
            }
        }

        pub(crate) fn start(&self) -> SearchRelaxation {
            SearchRelaxation {
                template: self.clone(),
                session: LpSession::new((*self.form).clone()),
                solved_bounds: self.model.bounds.clone(),
                primal: self.root_primal.to_vec(),
                certificate: self.root_certificate.clone(),
                next_depth: self.depth_interval,
                last_attempt_depth: 0,
                refactorizations: 0,
                stats: SolveStats::default(),
            }
        }
    }

    impl SearchRelaxation {
        pub(crate) fn check(&mut self, store: &Store, incumbent: Option<i64>) -> SearchCheck {
            let Some(incumbent) = incumbent else {
                return SearchCheck::Continue;
            };
            if let Some(certificate) = self.active_certificate(store) {
                if closes_incumbent(&certificate.value, incumbent, self.template.minimizing) {
                    return SearchCheck::Prune(certificate.premises.clone());
                }
            }
            let level = store.level();
            if level == 0 || self.template.node_time.is_zero() {
                return SearchCheck::Continue;
            }
            let current = store_bounds(store);
            let only_tightened = current
                .iter()
                .zip(&self.solved_bounds)
                .all(|(&(lower, upper), &(solved_lower, solved_upper))| lower >= solved_lower && upper <= solved_upper);
            if only_tightened && primal_within(&self.primal, &current) {
                return SearchCheck::Continue;
            }
            if level < self.last_attempt_depth {
                self.next_depth = next_scheduled_depth(level, self.template.depth_interval);
            }
            if level < self.next_depth {
                return SearchCheck::Continue;
            }

            for (index, &(lower, upper)) in current.iter().enumerate() {
                self.session.set_bounds(index, f64::from(lower), f64::from(upper));
            }
            self.session.deadline = Some(Instant::now() + self.template.node_time);
            let started = Instant::now();
            let solution = self.session.solve();
            self.stats.lp_solves = self.stats.lp_solves.saturating_add(1);
            self.stats.lp_micros = self.stats.lp_micros.saturating_add(u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX));
            self.stats.lp_timeouts = self.stats.lp_timeouts.saturating_add(u64::from(solution.status == LpStatus::TimeLimit));
            let refactorizations = self.session.refactorizations();
            self.stats.lp_refactorizations = self
                .stats
                .lp_refactorizations
                .saturating_add(u64::try_from(refactorizations.saturating_sub(self.refactorizations)).unwrap_or(u64::MAX));
            self.refactorizations = refactorizations;
            self.solved_bounds = current;
            self.last_attempt_depth = level;
            self.next_depth = level.saturating_add(self.template.depth_interval);

            if solution.status != LpStatus::Optimal {
                self.primal.clear();
                self.certificate = self.template.root_certificate.clone();
                return SearchCheck::Continue;
            }
            self.primal = solution.x;
            self.certificate = self.template.model.certify(&solution.duals, &self.solved_bounds);
            self.stats.lp_certified = self.stats.lp_certified.saturating_add(u64::from(self.certificate.is_some()));
            match self.active_certificate(store) {
                Some(certificate) if closes_incumbent(&certificate.value, incumbent, self.template.minimizing) => {
                    SearchCheck::Prune(certificate.premises.clone())
                }
                Some(_) | None => SearchCheck::Continue,
            }
        }

        fn active_certificate(&self, store: &Store) -> Option<&CertifiedLpBound> {
            self.certificate
                .as_ref()
                .filter(|certificate| premises_hold(store, &certificate.premises))
                .or_else(|| self.template.root_certificate.as_ref().filter(|certificate| premises_hold(store, &certificate.premises)))
        }

        pub(crate) fn record_prune(&mut self) {
            self.stats.lp_node_prunes = self.stats.lp_node_prunes.saturating_add(1);
        }

        pub(crate) fn stats(&self) -> SolveStats {
            self.stats
        }
    }

    fn store_bounds(store: &Store) -> Vec<(i32, i32)> {
        (0..store.num_vars())
            .map(|index| {
                let variable = VarId(u32::try_from(index).expect("store variable count exceeds VarId"));
                (store.min(variable), store.max(variable))
            })
            .collect()
    }

    fn primal_within(primal: &[f64], bounds: &[(i32, i32)]) -> bool {
        primal.len() == bounds.len()
            && primal
                .iter()
                .zip(bounds)
                .all(|(&value, &(lower, upper))| value.is_finite() && value >= f64::from(lower) - 1e-7 && value <= f64::from(upper) + 1e-7)
    }

    fn premises_hold(store: &Store, premises: &[Premise]) -> bool {
        premises.iter().all(|premise| match *premise {
            Premise::Ge { var, bound } => store.min(var) >= bound,
            Premise::Le { var, bound } => store.max(var) <= bound,
            Premise::Eq { var, val } => store.is_fixed(var) && store.value(var) == val,
            Premise::Ne { var, val } => !store.contains(var, val),
        })
    }

    fn closes_incumbent(normalized_bound: &BigInt, incumbent: i64, minimizing: bool) -> bool {
        let incumbent = BigInt::from(incumbent);
        normalized_bound >= &if minimizing { incumbent } else { -incumbent }
    }

    fn next_scheduled_depth(level: usize, interval: usize) -> usize {
        level.saturating_div(interval).saturating_add(1).saturating_mul(interval)
    }

    pub(crate) fn solve_root(
        solver: &Solver,
        search: &[VarId],
        objective: Objective<'_>,
        minimizing: bool,
        controls: LinearControls,
        stop: &AtomicBool,
    ) -> RootRelaxation {
        if controls.backend == LinearBackendMode::Native || controls.root_time.is_zero() || stop.load(Ordering::Acquire) {
            return RootRelaxation::default();
        }
        let Some(model) = LinearModel::build(solver, objective, minimizing, controls, stop) else {
            return RootRelaxation::default();
        };
        let model = Arc::new(model);
        let backend: Box<dyn LinearBackend> = match controls.backend {
            LinearBackendMode::Auto | LinearBackendMode::Amthal => Box::new(AmthalBackend),
            LinearBackendMode::Native => return RootRelaxation::default(),
        };
        let mut stats = SolveStats { lp_rows: u64::try_from(model.rows.len()).unwrap_or(u64::MAX), ..SolveStats::default() };
        let started = Instant::now();
        let Some(solution) = backend.solve(&model, controls.root_time, stop) else {
            return RootRelaxation { backend: Some(backend.name()), stats, ..RootRelaxation::default() };
        };
        stats.lp_solves = 1;
        stats.lp_refactorizations = solution.refactorizations;
        stats.lp_micros = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
        stats.lp_timeouts = u64::from(solution.status == LinearStatus::TimeLimit);
        if solution.status != LinearStatus::Optimal || stop.load(Ordering::Acquire) {
            let search = (!controls.node_time.is_zero())
                .then(|| SearchRelaxationTemplate::new(Arc::clone(&model), minimizing, Vec::new(), None, controls));
            return RootRelaxation { backend: Some(backend.name()), stats, search, ..RootRelaxation::default() };
        }
        let certificate = model.certify(&solution.row_duals, &model.bounds);
        let bound = certificate.as_ref().map(|certificate| {
            let normalized = if minimizing { certificate.value.clone() } else { -certificate.value.clone() };
            clamp_bigint(&normalized)
        });
        stats.lp_certified = u64::from(bound.is_some());
        stats.lp_root_bound = bound;
        let phase = phase_from_primal(solver, search, &solution.primal, controls.phase_max_variables, stop);
        let search = (!controls.node_time.is_zero())
            .then(|| SearchRelaxationTemplate::new(Arc::clone(&model), minimizing, solution.primal.clone(), certificate, controls));
        RootRelaxation { phase, bound, backend: Some(backend.name()), stats, search }
    }
}

#[cfg(feature = "lp-relaxation")]
pub(crate) use enabled::{solve_root, SearchCheck, SearchRelaxation, SearchRelaxationTemplate};
