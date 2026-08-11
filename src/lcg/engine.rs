//! CDCL search drivers.
//!
//! [`Cdcl::enumerate`] is the CSP driver: chronological-backtracking DFS, no
//! clause learning (which is unsound for enumerating every solution).
//! [`Cdcl::optimize`] is the COP driver: CDCL branch-and-bound with restarts.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use crate::expr::Expr;
use crate::ids::{PropId, VarId};
use crate::lcg::guarded_sum::GuardedSum;
use crate::lcg::lit::{Lit, LitOrConst};
use crate::lcg::trail::{Cdcl, Reason};
use crate::lcg::view::Tri;
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::search::{Objective, SearchControl, SharedObjectiveBound, SolveStats};
use crate::store::{Premise, Solver, Store};

/// One in every `REPHASE_PERIOD` restart segments rephases (ignores saved phases
/// and dives with a fresh polarity). The other segments keep saved phases, so
/// convergence is preserved and diversification stays a minority of the effort.
const REPHASE_PERIOD: u64 = 4;

/// Upper bound on the obvious work estimate for optional singleton filtering
/// of a symbolic objective. The root interval check remains enabled above it.
const MAX_EXPR_SINGLETON_PROBE_WORK: usize = 32_768;

/// Small objectives benefit from directed branching on the first dive. Large
/// objectives first need a cheap feasible assignment before their decisions
/// can safely dominate the feasibility heuristic.
const MAX_IMMEDIATE_OBJECTIVE_VARIABLES: usize = 32;

/// Literal evaluations reserved for a one-shot guarded-objective phase hint.
/// The first incumbent is published before this bounded local search starts.
const GUARDED_OBJECTIVE_HINT_WORK: usize = 32_000_000;

/// Enumerating up to a pseudo-random rank is acceptable only on compact
/// domains. Large bounds domains use a direct supported target or an endpoint
/// so value selection stays bounded at the search-loop level.
const MAX_ENUMERATED_RANDOM_DOMAIN: usize = 4_096;
const MIN_BOUNDED_DIVE_DECISIONS: u64 = 4_096;
const BOUNDED_DIVE_DECISIONS_PER_VARIABLE: u64 = 8;

use crate::mix64;

fn strict_bound(incumbent: i64, minimizing: bool) -> Option<i64> {
    if minimizing {
        incumbent.checked_sub(1)
    } else {
        incumbent.checked_add(1)
    }
}

impl Objective<'_> {
    fn value(self, solver: &Solver) -> i64 {
        match self {
            Self::Var(obj) | Self::VarWithAffine { objective: obj, .. } | Self::BoundedDiveVarWithAffine { objective: obj, .. } => {
                solver.store.value(obj) as i64
            }
            // Accumulate in i128 so large coeffs/many terms can't overflow, then
            // clamp: a wrapped objective would post a wrong bound (lost optimum).
            Self::Linear { coeffs, vars } | Self::BoundedDiveLinear { coeffs, vars } => coeffs
                .iter()
                .zip(vars)
                .map(|(&coeff, &var)| coeff as i128 * solver.store.value(var) as i128)
                .sum::<i128>()
                .clamp(i64::MIN as i128, i64::MAX as i128)
                as i64,
            Self::Expr(expr) | Self::BoundedDiveExpr(expr) => {
                expr.eval(&|var| solver.store.value(var) as i64).expect("objective expression is undefined at a solution")
            }
        }
    }
}

struct ObjectiveImpact {
    coeffs: Vec<u64>,
    directions: Vec<i8>,
    preferred_values: Vec<Option<i32>>,
    defer_until_incumbent: bool,
}

fn constant_expr_value(expression: &Expr) -> Option<i128> {
    match expression {
        Expr::Const(value) => Some(i128::from(*value)),
        Expr::Neg(value) => constant_expr_value(value)?.checked_neg(),
        Expr::Add(values) => values.iter().try_fold(0i128, |sum, value| sum.checked_add(constant_expr_value(value)?)),
        Expr::Sub(left, right) => constant_expr_value(left)?.checked_sub(constant_expr_value(right)?),
        Expr::Mul(values) => values.iter().try_fold(1i128, |product, value| product.checked_mul(constant_expr_value(value)?)),
        _ => None,
    }
}

/// Accumulate exact affine coefficients without allocating a second expression
/// tree. This is deliberately a value-ordering hint only: nonlinear or
/// overflowing expressions keep the regular seeded polarity.
fn accumulate_affine_expr_directions(expression: &Expr, factor: i128, directions: &mut [i128]) -> bool {
    match expression {
        Expr::Const(_) => true,
        Expr::Var(variable) => {
            let Some(value) = directions[variable.index()].checked_add(factor) else {
                return false;
            };
            directions[variable.index()] = value;
            true
        }
        Expr::Neg(value) => factor.checked_neg().is_some_and(|factor| accumulate_affine_expr_directions(value, factor, directions)),
        Expr::Add(values) => values.iter().all(|value| accumulate_affine_expr_directions(value, factor, directions)),
        Expr::Sub(left, right) => {
            accumulate_affine_expr_directions(left, factor, directions)
                && factor.checked_neg().is_some_and(|factor| accumulate_affine_expr_directions(right, factor, directions))
        }
        Expr::Mul(values) => {
            let mut scaled = factor;
            let mut nonconstant = None;
            for value in values {
                if let Some(constant) = constant_expr_value(value) {
                    let Some(next) = scaled.checked_mul(constant) else {
                        return false;
                    };
                    scaled = next;
                } else if nonconstant.replace(value).is_some() {
                    return false;
                }
            }
            nonconstant.is_none_or(|value| accumulate_affine_expr_directions(value, scaled, directions))
        }
        _ => false,
    }
}

fn affine_expr_directions(expression: &Expr, factor: i128, directions: &mut [i128]) -> bool {
    let mut candidate = directions.to_vec();
    if !accumulate_affine_expr_directions(expression, factor, &mut candidate) {
        return false;
    }
    directions.copy_from_slice(&candidate);
    true
}

/// Recover sound improving directions from a monotone expression that is not
/// affine, notably minima and maxima of affine terms. Direction `2` is an
/// internal conflict marker: the expression uses that variable with both
/// monotonicities, so value selection must remain neutral for it.
fn accumulate_monotone_expr_directions(expression: &Expr, factor: i8, directions: &mut [i8]) -> bool {
    fn merge(slot: &mut i8, direction: i8) {
        if direction == 0 || *slot == 2 {
            return;
        }
        if *slot == 0 || *slot == direction {
            *slot = direction;
        } else {
            *slot = 2;
        }
    }

    match expression {
        Expr::Const(_) => true,
        Expr::Var(variable) => {
            merge(&mut directions[variable.index()], factor);
            true
        }
        Expr::Neg(value) => accumulate_monotone_expr_directions(value, -factor, directions),
        Expr::Add(values) | Expr::Min(values) | Expr::Max(values) => {
            values.iter().all(|value| accumulate_monotone_expr_directions(value, factor, directions))
        }
        Expr::Sub(left, right) => {
            accumulate_monotone_expr_directions(left, factor, directions) && accumulate_monotone_expr_directions(right, -factor, directions)
        }
        Expr::Mul(values) => {
            let mut direction = factor;
            let mut symbolic = None;
            for value in values {
                if let Some(constant) = constant_expr_value(value) {
                    direction *= constant.signum() as i8;
                } else if symbolic.replace(value).is_some() {
                    return false;
                }
            }
            if direction == 0 {
                true
            } else {
                symbolic.is_none_or(|value| accumulate_monotone_expr_directions(value, direction, directions))
            }
        }
        Expr::Abs(_)
        | Expr::Eq(_, _)
        | Expr::Ne(_, _)
        | Expr::Lt(_, _)
        | Expr::Le(_, _)
        | Expr::Gt(_, _)
        | Expr::Ge(_, _)
        | Expr::And(_)
        | Expr::Or(_)
        | Expr::Not(_)
        | Expr::Imp(_, _)
        | Expr::Iff(_, _)
        | Expr::Div(_, _)
        | Expr::Mod(_, _)
        | Expr::IfThenElse(_, _, _) => false,
    }
}

fn monotone_expr_directions(expression: &Expr, factor: i8, directions: &mut [i128]) -> bool {
    let mut candidate = vec![0i8; directions.len()];
    if !accumulate_monotone_expr_directions(expression, factor, &mut candidate) {
        return false;
    }
    for (direction, candidate) in directions.iter_mut().zip(candidate) {
        *direction = if candidate == 2 { 0 } else { i128::from(candidate) };
    }
    true
}

#[cfg(test)]
pub(crate) fn audit_affine_expr_directions(expression: &Expr, variables: usize) -> (bool, Vec<i128>) {
    let mut directions = vec![0; variables];
    let affine = affine_expr_directions(expression, 1, &mut directions);
    (affine, directions)
}

/// Recover exact improving targets from an affine combination of equality
/// indicators. Constants, nested sums and products by constants are normalized
/// before checking signs. Every nonconstant leaf must be a complete
/// `variable == constant` indicator, and one variable may reward only one
/// target. A partial match would invent a preference for an unrelated term.
fn equality_indicator_preferences(expression: &Expr, store: &Store, minimizing: bool) -> Option<(Vec<u64>, Vec<Option<i32>>)> {
    fn equality_target(expression: &Expr) -> Option<(VarId, i32)> {
        let Expr::Eq(left, right) = expression else {
            return None;
        };
        match (&**left, &**right) {
            (Expr::Var(variable), Expr::Const(value)) | (Expr::Const(value), Expr::Var(variable)) => {
                Some((*variable, i32::try_from(*value).ok()?))
            }
            _ => None,
        }
    }

    fn collect(expression: &Expr, factor: i128, coefficients: &mut BTreeMap<(VarId, i32), i128>) -> Option<()> {
        match expression {
            Expr::Const(_) => Some(()),
            Expr::Add(terms) => {
                for term in terms {
                    collect(term, factor, coefficients)?;
                }
                Some(())
            }
            Expr::Sub(left, right) => {
                collect(left, factor, coefficients)?;
                collect(right, factor.checked_neg()?, coefficients)
            }
            Expr::Neg(value) => collect(value, factor.checked_neg()?, coefficients),
            Expr::Mul(factors) => {
                let mut scaled = factor;
                let mut symbolic = None;
                for value in factors {
                    if let Some(constant) = constant_expr_value(value) {
                        scaled = scaled.checked_mul(constant)?;
                    } else if symbolic.replace(value).is_some() {
                        return None;
                    }
                }
                if scaled == 0 {
                    return Some(());
                }
                match symbolic {
                    Some(value) => collect(value, scaled, coefficients),
                    None => Some(()),
                }
            }
            _ => {
                let target = equality_target(expression)?;
                let coefficient = coefficients.entry(target).or_default();
                *coefficient = coefficient.checked_add(factor)?;
                Some(())
            }
        }
    }

    let mut coefficients = BTreeMap::new();
    collect(expression, 1, &mut coefficients)?;
    coefficients.retain(|_, coefficient| *coefficient != 0);
    let mut impacts = vec![0u64; store.num_vars()];
    let mut preferences = vec![None; store.num_vars()];
    for ((variable, value), coefficient) in coefficients {
        let reward = if minimizing { coefficient.checked_neg()? } else { coefficient };
        let weight = u64::try_from(reward).ok().filter(|&weight| weight > 0)?;
        if variable.index() >= store.num_vars() {
            return None;
        }
        if !store.contains(variable, value) {
            // This equality is identically false on the declared domain, so
            // it contributes a constant zero and must not influence search.
            continue;
        }
        let slot = preferences.get_mut(variable.index())?;
        if slot.is_some_and(|current| current != value) {
            return None;
        }
        *slot = Some(value);
        impacts[variable.index()] = impacts[variable.index()].checked_add(weight)?;
    }
    preferences.iter().any(Option::is_some).then_some((impacts, preferences))
}

impl ObjectiveImpact {
    fn new(objective: Objective<'_>, solver: &Solver, minimizing: bool) -> Option<Self> {
        let store = &solver.store;
        let nvars = store.num_vars();
        let mut impact = vec![0u64; nvars];
        let mut signed = vec![0i128; nvars];
        let mut preferred_values = vec![None; nvars];
        let mut nonlinear = false;
        let mut defer_structural = false;
        let bounded_dive = matches!(
            objective,
            Objective::BoundedDiveVarWithAffine { .. } | Objective::BoundedDiveLinear { .. } | Objective::BoundedDiveExpr(_)
        );
        match objective {
            Objective::Var(var) => {
                // A materialized objective is often an auxiliary constrained by
                // the real decisions. Guide its value when it is selected, but
                // do not force it ahead of the feasibility heuristic.
                impact[var.index()] = 0;
                signed[var.index()] = if minimizing { 1 } else { -1 };
            }
            Objective::VarWithAffine { coeffs, vars, .. }
            | Objective::Linear { coeffs, vars }
            | Objective::BoundedDiveLinear { coeffs, vars }
            | Objective::BoundedDiveVarWithAffine { coeffs, vars, .. } => {
                // Every Boolean decision fixes its complete contribution to a
                // linear objective. A bounded dive can therefore prioritize a
                // wide Boolean sum cheaply. The complete pass preserves the
                // feasibility variable order for every wide objective and only
                // uses the exact coefficient signs for value selection after
                // it has found an incumbent.
                let boolean_objective =
                    vars.iter().all(|&variable| store.size(variable) <= 2 && store.min(variable) >= 0 && store.max(variable) <= 1);
                let prioritize_terms = vars.len() <= MAX_IMMEDIATE_OBJECTIVE_VARIABLES || (bounded_dive && boolean_objective);
                for (&coeff, &var) in coeffs.iter().zip(vars) {
                    if prioritize_terms {
                        impact[var.index()] = impact[var.index()].saturating_add(coeff.unsigned_abs());
                    }
                    let directed = if minimizing { i128::from(coeff) } else { -i128::from(coeff) };
                    signed[var.index()] = signed[var.index()].checked_add(directed)?;
                }
            }
            Objective::Expr(expression) | Objective::BoundedDiveExpr(expression) => {
                let factor = if minimizing { 1 } else { -1 };
                // Affine expressions provide a sound value direction. Do not
                // let coefficient magnitude eclipse dom/wdeg entirely:
                // feasibility variables still have to be interleaved with
                // objective decisions on tightly constrained models.
                if let Some((equality_impact, equality_preferences)) = equality_indicator_preferences(expression, store, minimizing) {
                    impact = equality_impact;
                    preferred_values = equality_preferences;
                    // On a constrained model, first preserve the ordinary
                    // feasibility trajectory. Once it supplies an
                    // incumbent, the exact rewarded values can drive
                    // improvement. An unconstrained equality sum can use
                    // its targets immediately.
                    defer_structural = solver.num_propagators() != 0;
                    if defer_structural {
                        // Once feasibility has produced an incumbent, use
                        // rewarded values as phases while keeping dom/wdeg
                        // in charge of variable order. Prioritizing every
                        // rewarded equality can repeatedly attack a jointly
                        // infeasible all-target assignment.
                        impact.fill(0);
                    }
                } else if !affine_expr_directions(expression, factor, &mut signed)
                    && !monotone_expr_directions(expression, factor as i8, &mut signed)
                {
                    nonlinear = true;
                }
                if nonlinear {
                    // An otherwise unsupported nonlinear objective still
                    // contains useful structural information. Syntactic
                    // occurrence is a conservative sensitivity proxy: it
                    // prioritizes selectors shared by many terms without
                    // inventing an improving value direction that may be false.
                    let mut variables = Vec::new();
                    expression.collect_vars(&mut variables);
                    for variable in variables {
                        impact[variable.index()] = impact[variable.index()].saturating_add(1);
                    }
                }
            }
        }
        let directions: Vec<i8> = signed.into_iter().map(|coeff| coeff.signum() as i8).collect();
        let mut active_variables = 0usize;
        for (index, ((coefficient, direction), preferred)) in impact.iter().zip(&directions).zip(&preferred_values).enumerate() {
            let variable = VarId(u32::try_from(index).ok()?);
            if store.size(variable) > 1 && (*coefficient != 0 || *direction != 0 || preferred.is_some()) {
                active_variables += 1;
            }
        }
        // A wide objective first follows the feasibility heuristic. Once that
        // supplies a verified incumbent, its exact value directions become a
        // useful improvement policy. A bounded dive opts into those directions
        // immediately and remains capped independently of model provenance.
        let defer_until_incumbent =
            !bounded_dive && (nonlinear || defer_structural || active_variables > MAX_IMMEDIATE_OBJECTIVE_VARIABLES);
        (active_variables > 0).then_some(Self { coeffs: impact, directions, preferred_values, defer_until_incumbent })
    }

    fn score(&self, solver: &Solver, var: VarId) -> u128 {
        let coeff = self.coeffs[var.index()] as u128;
        if coeff == 0 {
            return 0;
        }
        if self.preferred_values[var.index()].is_some() {
            return coeff;
        }
        let width = i64::from(solver.store.max(var)) - i64::from(solver.store.min(var));
        coeff.saturating_mul(width.max(0) as u128)
    }

    /// Preferred endpoint for a variable whose net objective contribution has
    /// a direction. Positive means that smaller values improve the objective;
    /// negative means that larger values do. Repeated terms that cancel leave
    /// value selection to the regular seeded polarity.
    fn prefers_max(&self, var: VarId) -> Option<bool> {
        match self.directions[var.index()].cmp(&0) {
            std::cmp::Ordering::Less => Some(true),
            std::cmp::Ordering::Greater => Some(false),
            std::cmp::Ordering::Equal => None,
        }
    }

    fn preferred_value(&self, var: VarId) -> Option<i32> {
        self.preferred_values[var.index()]
    }
}

/// Outcome of [`Cdcl::solve_under_assumptions`].
pub(crate) enum AssumptionOutcome {
    /// A full assignment over `vars` satisfying the root and the whole cube.
    Sat(Vec<i32>),
    /// A subset of the cube inconsistent with the root (an unsat core). Empty
    /// means the root alone is unsatisfiable, independent of the cube.
    Unsat(Vec<Lit>),
    /// The stop flag fired before a status was decided.
    Interrupted,
}

impl Cdcl<'_> {
    /// Branching heuristic: unfixed variable minimising `size / (wdeg + activity)`
    /// (dom/wdeg combined with VSIDS activity), or `None` if all fixed.
    fn select_var(&self, vars: &[VarId], objective: Option<&ObjectiveImpact>) -> Option<VarId> {
        for &v in &self.branch_order {
            if vars.contains(&v) && self.solver.store.size(v) > 1 {
                return Some(v);
            }
        }
        let mut best: Option<VarId> = None;
        let mut best_score = f64::INFINITY;
        let mut best_objective: Option<VarId> = None;
        let mut best_objective_impact = 0u128;
        let mut best_objective_score = f64::INFINITY;
        for &v in vars {
            let size = self.solver.store.size(v);
            if size > 1 {
                let wdeg = self.solver.store.var_weight_cached(v) as f64;
                let score = size as f64 / (wdeg + self.activity[v.index()]);
                if let Some(objective) = objective {
                    let impact = objective.score(self.solver, v);
                    if impact > best_objective_impact || (impact == best_objective_impact && impact > 0 && score < best_objective_score) {
                        best_objective_impact = impact;
                        best_objective_score = score;
                        best_objective = Some(v);
                    }
                }
                if score < best_score {
                    best_score = score;
                    best = Some(v);
                }
            }
        }
        best_objective.or(best)
    }

    /// Select from the semantic primary scope until it is exhausted, then
    /// complete the full assignment. Explicit branch order is evaluated only
    /// inside the active phase, so it cannot pull a completion variable ahead
    /// of an unfixed primary variable.
    fn select_branch_var(
        &self,
        vars: &[VarId],
        primary_branch_scope: Option<&[VarId]>,
        objective: Option<&ObjectiveImpact>,
    ) -> Option<VarId> {
        if let Some(primary) = primary_branch_scope {
            if let Some(variable) = self.select_var(primary, objective) {
                return Some(variable);
            }
        }
        self.select_var(vars, objective)
    }

    /// The decision literal for `v`: `[v = p]` toward the saved phase `p` when it
    /// is still in the domain (skipped on a rephasing segment), else a polarity
    /// endpoint chosen by [`rephase_value`](Self::rephase_value).
    fn decision_lit(&self, v: VarId, phase: &[Option<i32>], objective: Option<&ObjectiveImpact>) -> Lit {
        self.decision_lit_with_policy(v, phase, objective, false)
    }

    /// Variant used by the one-shot guarded-objective dive. Its locally
    /// optimized values are intentional phases, so directionless objective
    /// variables may consume them until a restart abandons that dive.
    fn decision_lit_with_policy(
        &self,
        v: VarId,
        phase: &[Option<i32>],
        objective: Option<&ObjectiveImpact>,
        honor_directionless_phase: bool,
    ) -> Lit {
        let nonlinear_objective_variable =
            objective.is_some_and(|impact| impact.prefers_max(v).is_none() && impact.score(self.solver, v) > 0);
        let honor_saved_phase = if nonlinear_objective_variable { honor_directionless_phase } else { self.rephase_mode == 0 };
        let val = match phase[v.index()] {
            Some(p) if honor_saved_phase && self.solver.store.contains(v, p) => p,
            _ => self.rephase_value(v, objective),
        };
        match self.atoms.eq(v, val) {
            LitOrConst::Lit(l) => l,
            _ => {
                unreachable!("an unfixed variable has a real equality atom for an in-domain value")
            }
        }
    }

    /// The domain endpoint to branch to when no saved phase applies: the
    /// objective-improving endpoint when available, otherwise a diversified
    /// endpoint. Rephasing only inverts the unguided choice: reversing a known
    /// objective direction was the source of long COP plateaus.
    fn rephase_value(&self, v: VarId, objective: Option<&ObjectiveImpact>) -> i32 {
        if let Some(value) = objective.and_then(|impact| impact.preferred_value(v)).filter(|&value| self.solver.store.contains(v, value)) {
            return value;
        }
        if objective.is_some_and(|impact| impact.prefers_max(v).is_none() && impact.score(self.solver, v) > 0) {
            // For a structural, directionless objective hint, both endpoints
            // are an unnecessarily small neighborhood on non-Boolean domains.
            // Choose any supported value deterministically and vary the choice
            // across restart segments.
            let size = self.solver.store.size(v);
            let variable_salt = (v.0 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let restart_salt = self.restarts_done.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            let random = mix64(self.seed ^ variable_salt ^ restart_salt);
            if size <= MAX_ENUMERATED_RANDOM_DOMAIN {
                let index = (random % size as u64) as usize;
                return self.solver.store.values(v).nth(index).expect("branch value index is inside the domain");
            }
            let min = self.solver.store.min(v);
            let max = self.solver.store.max(v);
            let span = (i64::from(max) - i64::from(min) + 1) as u64;
            let target = (i64::from(min) + (random % span) as i64) as i32;
            return if self.solver.store.contains(v, target) {
                target
            } else if random & 1 == 0 {
                min
            } else {
                max
            };
        }
        // A lone search always follows the exact improving direction. In a
        // clause-sharing portfolio, one seed class deliberately explores the
        // opposite branch so workers do not duplicate the same tree.
        let objective_preference = objective.and_then(|impact| impact.prefers_max(v)).map(|preference| {
            if self.clause_sharing.is_some() && self.seed & 3 == 1 {
                !preference
            } else {
                preference
            }
        });
        let want_max = objective_preference.unwrap_or_else(|| {
            // While an objective policy is active, variables without an exact
            // improving direction retain the historical deterministic hash.
            // Seed zero participates in that hash just like portfolio seeds.
            let diversified = objective.is_some() && mix64(self.seed ^ v.0 as u64) & 1 != 0;
            diversified ^ (self.rephase_mode != 0)
        });
        if want_max {
            self.solver.store.max(v)
        } else {
            self.solver.store.min(v)
        }
    }

    /// Restart to the root when the restart policy fires. Learned clauses and
    /// saved phases survive. Every `REPHASE_PERIOD` restarts the next segment
    /// *rephases*: it ignores saved phases and dives with an inverted or random
    /// polarity, escaping a region the saved phase keeps pulling search back to.
    fn maybe_restart(&mut self) -> bool {
        if self.should_restart() {
            self.backjump_to(0);
            self.restarts_done += 1;
            // Rephase only in a lone sequential search. Every `REPHASE_PERIOD`
            // restarts the next segment inverts the default polarity (not random:
            // an opposite dive diversifies the feasibility search without
            // shredding a structured nonlinear objective). Portfolio workers
            // already diversify across workers, and per-worker rephasing
            // *synchronizes* their polarity flips and hurts - so it is disabled
            // whenever clause sharing (a portfolio) is active.
            let lone = self.clause_sharing.is_none();
            self.rephase_mode = u8::from(lone && self.restarts_done.is_multiple_of(REPHASE_PERIOD));
            self.maybe_reduce_db();
            return self.sync_shared_clauses();
        }
        true
    }

    /// Enumerate solutions over `vars`, invoking `on_solution` per full
    /// assignment and stopping early on [`SearchControl::Stop`] or when `stop` is set.
    ///
    /// Plain chronological-backtracking DFS. Clause learning is deliberately not
    /// used: resolving against the blocking clauses (not entailed by the
    /// constraints) can drop sibling solutions when enumerating.
    // TODO(strong): a sound CDCL enumeration (restart-per-solution with blocking
    // clauses kept out of analysis, or dual reasoning) would prune harder.
    pub fn enumerate<F>(
        &mut self,
        vars: &[VarId],
        primary_branch_scope: Option<&[VarId]>,
        mut on_solution: F,
        stop: &AtomicBool,
    ) -> SolveStats
    where
        F: FnMut(&Solver) -> SearchControl,
    {
        let mut stats = SolveStats::default();
        if !self.init() || !self.root_probe(primary_branch_scope.unwrap_or(vars)) {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return stats; // root unsatisfiable
        }
        let mut phase: Vec<Option<i32>> = vec![None; self.solver.store.num_vars()];
        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            match self.propagate() {
                Ok(()) => match self.select_branch_var(vars, primary_branch_scope, None) {
                    None => {
                        // Full assignment.
                        for &v in vars {
                            phase[v.index()] = Some(self.solver.store.value(v));
                        }
                        stats.solutions += 1;
                        if matches!(on_solution(&*self.solver), SearchControl::Stop) {
                            break;
                        }
                        if !self.backtrack_and_forbid() {
                            break; // whole search space explored
                        }
                    }
                    Some(v) => {
                        stats.nodes += 1;
                        let lit = self.decision_lit(v, &phase, None);
                        self.decide(lit).expect("in-domain decision cannot fail");
                    }
                },
                Err(_) => {
                    stats.failures += 1;
                    if !self.backtrack_and_forbid() {
                        break; // dead end at the root: search exhausted
                    }
                }
            }
        }
        self.copy_inprocessing_stats(&mut stats);
        stats
    }

    /// Undo the deepest decision `x = v` and forbid `v` for `x` (DFS "try the next
    /// value"); climb if that empties `x`. Returns `false` once the space is exhausted.
    fn backtrack_and_forbid(&mut self) -> bool {
        loop {
            let d = self.decision_level();
            if d == 0 {
                return false;
            }
            let dec = self.deepest_decision();
            self.backjump_to(d - 1);
            // Assert the decision's negation via `assign` (not a direct domain
            // edit) so it is trailed at this level and its event is drained.
            match self.assign(dec.negate(), Reason::Decision) {
                Ok(()) => return true,
                Err(_) => continue, // emptied the variable: climb
            }
        }
    }

    /// The literal that demands a strictly better objective than `value`:
    /// `¬[obj ≥ value]` when minimizing, `[obj ≥ value+1]` when maximizing.
    /// `None` means no better value is representable; `value` is optimal.
    fn improvement_lit(&self, obj: VarId, value: i32, minimizing: bool) -> Option<Lit> {
        if minimizing {
            match self.atoms.ge(obj, value) {
                LitOrConst::Lit(l) => Some(l.negate()), // obj < value
                LitOrConst::True => None,               // obj ≥ value always: optimal
                LitOrConst::False => unreachable!("found objective value below its own minimum"),
            }
        } else {
            let next = value.checked_add(1)?;
            match self.atoms.ge(obj, next) {
                LitOrConst::Lit(l) => Some(l), // obj > value
                LitOrConst::False => None,     // obj ≤ value always: optimal
                LitOrConst::True => unreachable!("found objective value above its own maximum"),
            }
        }
    }

    /// Assert that the objective must improve on `incumbent`. Returns `false`
    /// when no better value exists under the imported bound.
    fn tighten_bound(&mut self, obj: VarId, minimizing: bool, incumbent: i32) -> bool {
        let Some(bound) = self.improvement_lit(obj, incumbent, minimizing) else {
            return false;
        };
        self.backjump_to(0);
        self.set_bound_scope(bound);
        self.assert_root_lit(bound)
    }

    /// Assert a strict improvement on a materialized or symbolic objective.
    ///
    /// The `Var` path asserts a root literal on the existing objective atom, so
    /// it never accumulates. `Linear`/`Expr` have no objective atom, so the bound
    /// is enforced by ONE persistent propagator whose rhs lives in `cell`; each
    /// improving incumbent updates the cell (bounds only ever tighten, keeping it
    /// sound) and re-enqueues that propagator instead of posting a new one.
    fn tighten_objective(&mut self, objective: Objective<'_>, minimizing: bool, incumbent: i64, cell: &mut ObjBoundCell) -> bool {
        match objective {
            Objective::Var(obj)
            | Objective::VarWithAffine { objective: obj, .. }
            | Objective::BoundedDiveVarWithAffine { objective: obj, .. } => self.tighten_bound(obj, minimizing, incumbent as i32),
            Objective::Linear { coeffs, vars } | Objective::BoundedDiveLinear { coeffs, vars } => {
                let Some(bound) = strict_bound(incumbent, minimizing) else {
                    return false;
                };
                self.backjump_to(0);
                // Minimizing: ∑ coeffs·vars ≤ bound. Maximizing: negate both so it
                // is still a `≤ c` constraint whose c falls as incumbents improve.
                let c = if minimizing { bound } else { -bound };
                match &cell.handle {
                    Some((id, atom)) => {
                        atom.store(c, Ordering::Relaxed);
                        self.solver.store.enqueue(*id);
                    }
                    None => {
                        let atom = Arc::new(AtomicI64::new(c));
                        let coeffs = coeffs
                            .iter()
                            .map(|&coefficient| {
                                let coefficient = i128::from(coefficient);
                                if minimizing {
                                    coefficient
                                } else {
                                    -coefficient
                                }
                            })
                            .collect();
                        let id = self.solver.post(Box::new(ObjLinearLeq::new(coeffs, vars, Arc::clone(&atom))));
                        cell.handle = Some((id, atom));
                    }
                }
                self.propagate_and_learn()
            }
            Objective::Expr(e) | Objective::BoundedDiveExpr(e) => {
                let Some(bound) = strict_bound(incumbent, minimizing) else {
                    return false;
                };
                self.backjump_to(0);
                match &cell.handle {
                    Some((id, atom)) => {
                        atom.store(bound, Ordering::Relaxed);
                        self.solver.store.enqueue(*id);
                    }
                    None => {
                        let atom = Arc::new(AtomicI64::new(bound));
                        let id = self.solver.post(Box::new(ObjExprBound::new(e.clone(), minimizing, Arc::clone(&atom))));
                        cell.handle = Some((id, atom));
                    }
                }
                self.propagate_and_learn()
            }
        }
    }

    /// Assert the root units defining one disjoint search cube.
    fn assume_cube(&mut self, cube: &[Lit]) -> bool {
        self.set_cube_scope(cube);
        cube.iter().copied().all(|lit| self.assert_root_lit(lit))
    }

    /// Assert `obj <= target` when minimizing or `obj >= target` when maximizing.
    fn assume_objective_bound(&mut self, obj: VarId, target: i32, minimizing: bool) -> bool {
        let bound = if minimizing {
            let Some(next) = target.checked_add(1) else {
                return true;
            };
            match self.atoms.ge(obj, next) {
                LitOrConst::True => return false,
                LitOrConst::False => return true,
                LitOrConst::Lit(l) => l.negate(),
            }
        } else {
            match self.atoms.ge(obj, target) {
                LitOrConst::True => return true,
                LitOrConst::False => return false,
                LitOrConst::Lit(l) => l,
            }
        };
        self.set_bound_scope(bound);
        self.assert_root_lit(bound)
    }

    /// Pick one binary split for a cube, or `None` when it is already terminal.
    pub(crate) fn split_cube(&mut self, vars: &[VarId], primary_branch_scope: Option<&[VarId]>, cube: &[Lit]) -> Option<Lit> {
        if self.stopped() || !self.init() || !self.assume_cube(cube) || self.stopped() {
            return None;
        }
        let phase = vec![None; self.solver.store.num_vars()];
        let split_scope = primary_branch_scope.unwrap_or(vars);
        self.select_var(split_scope, None).map(|v| self.decision_lit(v, &phase, None))
    }

    /// Find one solution under an optimistic objective bound.
    pub(crate) fn probe(
        &mut self,
        vars: &[VarId],
        primary_branch_scope: Option<&[VarId]>,
        obj: VarId,
        minimizing: bool,
        target: i32,
        stop: &AtomicBool,
    ) -> (Option<(Vec<i32>, i32)>, SolveStats, bool) {
        let mut stats = SolveStats::default();
        if !self.init() || !self.assume_objective_bound(obj, target, minimizing) || !self.sync_shared_clauses() {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return (None, stats, true);
        }
        let phase = vec![None; self.solver.store.num_vars()];
        let mut complete = true;
        let found = loop {
            if stop.load(Ordering::Relaxed) {
                complete = false;
                break None;
            }
            if !self.maybe_restart() {
                break None;
            }
            match self.select_branch_var(vars, primary_branch_scope, None) {
                None => {
                    stats.solutions += 1;
                    let value = self.solver.store.value(obj);
                    let assignment = vars.iter().map(|&v| self.solver.store.value(v)).collect();
                    break Some((assignment, value));
                }
                Some(v) => {
                    stats.nodes += 1;
                    let lit = self.decision_lit(v, &phase, None);
                    self.decide(lit).expect("in-domain decision cannot fail");
                    if !self.propagate_and_learn() {
                        break None;
                    }
                }
            }
        };
        stats.failures = self.conflicts;
        self.copy_inprocessing_stats(&mut stats);
        (found, stats, complete)
    }

    /// Record a new incumbent, report it, and assert a strictly-better objective
    /// bound at level 0. Returns `false` when the incumbent is proven optimal.
    #[allow(clippy::too_many_arguments)]
    fn accept_solution<F: FnMut(i64, &[i32])>(
        &mut self,
        vars: &[VarId],
        objective: Objective<'_>,
        minimizing: bool,
        best: &mut Option<(Vec<i32>, i64)>,
        enforced: &mut Option<i64>,
        phase: &mut [Option<i32>],
        stats: &mut SolveStats,
        cell: &mut ObjBoundCell,
        on_improve: &mut F,
    ) -> bool {
        let value = objective.value(self.solver);
        let assignment: Vec<i32> = vars.iter().map(|&v| self.solver.store.value(v)).collect();
        for &v in vars {
            phase[v.index()] = Some(self.solver.store.value(v));
        }
        stats.solutions += 1;
        on_improve(value, &assignment);
        *best = Some((assignment, value));
        *enforced = Some(value);
        self.tighten_objective(objective, minimizing, value, cell)
    }

    /// CDCL branch-and-bound with restarts. Returns the best `(assignment,
    /// value)`, proven optimal unless `stop` was set.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn optimize<F: FnMut(i64, &[i32])>(
        &mut self,
        vars: &[VarId],
        primary_branch_scope: Option<&[VarId]>,
        objective: Objective<'_>,
        minimizing: bool,
        stop: &AtomicBool,
        shared_bound: Option<&SharedObjectiveBound>,
        cube: &[Lit],
        conflict_budget: Option<u64>,
        mut on_improve: F,
    ) -> (Option<(Vec<i32>, i64)>, SolveStats, bool) {
        self.optimize_with_mode(
            vars,
            primary_branch_scope,
            objective,
            minimizing,
            stop,
            shared_bound,
            cube,
            conflict_budget,
            &mut on_improve,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn optimize_with_mode<F: FnMut(i64, &[i32])>(
        &mut self,
        vars: &[VarId],
        primary_branch_scope: Option<&[VarId]>,
        objective: Objective<'_>,
        minimizing: bool,
        stop: &AtomicBool,
        shared_bound: Option<&SharedObjectiveBound>,
        cube: &[Lit],
        conflict_budget: Option<u64>,
        on_improve: &mut F,
    ) -> (Option<(Vec<i32>, i64)>, SolveStats, bool) {
        if let Objective::VarWithAffine { coeffs, vars, .. }
        | Objective::Linear { coeffs, vars }
        | Objective::BoundedDiveLinear { coeffs, vars }
        | Objective::BoundedDiveVarWithAffine { coeffs, vars, .. } = objective
        {
            assert_eq!(coeffs.len(), vars.len(), "affine objective view: coeffs/terms length mismatch");
        }
        let mut stats = SolveStats::default();
        let mut best: Option<(Vec<i32>, i64)> = None;
        let bounded_dive_decisions = matches!(
            objective,
            Objective::BoundedDiveVarWithAffine { .. } | Objective::BoundedDiveLinear { .. } | Objective::BoundedDiveExpr(_)
        )
        .then(|| {
            u64::try_from(vars.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(BOUNDED_DIVE_DECISIONS_PER_VARIABLE)
                .max(MIN_BOUNDED_DIVE_DECISIONS)
        });
        self.set_conflict_budget(conflict_budget);
        if !self.init() {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return (best, stats, !self.conflict_budget_exhausted());
        }
        if cube.is_empty() && conflict_budget.is_none() && !self.root_probe(primary_branch_scope.unwrap_or(vars)) {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return (best, stats, true);
        }
        if !self.assume_cube(cube) {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return (best, stats, !self.conflict_budget_exhausted());
        }
        // Seed the saved-phase array with the caller's value-ordering hint (e.g.
        // nearest-neighbour successors) when supplied, else start blank.
        let mut phase: Vec<Option<i32>> = if self.initial_phase.len() == self.solver.store.num_vars() {
            self.initial_phase.clone()
        } else {
            vec![None; self.solver.store.num_vars()]
        };
        let objective_impact = ObjectiveImpact::new(objective, self.solver, minimizing);
        // A generated guarded-sum objective admits a much cheaper local
        // minimization than a generic expression tree. Keep that computation
        // outside the propagator: its assignment is only a phase hint, while
        // the regular CP search remains responsible for feasibility and proof.
        // Explicit caller guidance retains its requested behavior. A bounded
        // dive may still use this one-shot phase hint: the conflict budget
        // continues to bound the exact search, while the hint itself never
        // publishes a candidate or claims feasibility.
        let mut guarded_hint_pending = minimizing
            && cube.is_empty()
            && self.initial_phase.iter().all(Option::is_none)
            && self.branch_order.is_empty()
            && matches!(objective, Objective::Expr(_) | Objective::BoundedDiveExpr(_));
        let mut guarded_hint_restart = None;
        // A preceding dive or another worker may already have published an
        // incumbent. In that case the imported strict cutoff can prevent this
        // worker from accepting the ordinary first solution that used to
        // trigger the guarded-sum hint. Seed the phase up front when the hint
        // itself improves the shared bound. This remains only a value-ordering
        // policy: CP still checks every constraint before publishing anything.
        if guarded_hint_pending {
            let shared_incumbent = shared_bound.and_then(SharedObjectiveBound::load);
            let expression = match objective {
                Objective::Expr(expression) | Objective::BoundedDiveExpr(expression) => Some(expression),
                Objective::Var(_)
                | Objective::VarWithAffine { .. }
                | Objective::BoundedDiveVarWithAffine { .. }
                | Objective::Linear { .. }
                | Objective::BoundedDiveLinear { .. } => None,
            };
            if let (Some(incumbent), Some(expression)) = (shared_incumbent, expression) {
                if let Some((_, assignment)) = GuardedSum::compile(expression)
                    .and_then(|guarded| guarded.minimize_hint(&self.solver.store, self.seed, stop, GUARDED_OBJECTIVE_HINT_WORK))
                    .filter(|(hint_value, _)| *hint_value < incumbent)
                {
                    for (variable, value) in assignment {
                        phase[variable.index()] = Some(value);
                    }
                    guarded_hint_pending = false;
                    guarded_hint_restart = Some(self.restarts_done);
                }
            }
        }
        let mut enforced = None;
        let mut obj_bound = ObjBoundCell::default();
        let mut complete = true;
        if !self.sync_shared_clauses() {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return (best, stats, !self.conflict_budget_exhausted());
        }

        loop {
            if stop.load(Ordering::Relaxed)
                || self.conflict_budget_reached()
                || bounded_dive_decisions.is_some_and(|budget| stats.nodes >= budget)
            {
                complete = false;
                break;
            }
            if let Some(value) = shared_bound.and_then(SharedObjectiveBound::load) {
                let stronger = enforced.is_none_or(|old| if minimizing { value < old } else { value > old });
                if stronger {
                    enforced = Some(value);
                    if !self.tighten_objective(objective, minimizing, value, &mut obj_bound) {
                        if self.conflict_budget_exhausted() {
                            complete = false;
                        }
                        break;
                    }
                }
            }
            if !self.maybe_restart() {
                if self.conflict_budget_exhausted() {
                    complete = false;
                }
                break;
            }
            if guarded_hint_restart.is_some_and(|restart| restart != self.restarts_done) {
                guarded_hint_restart = None;
            }
            // A nonlinear occurrence score has no sound improving polarity.
            // Activate it only after the regular feasibility heuristic has
            // supplied an incumbent and therefore a useful objective bound.
            let active_objective =
                objective_impact.as_ref().filter(|impact| !impact.defer_until_incumbent || best.is_some() || enforced.is_some());
            match self.select_branch_var(vars, primary_branch_scope, active_objective) {
                None => {
                    // A successful hinted dive consumes the one-shot policy.
                    // Its accepted incumbent becomes the ordinary saved phase
                    // and must not replay as another guarded dive.
                    guarded_hint_restart = None;
                    let keep_searching = self.accept_solution(
                        vars,
                        objective,
                        minimizing,
                        &mut best,
                        &mut enforced,
                        &mut phase,
                        &mut stats,
                        &mut obj_bound,
                        on_improve,
                    );
                    if stop.load(Ordering::Relaxed) {
                        complete = false;
                        break;
                    }
                    if !keep_searching {
                        if self.conflict_budget_exhausted() {
                            complete = false;
                        }
                        break; // optimal
                    }
                    if guarded_hint_pending {
                        guarded_hint_pending = false;
                        let incumbent = best.as_ref().expect("the accepted solution is the incumbent").1;
                        let hint = match objective {
                            Objective::Expr(expression) | Objective::BoundedDiveExpr(expression) => GuardedSum::compile(expression)
                                .and_then(|guarded| {
                                    guarded.minimize_hint(&self.solver.store, self.seed, stop, GUARDED_OBJECTIVE_HINT_WORK)
                                }),
                            Objective::Var(_)
                            | Objective::VarWithAffine { .. }
                            | Objective::BoundedDiveVarWithAffine { .. }
                            | Objective::Linear { .. }
                            | Objective::BoundedDiveLinear { .. } => None,
                        };
                        if let Some((hint_value, assignment)) = hint {
                            // Only displace the incumbent phase when the exact
                            // guarded objective improved. The hinted assignment
                            // may violate other constraints, so it is never
                            // published directly and cannot affect correctness.
                            if hint_value < incumbent {
                                for (variable, value) in assignment {
                                    phase[variable.index()] = Some(value);
                                }
                                guarded_hint_restart = Some(self.restarts_done);
                            }
                        }
                    }
                }
                Some(v) => {
                    stats.nodes += 1;
                    let lit = self.decision_lit_with_policy(v, &phase, active_objective, guarded_hint_restart.is_some());
                    self.decide(lit).expect("in-domain decision cannot fail");
                    if !self.propagate_and_learn() {
                        if self.conflict_budget_exhausted() {
                            complete = false;
                        }
                        break; // tree exhausted under the bound: optimal
                    }
                }
            }
        }
        stats.failures = self.conflicts;
        self.copy_inprocessing_stats(&mut stats);
        (best, stats, complete)
    }

    /// CDCL decision driver for CSP: find one solution or prove UNSAT.
    /// It never posts solution-blocking clauses, so learned clauses remain
    /// consequences of the model. Use enumeration for counting/all-solutions.
    pub(crate) fn decide_sat(
        &mut self,
        vars: &[VarId],
        primary_branch_scope: Option<&[VarId]>,
        stop: &AtomicBool,
    ) -> (Option<Vec<i32>>, SolveStats, bool) {
        let mut stats = SolveStats::default();
        if !self.init() || !self.root_probe(primary_branch_scope.unwrap_or(vars)) || !self.sync_shared_clauses() {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return (None, stats, true);
        }
        if self.initial_phase.len() == self.solver.store.num_vars() {
            self.saved_phase = self.initial_phase.clone();
        }
        let mut complete = true;
        let solution = loop {
            if stop.load(Ordering::Relaxed) {
                complete = false;
                break None;
            }
            if !self.maybe_restart() {
                break None;
            }
            match self.select_branch_var(vars, primary_branch_scope, None) {
                None => {
                    stats.solutions += 1;
                    break Some(vars.iter().map(|&v| self.solver.store.value(v)).collect());
                }
                Some(v) => {
                    stats.nodes += 1;
                    let lit = self.decision_lit(v, &self.saved_phase, None);
                    self.decide(lit).expect("in-domain decision cannot fail");
                    if !self.propagate_and_learn() {
                        break None;
                    }
                }
            }
        };
        stats.failures = self.conflicts;
        self.copy_inprocessing_stats(&mut stats);
        (solution, stats, complete)
    }

    /// CDCL decision driver under a temporary assumption cube. Learned clauses
    /// exported from this run are scoped by the negated cube literals.
    pub(crate) fn decide_sat_assuming(
        &mut self,
        vars: &[VarId],
        primary_branch_scope: Option<&[VarId]>,
        cube: &[Lit],
        conflict_budget: Option<u64>,
        stop: &AtomicBool,
    ) -> (Option<Vec<i32>>, SolveStats, bool) {
        let mut stats = SolveStats::default();
        self.set_conflict_budget(conflict_budget);
        if !self.init() || !self.assume_cube(cube) || !self.sync_shared_clauses() {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return (None, stats, !self.conflict_budget_exhausted());
        }
        if self.initial_phase.len() == self.solver.store.num_vars() {
            self.saved_phase = self.initial_phase.clone();
        }
        let mut complete = true;
        let solution = loop {
            if stop.load(Ordering::Relaxed) || self.conflict_budget_reached() {
                complete = false;
                break None;
            }
            if !self.maybe_restart() {
                complete = !self.conflict_budget_exhausted();
                break None;
            }
            match self.select_branch_var(vars, primary_branch_scope, None) {
                None => {
                    stats.solutions += 1;
                    break Some(vars.iter().map(|&v| self.solver.store.value(v)).collect());
                }
                Some(v) => {
                    stats.nodes += 1;
                    let lit = self.decision_lit(v, &self.saved_phase, None);
                    self.decide(lit).expect("in-domain decision cannot fail");
                    if !self.propagate_and_learn() {
                        complete = !self.conflict_budget_exhausted();
                        break None;
                    }
                }
            }
        };
        stats.failures = self.conflicts;
        self.copy_inprocessing_stats(&mut stats);
        (solution, stats, complete)
    }

    /// Solve the root constraints under `cube`, a set of assumption literals,
    /// returning a model, an unsat core (subset of `cube`), or interruption.
    ///
    /// MiniSat-style assumption solving: each assumption is placed as its own
    /// decision (not a root unit), so learned clauses stay entailed by the root
    /// alone and a refutation traces back through [`analyze_final`] to exactly
    /// the assumptions that participated. `decision_level()` doubles as the index
    /// of the next assumption to place; already-implied assumptions get a dummy
    /// level to keep that alignment (and to be replayed identically after a
    /// restart, which backjumps to the root).
    pub(crate) fn solve_under_assumptions(&mut self, vars: &[VarId], cube: &[Lit], stop: &AtomicBool) -> AssumptionOutcome {
        if !self.init() {
            return AssumptionOutcome::Unsat(Vec::new()); // root already unsatisfiable
        }
        self.resolve_assumptions(vars, cube, stop)
    }

    /// Re-solve under `cube` from the current root, reusing the engine's learned
    /// clauses (all entailed by the root, hence valid under any cube). The caller
    /// must have run [`init`](Cdcl::init) once. This is the per-query primitive
    /// for MUS minimisation, which fires many cubes at one engine.
    pub(crate) fn resolve_assumptions(&mut self, vars: &[VarId], cube: &[Lit], stop: &AtomicBool) -> AssumptionOutcome {
        self.backjump_to(0);
        loop {
            if stop.load(Ordering::Relaxed) {
                return AssumptionOutcome::Interrupted;
            }
            if !self.maybe_restart() {
                return AssumptionOutcome::Unsat(Vec::new());
            }
            let d = self.decision_level();
            if d < cube.len() {
                // Phase 1: place the next assumption as a decision.
                let p = cube[d];
                match self.value(p) {
                    Tri::True => self.open_level(), // already implied: dummy level
                    Tri::False => {
                        let core = self.analyze_final(p);
                        self.backjump_to(0);
                        return AssumptionOutcome::Unsat(core);
                    }
                    Tri::Unknown => {
                        self.decide(p).expect("in-domain assumption decision cannot fail");
                        if !self.propagate_and_learn() {
                            // Level-0 refutation: every learnt clause is root-entailed,
                            // so this means the root alone is unsatisfiable.
                            self.backjump_to(0);
                            return AssumptionOutcome::Unsat(Vec::new());
                        }
                    }
                }
                continue;
            }
            // Phase 2: assumptions all satisfied, branch for a model. A conflict
            // that refutes the cube backjumps below `cube.len()`, re-entering
            // Phase 1 where `analyze_final` catches the now-false assumption.
            match self.select_var(vars, None) {
                None => return AssumptionOutcome::Sat(vars.iter().map(|&v| self.solver.store.value(v)).collect()),
                Some(v) => {
                    let lit = self.decision_lit(v, &self.saved_phase, None);
                    self.decide(lit).expect("in-domain decision cannot fail");
                    if !self.propagate_and_learn() {
                        self.backjump_to(0);
                        return AssumptionOutcome::Unsat(Vec::new()); // root alone is unsat
                    }
                }
            }
        }
    }
}

/// Handle to the single persistent objective-bound propagator (Linear/Expr).
/// Owned by one `optimize` run; the propagator is posted lazily on the first
/// improving incumbent, so a solver cloned from the immutable model template
/// (parallel/cube workers, see `parallel.rs`) never aliases another worker's
/// bound cell, each posts its own on its own first incumbent.
// ponytail: per-run cell; no cross-run reuse needed since a solver runs one optimize.
#[derive(Default)]
struct ObjBoundCell {
    handle: Option<(PropId, Arc<AtomicI64>)>,
}

/// `∑ coeffs·vars ≤ bound`, with `bound` read from a shared cell each run so it
/// can tighten monotonically in place. Filtering mirrors `constraints::linear`'s
/// `LinearLeq`; the only difference is the mutable rhs.
#[derive(Clone)]
struct ObjLinearLeq {
    coeffs: Vec<i128>,
    vars: Vec<VarId>,
    bound: Arc<AtomicI64>,
    term_min: Vec<i128>,
}

impl ObjLinearLeq {
    fn new(coeffs: Vec<i128>, vars: &[VarId], bound: Arc<AtomicI64>) -> Self {
        let n = vars.len();
        Self { coeffs, vars: vars.to_vec(), bound, term_min: vec![0; n] }
    }

    fn min_side(&self, store: &Store, skip: usize) -> Vec<Premise> {
        let mut why = Vec::new();
        for (j, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
            if j == skip || a == 0 {
                continue;
            }
            why.push(if a > 0 { Premise::Ge { var: v, bound: store.min(v) } } else { Premise::Le { var: v, bound: store.max(v) } });
        }
        why
    }
}

impl Propagator for ObjLinearLeq {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::BoundChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let c = i128::from(self.bound.load(Ordering::Relaxed));
        loop {
            let mut sum_min = 0i128;
            let mut sum_max = 0i128;
            for (slot, (&a, &v)) in self.term_min.iter_mut().zip(self.coeffs.iter().zip(&self.vars)) {
                let lo = i128::from(store.min(v));
                let hi = i128::from(store.max(v));
                let (tmin, tmax) = if a >= 0 { (a * lo, a * hi) } else { (a * hi, a * lo) };
                *slot = tmin;
                let Some(next_min) = sum_min.checked_add(tmin) else {
                    return Ok(());
                };
                let Some(next_max) = sum_max.checked_add(tmax) else {
                    return Ok(());
                };
                sum_min = next_min;
                sum_max = next_max;
            }

            if sum_min > c {
                return Err(store.fail_because(self.min_side(store, usize::MAX)));
            }
            if sum_max <= c {
                return Ok(()); // entailed
            }

            let mut changed = false;
            for (idx, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
                if a == 0 {
                    continue;
                }
                let Some(other_min) = sum_min.checked_sub(self.term_min[idx]) else {
                    return Ok(());
                };
                let Some(allowed) = c.checked_sub(other_min) else {
                    return Ok(());
                };
                if a > 0 {
                    let bound = clamp_i128_i32(floor_div(allowed, a));
                    if bound < store.max(v) {
                        store.remove_above_because(v, bound, self.min_side(store, idx))?;
                        changed = true;
                    }
                } else {
                    let bound = clamp_i128_i32(ceil_div(allowed, a));
                    if bound > store.min(v) {
                        store.remove_below_because(v, bound, self.min_side(store, idx))?;
                        changed = true;
                    }
                }
            }

            if !changed {
                return Ok(());
            }
        }
    }
}

/// `expr ≤ bound` when minimizing, or `expr ≥ bound` when maximizing, with the
/// bound read from a shared cell. Keeping the relation explicit avoids negating
/// `i64::MIN` while preserving one persistent propagator per optimization run.
#[derive(Clone)]
struct ObjExprBound {
    expr: Expr,
    guarded_sum: Option<GuardedSum>,
    vars: Vec<VarId>,
    bound: Arc<AtomicI64>,
    minimizing: bool,
    scratch: Vec<i32>,
    probe_cost_per_value: usize,
}

impl ObjExprBound {
    fn new(expr: Expr, minimizing: bool, bound: Arc<AtomicI64>) -> Self {
        let guarded_sum = GuardedSum::compile(&expr);
        let mut vars = Vec::new();
        expr.collect_vars(&mut vars);
        let occurrences = vars.len();
        vars.sort_unstable();
        vars.dedup();
        if let Some(guarded) = &guarded_sum {
            vars = guarded.vars().to_vec();
        }
        // Singleton probing recomputes the complete expression for every value
        // of every unfixed variable. It is useful on compact expressions, but
        // becomes the dominant cost on generated guarded sums and wide
        // domains. Keep the sound interval check in all cases; `propagate`
        // enables this optional filtering only when the current domain-size
        // estimate also fits the work ceiling.
        Self { expr, guarded_sum, vars, bound, minimizing, scratch: Vec::new(), probe_cost_per_value: occurrences.max(1) }
    }

    fn impossible(&self, lo: i64, hi: i64, bound: i64) -> bool {
        if self.minimizing {
            lo > bound
        } else {
            hi < bound
        }
    }

    fn entailed(&self, lo: i64, hi: i64, bound: i64) -> bool {
        if self.minimizing {
            hi <= bound
        } else {
            lo >= bound
        }
    }

    fn accepts(&self, value: i64, bound: i64) -> bool {
        if self.minimizing {
            value <= bound
        } else {
            value >= bound
        }
    }
}

impl Propagator for ObjExprBound {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let c = self.bound.load(Ordering::Relaxed);
        let (lo, hi) = self.guarded_sum.as_ref().map_or_else(
            || {
                let dom = |x: VarId| (store.min(x) as i64, store.max(x) as i64);
                self.expr.bounds(&dom)
            },
            |guarded| guarded.bounds(store),
        );
        if self.impossible(lo, hi, c) {
            return Err(Inconsistency); // can never hold
        }
        if self.entailed(lo, hi, c) {
            return Ok(()); // entailed
        }

        let probe_work = self
            .vars
            .iter()
            .try_fold(0usize, |work, &variable| work.checked_add(store.size(variable)))
            .and_then(|values| values.checked_mul(self.probe_cost_per_value));
        if probe_work.is_some_and(|work| work <= MAX_EXPR_SINGLETON_PROBE_WORK) {
            for &v in &self.vars {
                if store.is_fixed(v) {
                    continue;
                }
                self.scratch.clear();
                self.scratch.extend(store.values(v));
                for &val in &self.scratch {
                    let dead = self.guarded_sum.as_ref().map_or_else(
                        || {
                            let dom = |x: VarId| {
                                if x == v {
                                    (val as i64, val as i64)
                                } else {
                                    (store.min(x) as i64, store.max(x) as i64)
                                }
                            };
                            let (lo, hi) = self.expr.bounds(&dom);
                            self.impossible(lo, hi, c)
                        },
                        |guarded| {
                            let (lo, hi) = guarded.bounds_with_value(store, v, val);
                            self.impossible(lo, hi, c)
                        },
                    );
                    if dead {
                        store.remove(v, val)?;
                    }
                }
            }
        }

        if self.vars.iter().all(|&x| store.is_fixed(x)) {
            match self.expr.eval(&|x| store.value(x) as i64) {
                Some(n) if self.accepts(n, c) => {}
                _ => return Err(Inconsistency),
            }
        }
        Ok(())
    }
}

fn clamp_i128_i32(value: i128) -> i32 {
    value.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
}

/// Floor of `a / b` for integers (Rust `/` truncates toward zero).
fn floor_div(a: i128, b: i128) -> i128 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// Ceiling of `a / b` for integers.
fn ceil_div(a: i128, b: i128) -> i128 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) == (b < 0)) {
        q + 1
    } else {
        q
    }
}
