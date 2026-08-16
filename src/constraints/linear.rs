//! Linear (weighted-sum) constraints. All arithmetic in `i64` so sums of `i32`
//! terms cannot overflow.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Priority, Propagator};
use crate::store::{Premise, Solver, Store};

/// Sentinel for "exclude no term" (conflict reason: every term contributes).
const NO_SKIP: usize = usize::MAX;

/// Comparison operator for a linear constraint.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Relation {
    /// \( \sum_i a_i x_i = \texttt{rhs} \)
    Eq,
    /// \( \sum_i a_i x_i \ne \texttt{rhs} \)
    Ne,
    /// \( \sum_i a_i x_i \le \texttt{rhs} \)
    Le,
    /// \( \sum_i a_i x_i < \texttt{rhs} \)
    Lt,
    /// \( \sum_i a_i x_i \ge \texttt{rhs} \)
    Ge,
    /// \( \sum_i a_i x_i > \texttt{rhs} \)
    Gt,
}

/// \( \sum_i a_i x_i \le c \), bounds propagation to a local fixpoint per call.
#[derive(Clone)]
struct LinearLeq {
    coeffs: Vec<i64>,
    vars: Vec<VarId>,
    c: i64,
    /// Reused scratch for per-term minimum contributions.
    term_min: Vec<i64>,
}

impl LinearLeq {
    fn new(coeffs: &[i64], vars: &[VarId], c: i64) -> Self {
        debug_assert_eq!(coeffs.len(), vars.len());
        Self { coeffs: coeffs.to_vec(), vars: vars.to_vec(), c, term_min: vec![0; vars.len()] }
    }

    /// Near bound of every term but `skip` (`x_j ≥ min` if `a_j > 0`, else
    /// `x_j ≤ max`); entails `∑_{j≠skip} a_j x_j ≥ sum_min − term_min[skip]`.
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

impl Propagator for LinearLeq {
    fn priority(&self) -> Priority {
        Priority::Linear
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::BoundChange);
        }
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        register_variables_until(store, me, &self.vars, Event::BoundChange, should_stop)
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        loop {
            let mut sum_min: i64 = 0;
            let mut sum_max: i64 = 0;
            for (slot, (&a, &v)) in self.term_min.iter_mut().zip(self.coeffs.iter().zip(&self.vars)) {
                let lo = store.min(v) as i64;
                let hi = store.max(v) as i64;
                let (tmin, tmax) = if a >= 0 { (a * lo, a * hi) } else { (a * hi, a * lo) };
                *slot = tmin;
                sum_min += tmin;
                sum_max += tmax;
            }

            if sum_min > self.c {
                return Err(store.fail_because(self.min_side(store, NO_SKIP)));
            }
            if sum_max <= self.c {
                return Ok(()); // entailed
            }

            let mut changed = false;
            for (idx, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
                if a == 0 {
                    continue;
                }
                let allowed = self.c - (sum_min - self.term_min[idx]);
                if a > 0 {
                    let bound = clamp_i32(floor_div(allowed, a));
                    let before = store.max(v);
                    if bound < before {
                        store.remove_above_because(v, bound, self.min_side(store, idx))?;
                        changed = true;
                    }
                } else {
                    let bound = clamp_i32(ceil_div(allowed, a));
                    let before = store.min(v);
                    if bound > before {
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

/// \( \sum_i a_i x_i = c \), both-sided bounds propagation to a local fixpoint.
#[derive(Clone)]
struct LinearEq {
    coeffs: Vec<i64>,
    vars: Vec<VarId>,
    c: i64,
    term_min: Vec<i64>,
    term_max: Vec<i64>,
}

impl LinearEq {
    fn new(coeffs: &[i64], vars: &[VarId], c: i64) -> Self {
        debug_assert_eq!(coeffs.len(), vars.len());
        Self { coeffs: coeffs.to_vec(), vars: vars.to_vec(), c, term_min: vec![0; vars.len()], term_max: vec![0; vars.len()] }
    }

    /// Near bound of every term but `skip`; entails
    /// `∑_{j≠skip} a_j x_j ≥ sum_min − term_min[skip]`.
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

    /// Far bound of every term but `skip`; entails
    /// `∑_{j≠skip} a_j x_j ≤ sum_max − term_max[skip]`.
    fn max_side(&self, store: &Store, skip: usize) -> Vec<Premise> {
        let mut why = Vec::new();
        for (j, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
            if j == skip || a == 0 {
                continue;
            }
            why.push(if a > 0 { Premise::Le { var: v, bound: store.max(v) } } else { Premise::Ge { var: v, bound: store.min(v) } });
        }
        why
    }
}

impl Propagator for LinearEq {
    fn priority(&self) -> Priority {
        Priority::Linear
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::BoundChange);
        }
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        register_variables_until(store, me, &self.vars, Event::BoundChange, should_stop)
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        loop {
            let mut sum_min: i64 = 0;
            let mut sum_max: i64 = 0;
            for (i, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
                let lo = store.min(v) as i64;
                let hi = store.max(v) as i64;
                let (tmin, tmax) = if a >= 0 { (a * lo, a * hi) } else { (a * hi, a * lo) };
                self.term_min[i] = tmin;
                self.term_max[i] = tmax;
                sum_min += tmin;
                sum_max += tmax;
            }
            if sum_min > self.c {
                return Err(store.fail_because(self.min_side(store, NO_SKIP)));
            }
            if sum_max < self.c {
                return Err(store.fail_because(self.max_side(store, NO_SKIP)));
            }

            let mut changed = false;
            for (i, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
                if a == 0 {
                    continue;
                }
                // a_i*x_i in [tlo, thi]; sign of a_i decides x_i's lo vs hi bound.
                let tlo = self.c - (sum_max - self.term_max[i]);
                let thi = self.c - (sum_min - self.term_min[i]);
                let (lo, hi) = if a > 0 { (ceil_div(tlo, a), floor_div(thi, a)) } else { (ceil_div(thi, a), floor_div(tlo, a)) };
                let before_min = store.min(v);
                let before_max = store.max(v);
                let lo = clamp_i32(lo);
                let hi = clamp_i32(hi);
                if lo > before_min {
                    let why = if a > 0 { self.max_side(store, i) } else { self.min_side(store, i) };
                    store.remove_below_because(v, lo, why)?;
                    changed = true;
                }
                if hi < before_max {
                    let why = if a > 0 { self.min_side(store, i) } else { self.max_side(store, i) };
                    store.remove_above_because(v, hi, why)?;
                    changed = true;
                }
            }
            if !changed {
                return Ok(());
            }
        }
    }
}

/// \( \sum_i a_i x_i \ne c \). Weak: only filters when at most one variable is still free.
#[derive(Clone)]
struct LinearNeq {
    coeffs: Vec<i64>,
    vars: Vec<VarId>,
    c: i64,
}

impl LinearNeq {
    fn new(coeffs: &[i64], vars: &[VarId], c: i64) -> Self {
        debug_assert_eq!(coeffs.len(), vars.len());
        Self { coeffs: coeffs.to_vec(), vars: vars.to_vec(), c }
    }
}

impl Propagator for LinearNeq {
    fn priority(&self) -> Priority {
        Priority::Linear
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::Fix);
        }
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        register_variables_until(store, me, &self.vars, Event::Fix, should_stop)
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let mut fixed_sum: i64 = 0;
        let mut free: Option<usize> = None;
        let mut free_count = 0usize;
        for (idx, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
            if store.is_fixed(v) {
                fixed_sum += a * store.value(v) as i64;
            } else {
                free = Some(idx);
                free_count += 1;
            }
        }

        match free_count {
            0 => {
                if fixed_sum == self.c {
                    return Err(Inconsistency);
                }
            }
            1 => {
                let idx = free.unwrap();
                let a = self.coeffs[idx];
                if a != 0 {
                    // a*x = c - fixed_sum: forbid x only if the quotient is integral.
                    let rem = self.c - fixed_sum;
                    if rem % a == 0 {
                        let forbidden = rem / a;
                        if (i32::MIN as i64..=i32::MAX as i64).contains(&forbidden) {
                            store.remove(self.vars[idx], forbidden as i32)?;
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PredicateTruth {
    True,
    False,
    Unknown,
}

/// A Boolean selector guarding an affine predicate. With `equivalence=false`,
/// only `selector == active_value` implies the predicate. With equivalence, the
/// opposite selector value enforces the exact complementary relation too.
#[derive(Clone)]
struct ReifiedLinear {
    selector: VarId,
    active_value: i32,
    equivalence: bool,
    coefficients: Vec<i64>,
    variables: Vec<VarId>,
    relation: Relation,
    rhs: i64,
    when_true: Box<dyn Propagator>,
    when_false: Option<Box<dyn Propagator>>,
}

impl ReifiedLinear {
    fn activity(&self, store: &Store) -> (i128, i128) {
        self.coefficients.iter().zip(&self.variables).fold((0i128, 0i128), |(minimum, maximum), (&coefficient, &variable)| {
            let coefficient = i128::from(coefficient);
            let lo = i128::from(store.min(variable));
            let hi = i128::from(store.max(variable));
            let (term_min, term_max) =
                if coefficient >= 0 { (coefficient * lo, coefficient * hi) } else { (coefficient * hi, coefficient * lo) };
            (minimum + term_min, maximum + term_max)
        })
    }

    fn truth(&self, store: &Store) -> PredicateTruth {
        let (minimum, maximum) = self.activity(store);
        let rhs = i128::from(self.rhs);
        match self.relation {
            Relation::Eq if minimum == maximum && minimum == rhs => PredicateTruth::True,
            Relation::Eq if rhs < minimum || rhs > maximum => PredicateTruth::False,
            Relation::Ne if rhs < minimum || rhs > maximum => PredicateTruth::True,
            Relation::Ne if minimum == maximum && minimum == rhs => PredicateTruth::False,
            Relation::Le if maximum <= rhs => PredicateTruth::True,
            Relation::Le if minimum > rhs => PredicateTruth::False,
            Relation::Lt if maximum < rhs => PredicateTruth::True,
            Relation::Lt if minimum >= rhs => PredicateTruth::False,
            Relation::Ge if minimum >= rhs => PredicateTruth::True,
            Relation::Ge if maximum < rhs => PredicateTruth::False,
            Relation::Gt if minimum > rhs => PredicateTruth::True,
            Relation::Gt if maximum <= rhs => PredicateTruth::False,
            _ => PredicateTruth::Unknown,
        }
    }

    fn reason(&self, store: &Store) -> Vec<Premise> {
        let mut reason = Vec::new();
        for &variable in &self.variables {
            store.domain_premises(variable, &mut reason);
        }
        reason
    }

    fn propagate_active(&mut self, store: &mut Store, should_stop: &dyn Fn() -> bool) -> Result<(), Inconsistency> {
        if should_stop() {
            return Ok(());
        }
        let value = store.value(self.selector);
        let inner = if value == self.active_value {
            &mut self.when_true
        } else if self.equivalence {
            self.when_false.as_mut().expect("equivalent reification owns a complement propagator")
        } else {
            return Ok(());
        };
        let premise = Premise::Eq { var: self.selector, val: value };
        store.set_injected_premise(premise);
        let result = inner.propagate_until(store, should_stop);
        if result.is_err() {
            store.push_conflict_premise(premise);
        }
        store.clear_injected_premise();
        result
    }
}

impl Propagator for ReifiedLinear {
    fn priority(&self) -> Priority {
        Priority::Linear
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.selector, me, Event::DomainChange);
        for &variable in &self.variables {
            store.subscribe(variable, me, Event::DomainChange);
        }
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        if should_stop() {
            return false;
        }
        store.subscribe(self.selector, me, Event::DomainChange);
        register_variables_until(store, me, &self.variables, Event::DomainChange, should_stop)
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        self.propagate_until(store, &|| false)
    }

    fn propagate_until(&mut self, store: &mut Store, should_stop: &dyn Fn() -> bool) -> Result<(), Inconsistency> {
        if should_stop() {
            return Ok(());
        }
        if store.is_fixed(self.selector) {
            return self.propagate_active(store, should_stop);
        }
        let value = match self.truth(store) {
            PredicateTruth::True if self.equivalence => Some(self.active_value),
            PredicateTruth::False => Some(1 - self.active_value),
            PredicateTruth::True | PredicateTruth::Unknown => None,
        };
        if let Some(value) = value {
            let reason = self.reason(store);
            store.fix_because(self.selector, value, reason)?;
        }
        Ok(())
    }
}

/// Floor of `a / b` for integers (Rust `/` truncates toward zero).
fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// Ceiling of `a / b` for integers.
fn ceil_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) == (b < 0)) {
        q + 1
    } else {
        q
    }
}

/// Saturate an `i64` bound into the `i32` domain range.
pub(crate) fn clamp_i32(x: i64) -> i32 {
    x.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

fn neg(coeffs: &[i64]) -> Vec<i64> {
    coeffs.iter().map(|&a| -a).collect()
}

fn complement(relation: Relation) -> Relation {
    match relation {
        Relation::Eq => Relation::Ne,
        Relation::Ne => Relation::Eq,
        Relation::Le => Relation::Gt,
        Relation::Lt => Relation::Ge,
        Relation::Ge => Relation::Lt,
        Relation::Gt => Relation::Le,
    }
}

fn boxed_linear_checked(mut coefficients: Vec<i64>, variables: Vec<VarId>, relation: Relation, rhs: i64) -> Option<Box<dyn Propagator>> {
    Some(match relation {
        Relation::Le => Box::new(LinearLeq { term_min: vec![0; variables.len()], coeffs: coefficients, vars: variables, c: rhs }),
        Relation::Lt => {
            Box::new(LinearLeq { term_min: vec![0; variables.len()], coeffs: coefficients, vars: variables, c: rhs.checked_sub(1)? })
        }
        Relation::Ge | Relation::Gt => {
            for coefficient in &mut coefficients {
                *coefficient = coefficient.checked_neg()?;
            }
            let bound = if relation == Relation::Ge { rhs.checked_neg()? } else { rhs.checked_add(1)?.checked_neg()? };
            Box::new(LinearLeq { term_min: vec![0; variables.len()], coeffs: coefficients, vars: variables, c: bound })
        }
        Relation::Eq => Box::new(LinearEq {
            term_min: vec![0; variables.len()],
            term_max: vec![0; variables.len()],
            coeffs: coefficients,
            vars: variables,
            c: rhs,
        }),
        Relation::Ne => Box::new(LinearNeq { coeffs: coefficients, vars: variables, c: rhs }),
    })
}

/// Whether the existing native linear propagators can represent this row
/// without an intermediate `i64` overflow.
pub(crate) fn native_supported(store: &Store, coefficients: &[i64], variables: &[VarId], relation: Relation, rhs: i64) -> bool {
    if coefficients.len() != variables.len() {
        return false;
    }
    let activity_fits = coefficients
        .iter()
        .zip(variables)
        .try_fold((0i64, 0i64), |(minimum, maximum), (&coefficient, &variable)| {
            let lo = i64::from(store.min(variable));
            let hi = i64::from(store.max(variable));
            let (term_min, term_max) = if coefficient >= 0 {
                (coefficient.checked_mul(lo)?, coefficient.checked_mul(hi)?)
            } else {
                (coefficient.checked_mul(hi)?, coefficient.checked_mul(lo)?)
            };
            Some((minimum.checked_add(term_min)?, maximum.checked_add(term_max)?))
        })
        .is_some();
    activity_fits && boxed_linear_checked(coefficients.to_vec(), variables.to_vec(), relation, rhs).is_some()
}

pub(crate) struct ReifiedLinearSpec {
    pub selector: VarId,
    pub active_value: i32,
    pub coefficients: Vec<i64>,
    pub variables: Vec<VarId>,
    pub relation: Relation,
    pub rhs: i64,
    pub equivalence: bool,
}

/// Post an implication or equivalence between a Boolean literal and an affine
/// predicate using native linear propagators only.
pub(crate) fn reified_linear_interruptible(solver: &mut Solver, spec: ReifiedLinearSpec, stop: &AtomicBool) -> bool {
    let ReifiedLinearSpec { selector, active_value, coefficients, variables, relation, rhs, equivalence } = spec;
    if stop.load(Ordering::Acquire)
        || !matches!(active_value, 0 | 1)
        || !native_supported(&solver.store, &coefficients, &variables, relation, rhs)
        || (equivalence && !native_supported(&solver.store, &coefficients, &variables, complement(relation), rhs))
    {
        return false;
    }
    let when_true = boxed_linear_checked(coefficients.clone(), variables.clone(), relation, rhs)
        .expect("native support was checked before building the reified predicate");
    let when_false = equivalence.then(|| {
        boxed_linear_checked(coefficients.clone(), variables.clone(), complement(relation), rhs)
            .expect("native support was checked before building the reified complement")
    });
    let propagator = ReifiedLinear { selector, active_value, equivalence, coefficients, variables, relation, rhs, when_true, when_false };
    let should_stop = || stop.load(Ordering::Acquire);
    solver.post_until(Box::new(propagator), &should_stop).is_some()
}

fn register_variables_until(store: &mut Store, me: PropId, variables: &[VarId], event: Event, should_stop: &dyn Fn() -> bool) -> bool {
    for &variable in variables {
        if should_stop() {
            return false;
        }
        store.subscribe(variable, me, event);
    }
    !should_stop()
}

fn post_leq(solver: &mut Solver, coeffs: &[i64], vars: &[VarId], c: i64) {
    solver.post(Box::new(LinearLeq::new(coeffs, vars, c)));
}

/// Post \( \sum_i \texttt{coeffs}[i] \cdot \texttt{vars}[i] \;\texttt{rel}\; \texttt{rhs} \).
pub fn linear(solver: &mut Solver, coeffs: &[i64], vars: &[VarId], rel: Relation, rhs: i64) {
    assert_eq!(coeffs.len(), vars.len(), "linear: coeffs/vars length mismatch");
    record_relaxation(solver, coeffs, vars, rel, rhs);
    match rel {
        Relation::Le => post_leq(solver, coeffs, vars, rhs),
        Relation::Lt => post_leq(solver, coeffs, vars, rhs - 1),
        Relation::Ge => post_leq(solver, &neg(coeffs), vars, -rhs),
        Relation::Gt => post_leq(solver, &neg(coeffs), vars, -(rhs + 1)),
        Relation::Eq => {
            solver.post(Box::new(LinearEq::new(coeffs, vars, rhs)));
        }
        Relation::Ne => {
            solver.post(Box::new(LinearNeq::new(coeffs, vars, rhs)));
        }
    }
}

/// Owned-input interruptible posting used by physical compilers. A `false`
/// result requires discarding `solver`.
pub(crate) fn linear_interruptible(
    solver: &mut Solver,
    coeffs: Vec<i64>,
    vars: Vec<VarId>,
    rel: Relation,
    rhs: i64,
    stop: &AtomicBool,
) -> bool {
    linear_interruptible_with_relaxation(solver, coeffs, vars, rel, rhs, stop, true)
}

/// Post a row that has already been represented in the relaxation by its
/// semantic source. This keeps physical normalization from duplicating LP rows.
pub(crate) fn linear_without_relaxation_interruptible(
    solver: &mut Solver,
    coeffs: Vec<i64>,
    vars: Vec<VarId>,
    rel: Relation,
    rhs: i64,
    stop: &AtomicBool,
) -> bool {
    linear_interruptible_with_relaxation(solver, coeffs, vars, rel, rhs, stop, false)
}

fn linear_interruptible_with_relaxation(
    solver: &mut Solver,
    mut coeffs: Vec<i64>,
    vars: Vec<VarId>,
    rel: Relation,
    rhs: i64,
    stop: &AtomicBool,
    record: bool,
) -> bool {
    assert_eq!(coeffs.len(), vars.len(), "linear: coeffs/vars length mismatch");
    if stop.load(Ordering::Acquire) {
        return false;
    }
    if record {
        record_relaxation(solver, &coeffs, &vars, rel, rhs);
    }
    let should_stop = || stop.load(Ordering::Acquire);
    let propagator: Box<dyn Propagator> = match rel {
        Relation::Le => Box::new(LinearLeq { term_min: vec![0; vars.len()], coeffs, vars, c: rhs }),
        Relation::Lt => Box::new(LinearLeq { term_min: vec![0; vars.len()], coeffs, vars, c: rhs - 1 }),
        Relation::Ge | Relation::Gt => {
            for coefficient in &mut coeffs {
                if stop.load(Ordering::Acquire) {
                    return false;
                }
                *coefficient = -*coefficient;
            }
            let bound = if rel == Relation::Ge { -rhs } else { -(rhs + 1) };
            Box::new(LinearLeq { term_min: vec![0; vars.len()], coeffs, vars, c: bound })
        }
        Relation::Eq => Box::new(LinearEq { term_min: vec![0; vars.len()], term_max: vec![0; vars.len()], coeffs, vars, c: rhs }),
        Relation::Ne => Box::new(LinearNeq { coeffs, vars, c: rhs }),
    };
    if stop.load(Ordering::Acquire) {
        return false;
    }
    solver.post_until(propagator, &should_stop).is_some()
}

#[cfg(feature = "lp-relaxation")]
pub(crate) fn record_relaxation(solver: &mut Solver, coefficients: &[i64], variables: &[VarId], relation: Relation, rhs: i64) {
    match relation {
        Relation::Le => solver.record_linear_relaxation(coefficients, variables, None, Some(rhs)),
        Relation::Lt => {
            if let Some(upper) = rhs.checked_sub(1) {
                solver.record_linear_relaxation(coefficients, variables, None, Some(upper));
            }
        }
        Relation::Ge => solver.record_linear_relaxation(coefficients, variables, Some(rhs), None),
        Relation::Gt => {
            if let Some(lower) = rhs.checked_add(1) {
                solver.record_linear_relaxation(coefficients, variables, Some(lower), None);
            }
        }
        Relation::Eq => solver.record_linear_relaxation(coefficients, variables, Some(rhs), Some(rhs)),
        Relation::Ne => {}
    }
}

#[cfg(not(feature = "lp-relaxation"))]
#[inline]
pub(crate) fn record_relaxation(_solver: &mut Solver, _coefficients: &[i64], _variables: &[VarId], _relation: Relation, _rhs: i64) {}

/// Post \( \sum_i \texttt{vars}[i] \;\texttt{rel}\; \texttt{rhs} \) (all coefficients 1).
pub fn sum(solver: &mut Solver, vars: &[VarId], rel: Relation, rhs: i64) {
    let coeffs = vec![1i64; vars.len()];
    linear(solver, &coeffs, vars, rel, rhs);
}

pub(crate) fn sum_interruptible(solver: &mut Solver, vars: &[VarId], rel: Relation, rhs: i64, stop: &AtomicBool) -> bool {
    let mut coefficients = Vec::with_capacity(vars.len());
    let mut variables = Vec::with_capacity(vars.len());
    for &variable in vars {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        coefficients.push(1);
        variables.push(variable);
    }
    linear_interruptible(solver, coefficients, variables, rel, rhs, stop)
}
