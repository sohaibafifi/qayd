//! Incumbent-only local search for `--fast-cop` runs.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::constraints::linear::Relation;
use crate::constraints::table::{Dfa, Mdd, STAR};
use crate::expr::Expr;
use crate::ids::VarId;
use crate::mix64;
use crate::problem::{Objective, Problem};

const MAX_DOMAIN_VALUES: usize = 4096;
const MAX_SAMPLED_VARS: usize = 48;
const RANDOM_WALK_PERIOD: u64 = 17;
const RESTART_AFTER: u64 = 200;
const CONSTRUCTIVE_KICK_PERIOD: u64 = 5;
const WORD_PLACEMENT_NODE_LIMIT: usize = 20_000;
/// Min-conflicts (#1b): domains with at most this many values are scanned in full
/// (cheap and optimal); larger ones use the bounded candidate set.
const MIN_CONFLICTS_FULL: usize = 24;
/// Min-conflicts (#1b): random domain samples added to the candidate set on a
/// large domain, alongside the current value and any structural suggestions.
const MIN_CONFLICTS_SAMPLES: usize = 8;

#[derive(Clone)]
pub(crate) enum LocalConstraint {
    Expr(Expr),
    Linear {
        coeffs: Vec<i64>,
        vars: Vec<VarId>,
        rel: Relation,
        rhs: i64,
    },
    AllDifferent(Vec<VarId>),
    AllDifferentRows(Vec<Vec<VarId>>),
    AllDifferentExcept {
        vars: Vec<VarId>,
        except: Vec<i32>,
    },
    AllEqual(Vec<VarId>),
    Extension {
        vars: Vec<VarId>,
        tuples: Vec<Vec<i32>>,
    },
    /// Negative (conflict) table: the listed tuples are forbidden.
    NegExtension {
        vars: Vec<VarId>,
        tuples: Vec<Vec<i32>>,
    },
    Lex {
        rows: Vec<Vec<VarId>>,
        strict: bool,
    },
    Count {
        vars: Vec<VarId>,
        values: Vec<i32>,
        rel: Relation,
        rhs: LocalRhs,
    },
    CountAllowed {
        vars: Vec<VarId>,
        values: Vec<i32>,
        allowed: Vec<i32>,
    },
    NValues {
        vars: Vec<VarId>,
        rel: Relation,
        rhs: LocalRhs,
    },
    Cardinality {
        vars: Vec<VarId>,
        values: Vec<i32>,
        low: Vec<i64>,
        high: Vec<i64>,
        closed: bool,
    },
    Extremum {
        vars: Vec<VarId>,
        is_min: bool,
        rel: Relation,
        rhs: LocalRhs,
    },
    ElementMember {
        array: Vec<VarId>,
        value: i32,
    },
    Cumulative {
        starts: Vec<VarId>,
        durations: Vec<VarId>,
        heights: Vec<VarId>,
        cap: LocalRhs,
    },
    ChannelInverse {
        xs: Vec<VarId>,
        x_start: i32,
        ys: Vec<VarId>,
        y_start: i32,
    },
    ChannelOneHot {
        xs: Vec<VarId>,
        value: VarId,
        start_index: i32,
    },
    Precedence {
        vars: Vec<VarId>,
        values: Vec<i32>,
        covered: bool,
    },
    Circuit(Vec<VarId>),
    BinPacking {
        items: Vec<VarId>,
        sizes: Vec<i64>,
        limits: Vec<LocalRhs>,
        exact: bool,
    },
    NoOverlap {
        origins: Vec<Vec<VarId>>,
        lengths: Vec<Vec<Expr>>,
        zero_ignored: bool,
    },
    Regular {
        vars: Vec<VarId>,
        dfa: Dfa,
    },
    Mdd {
        vars: Vec<VarId>,
        mdd: Mdd,
    },
}

#[derive(Clone)]
enum Functional {
    Expr { target: VarId, expr: Expr },
    Linear { target: VarId, coeff: i64, terms: Vec<(i64, VarId)>, rhs: i64 },
    Element { target: VarId, array: Vec<VarId>, index: VarId, start_index: i32 },
    BoolTable { target: VarId, left: VarId, right: VarId, true_pairs: Vec<(i32, i32)> },
}

#[derive(Clone, Copy)]
pub(crate) enum LocalRhs {
    Const(i64),
    Var(VarId),
}

#[derive(Clone, Default)]
pub(crate) struct LocalSearchSpec {
    constraints: Vec<LocalConstraint>,
    functionals: Vec<Functional>,
    derived: Vec<bool>,
    unsupported: usize,
}

/// Behaviour toggles for the local-search engine. `--fast-cop` uses the default
/// (everything off: plain min-conflicts descent); `--turbo` switches on the
/// autonomous upgrades. See TURBO.md (Tier-B). New toggles are added here as
/// each Tier-B feature lands.
#[derive(Clone, Copy, Default)]
pub(crate) struct LsConfig {
    /// Guided Local Search: at a local minimum, penalise the still-violated
    /// constraints (bump their weights) so search is pushed off the plateau and
    /// toward the genuinely hard constraints. TURBO.md §4.2 / #2.
    pub(crate) gls: bool,
    /// Min-conflicts value selection: on large domains, evaluate a small candidate
    /// set (current value + structure-suggested values + random samples) instead
    /// of scanning the whole domain, so more variables are tried per iteration.
    /// Only bites when a domain exceeds `MIN_CONFLICTS_FULL`. TURBO.md #1b.
    pub(crate) min_conflicts: bool,
    /// Adaptive operator selection over the existing LS kicks. TURBO.md #5b.
    pub(crate) kick_bandit: bool,
}

pub(crate) struct LocalSearchOutcome {
    pub(crate) best: Option<(Vec<i32>, i64)>,
    pub(crate) iterations: u64,
    pub(crate) moves: u64,
    pub(crate) restarts: u64,
    pub(crate) constraints: usize,
    pub(crate) functionals: usize,
    pub(crate) unsupported: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Score {
    violation: i64,
    objective: i64,
}

// Restart is deliberately NOT a bandit operator — see the stagnation fallback in
// the search loop. These are the refine operators the bandit chooses among.
const KICK_OPERATOR_COUNT: usize = 3;

#[derive(Clone, Copy)]
enum KickOperator {
    Repair,
    Objective,
    Constructive,
}

impl KickOperator {
    fn index(self) -> usize {
        match self {
            Self::Repair => 0,
            Self::Objective => 1,
            Self::Constructive => 2,
        }
    }

    fn source(self) -> &'static str {
        match self {
            Self::Repair | Self::Objective => "local-search",
            Self::Constructive => "constructive",
        }
    }
}

#[derive(Clone)]
struct KickBandit {
    pulls: [u64; KICK_OPERATOR_COUNT],
    rewards: [f64; KICK_OPERATOR_COUNT],
    total: u64,
}

impl KickBandit {
    fn new() -> Self {
        Self { pulls: [0; KICK_OPERATOR_COUNT], rewards: [0.0; KICK_OPERATOR_COUNT], total: 0 }
    }

    fn select(&self, available: &[KickOperator], seed: u64, iter: u64) -> KickOperator {
        let untried = available.iter().copied().filter(|op| self.pulls[op.index()] == 0).collect::<Vec<_>>();
        if !untried.is_empty() {
            return untried[mix64(seed ^ iter ^ 0x5B) as usize % untried.len()];
        }
        let total = self.total.max(1) as f64;
        available
            .iter()
            .copied()
            .max_by(|&a, &b| {
                let sa = self.ucb_score(a, total);
                let sb = self.ucb_score(b, total);
                sa.total_cmp(&sb)
            })
            .unwrap_or(KickOperator::Repair)
    }

    fn record(&mut self, op: KickOperator, reward: f64) {
        let i = op.index();
        self.pulls[i] += 1;
        self.rewards[i] += reward;
        self.total += 1;
    }

    fn ucb_score(&self, op: KickOperator, total: f64) -> f64 {
        let i = op.index();
        let pulls = self.pulls[i] as f64;
        self.rewards[i] / pulls + (2.0 * total.ln() / pulls).sqrt()
    }
}

#[derive(Clone)]
struct LocalDomain {
    min: i32,
    max: i32,
    values: Vec<i32>,
}

impl LocalDomain {
    fn contains(&self, value: i32) -> bool {
        if self.values.is_empty() {
            self.min <= value && value <= self.max
        } else {
            self.values.contains(&value)
        }
    }

    fn initial_value(&self, seed: u64) -> i32 {
        if self.values.is_empty() {
            self.min
        } else {
            self.values[mix64(seed) as usize % self.values.len()]
        }
    }

    fn min_value(&self) -> i32 {
        if self.values.is_empty() {
            self.min
        } else {
            self.values.iter().copied().min().unwrap_or(self.min)
        }
    }

    fn max_value(&self) -> i32 {
        if self.values.is_empty() {
            self.max
        } else {
            self.values.iter().copied().max().unwrap_or(self.max)
        }
    }

    fn is_bool(&self) -> bool {
        self.contains(0) && self.contains(1) && self.min_value() == 0 && self.max_value() == 1
    }
}

struct LocalModel {
    domains: Vec<LocalDomain>,
    mutable: Vec<VarId>,
    search: Vec<VarId>,
    objective: Objective,
    constraints: Vec<LocalConstraint>,
    functionals: Vec<Functional>,
    bool_tables: HashMap<(VarId, VarId), Vec<(i32, i32)>>,
    exact_covers: Vec<Vec<VarId>>,
    /// Incidence index for incremental move scoring: `affected[v]` is the set of
    /// constraint indices whose violation depends on variable `v`, directly or
    /// transitively through functionals (a flipped `v` propagates to functional
    /// targets via `complete()`, so constraints over those targets are included).
    /// Built once in [`LocalModel::new`]; reused by the delta-scoring move loop.
    affected: Vec<Vec<usize>>,
}

// TODO(simplify): the guarded-word placement subsystem below (GuardedWord,
// WordPlacementState, try_place_word/word_trial/place_word_letters, guarded_*,
// place_guarded_elements) is ~250 LOC of crossword-specific constructive
// heuristic, fired only for the Extension+Element+guard pattern via
// has_extension(). It is the largest single simplification target in this file.
// Decide on bench evidence: if it does not measurably beat generic LS on the
// word/crossword instances, delete it and let those fall back to the generic
// constructive path. Until then it stays, gated and isolated.
struct GuardedWord<'a> {
    guard: VarId,
    weight: i64,
    array: &'a [VarId],
    letters: Vec<(VarId, i32)>,
}

struct WordPlacementState {
    values: Vec<Option<i32>>,
    cells: Vec<usize>,
    nodes: usize,
}

type ElementViews<'a> = HashMap<VarId, (&'a [VarId], VarId, i32)>;
type Placement = (VarId, i32);
type TrialPlacement = (Vec<i32>, Vec<Placement>);

fn contains_var(expr: &Expr, target: VarId) -> bool {
    let mut vars = Vec::new();
    expr.collect_vars(&mut vars);
    vars.contains(&target)
}

fn functional_from_expr(expr: &Expr, derived: &[bool]) -> Option<Functional> {
    let Expr::Eq(a, b) = expr else {
        return None;
    };
    match (&**a, &**b) {
        (Expr::Var(lhs), Expr::Var(rhs)) => {
            let lhs_derived = derived.get(lhs.index()).copied().unwrap_or(false);
            let rhs_derived = derived.get(rhs.index()).copied().unwrap_or(false);
            if lhs_derived && !rhs_derived {
                Some(Functional::Expr { target: *rhs, expr: Expr::Var(*lhs) })
            } else {
                Some(Functional::Expr { target: *lhs, expr: Expr::Var(*rhs) })
            }
        }
        (Expr::Var(target), value) if !contains_var(value, *target) => Some(Functional::Expr { target: *target, expr: value.clone() }),
        (value, Expr::Var(target)) if !contains_var(value, *target) => Some(Functional::Expr { target: *target, expr: value.clone() }),
        _ => None,
    }
}

fn functional_from_linear(coeffs: &[i64], vars: &[VarId], rel: Relation, rhs: i64, derived: &[bool]) -> Option<Functional> {
    if !matches!(rel, Relation::Eq) {
        return None;
    }
    let target_pos = if vars.len() == 1 && coeffs.first().is_some_and(|c| c.abs() == 1) {
        0
    } else {
        vars.iter()
            .enumerate()
            .filter(|(i, &var)| coeffs[*i] == -1 && !derived.get(var.index()).copied().unwrap_or(false))
            .max_by_key(|(_, &var)| var.index())
            .map(|(i, _)| i)?
    };
    let target = vars[target_pos];
    let coeff = coeffs[target_pos];
    let terms = coeffs.iter().zip(vars).enumerate().filter_map(|(i, (&c, &v))| (i != target_pos && c != 0).then_some((c, v))).collect();
    Some(Functional::Linear { target, coeff, terms, rhs })
}

fn functional_from_extension(vars: &[VarId], tuples: &[Vec<i32>]) -> Option<Functional> {
    let [left, right, target] = *vars else {
        return None;
    };
    let mut true_pairs = Vec::new();
    for tuple in tuples {
        let [a, b, value] = tuple.as_slice() else {
            return None;
        };
        match *value {
            0 => {}
            1 => true_pairs.push((*a, *b)),
            _ => return None,
        }
    }
    true_pairs.sort_unstable();
    true_pairs.dedup();
    Some(Functional::BoolTable { target, left, right, true_pairs })
}

fn relation_violation(lhs: i64, rel: Relation, rhs: i64) -> i64 {
    match rel {
        Relation::Eq => (lhs - rhs).abs(),
        Relation::Ne => i64::from(lhs == rhs),
        Relation::Le => (lhs - rhs).max(0),
        Relation::Lt => (lhs - rhs + 1).max(0),
        Relation::Ge => (rhs - lhs).max(0),
        Relation::Gt => (rhs - lhs + 1).max(0),
    }
}

/// Combine the incompleteness penalty and the (non-negative) sum of constraint
/// violations into a `Score.violation`, clamping at `i64::MAX` to match the
/// `saturating_add` of [`LocalModel::score_breakdown`].
fn combine_violation(penalty: i128, sum: i128) -> i64 {
    let total = penalty + sum;
    if total > i64::MAX as i128 {
        i64::MAX
    } else {
        total as i64
    }
}

/// GLS-weighted sum of per-constraint violations: `Σ wᵢ·violᵢ`.
fn weighted_sum(con_viol: &[i64], weights: &[i64]) -> i128 {
    con_viol.iter().zip(weights).map(|(&v, &w)| i128::from(v) * i128::from(w)).sum()
}

/// Min-conflicts (#1b) candidate-value set for `var` on a large domain: the current
/// value (the "stay" baseline), the value any single linear constraint over `var`
/// would need to hit its rhs boundary, and a few random domain samples. Every value
/// emitted is a genuine domain member (checked against `value_set`), so the caller
/// can score it with the same exact delta as a full scan.
fn min_conflict_candidates(
    model: &LocalModel,
    var: VarId,
    assignment: &[i32],
    value_set: &HashSet<i32>,
    seed: u64,
    iter: u64,
    out: &mut Vec<i32>,
) {
    out.clear();
    let j = var.index();
    out.push(assignment[j]);
    for &c in &model.affected[j] {
        if let LocalConstraint::Linear { coeffs, vars, rhs, .. } = &model.constraints[c] {
            if let Some(p) = vars.iter().position(|&v| v == var) {
                let a = coeffs[p];
                if a != 0 {
                    let rest: i64 = coeffs
                        .iter()
                        .zip(vars)
                        .enumerate()
                        .filter(|(i, _)| *i != p)
                        .map(|(_, (&co, &v))| co * i64::from(assignment[v.index()]))
                        .sum();
                    let target = *rhs - rest;
                    if target % a == 0 {
                        if let Ok(v) = i32::try_from(target / a) {
                            if value_set.contains(&v) {
                                out.push(v);
                            }
                        }
                    }
                }
            }
        }
    }
    let values = &model.domains[j].values;
    if !values.is_empty() {
        for k in 0..MIN_CONFLICTS_SAMPLES as u64 {
            let pick = mix64(seed ^ iter.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add((j as u64) << 1).wrapping_add(k));
            out.push(values[(pick as usize) % values.len()]);
        }
    }
    out.sort_unstable();
    out.dedup();
}

/// Append every variable whose value a constraint's violation reads.
fn constraint_vars(constraint: &LocalConstraint, out: &mut Vec<VarId>) {
    match constraint {
        LocalConstraint::Expr(expr) => expr.collect_vars(out),
        LocalConstraint::Linear { vars, .. }
        | LocalConstraint::AllDifferent(vars)
        | LocalConstraint::AllDifferentExcept { vars, .. }
        | LocalConstraint::AllEqual(vars)
        | LocalConstraint::Extension { vars, .. }
        | LocalConstraint::NegExtension { vars, .. }
        | LocalConstraint::CountAllowed { vars, .. }
        | LocalConstraint::Cardinality { vars, .. }
        | LocalConstraint::ElementMember { array: vars, .. }
        | LocalConstraint::Precedence { vars, .. }
        | LocalConstraint::Circuit(vars)
        | LocalConstraint::Regular { vars, .. }
        | LocalConstraint::Mdd { vars, .. } => out.extend(vars.iter().copied()),
        LocalConstraint::AllDifferentRows(rows) | LocalConstraint::Lex { rows, .. } => {
            for row in rows {
                out.extend(row.iter().copied());
            }
        }
        LocalConstraint::Count { vars, rhs, .. }
        | LocalConstraint::NValues { vars, rhs, .. }
        | LocalConstraint::Extremum { vars, rhs, .. } => {
            out.extend(vars.iter().copied());
            if let LocalRhs::Var(v) = rhs {
                out.push(*v);
            }
        }
        LocalConstraint::Cumulative { starts, durations, heights, cap } => {
            out.extend(starts.iter().copied());
            out.extend(durations.iter().copied());
            out.extend(heights.iter().copied());
            if let LocalRhs::Var(v) = cap {
                out.push(*v);
            }
        }
        LocalConstraint::ChannelInverse { xs, ys, .. } => {
            out.extend(xs.iter().copied());
            out.extend(ys.iter().copied());
        }
        LocalConstraint::ChannelOneHot { xs, value, .. } => {
            out.extend(xs.iter().copied());
            out.push(*value);
        }
        LocalConstraint::BinPacking { items, limits, .. } => {
            out.extend(items.iter().copied());
            for limit in limits {
                if let LocalRhs::Var(v) = limit {
                    out.push(*v);
                }
            }
        }
        LocalConstraint::NoOverlap { origins, lengths, .. } => {
            for row in origins {
                out.extend(row.iter().copied());
            }
            for row in lengths {
                for expr in row {
                    expr.collect_vars(out);
                }
            }
        }
    }
}

/// Append the input variables a functional reads to compute its target.
fn functional_inputs(functional: &Functional, out: &mut Vec<VarId>) {
    match functional {
        Functional::Expr { expr, .. } => expr.collect_vars(out),
        Functional::Linear { terms, .. } => out.extend(terms.iter().map(|&(_, v)| v)),
        Functional::Element { array, index, .. } => {
            out.push(*index);
            out.extend(array.iter().copied());
        }
        Functional::BoolTable { left, right, .. } => {
            out.push(*left);
            out.push(*right);
        }
    }
}

fn functional_target(functional: &Functional) -> VarId {
    match functional {
        Functional::Expr { target, .. }
        | Functional::Linear { target, .. }
        | Functional::Element { target, .. }
        | Functional::BoolTable { target, .. } => *target,
    }
}

/// Build the [`LocalModel::affected`] incidence index. Seed it with each
/// constraint's direct variable scope, then walk functionals in reverse
/// topological order (`functionals` is in forward topo order for `complete()`'s
/// single pass) pushing each target's dependent-constraint set back onto the
/// input variables that determine it. The result: `affected[v]` lists every
/// constraint whose violation can change when `v` is flipped.
fn build_affected(num_vars: usize, constraints: &[LocalConstraint], functionals: &[Functional]) -> Vec<Vec<usize>> {
    let mut affected: Vec<Vec<usize>> = vec![Vec::new(); num_vars];
    let mut scratch = Vec::new();
    for (ci, constraint) in constraints.iter().enumerate() {
        scratch.clear();
        constraint_vars(constraint, &mut scratch);
        for &v in &scratch {
            if v.index() < num_vars {
                affected[v.index()].push(ci);
            }
        }
    }
    for functional in functionals.iter().rev() {
        let target = functional_target(functional);
        if target.index() >= num_vars {
            continue;
        }
        let dependents = affected[target.index()].clone();
        if dependents.is_empty() {
            continue;
        }
        scratch.clear();
        functional_inputs(functional, &mut scratch);
        for &input in &scratch {
            if input.index() < num_vars {
                affected[input.index()].extend(dependents.iter().copied());
            }
        }
    }
    for entry in &mut affected {
        entry.sort_unstable();
        entry.dedup();
    }
    affected
}

fn order_functionals(mut functionals: Vec<Functional>) -> Vec<Functional> {
    let mut ordered = Vec::with_capacity(functionals.len());
    while !functionals.is_empty() {
        let remaining_targets = functionals.iter().map(functional_target).collect::<HashSet<_>>();
        let ready = functionals.iter().position(|functional| {
            let mut inputs = Vec::new();
            functional_inputs(functional, &mut inputs);
            inputs.into_iter().all(|input| !remaining_targets.contains(&input))
        });
        match ready {
            Some(index) => ordered.push(functionals.remove(index)),
            None => {
                ordered.extend(functionals);
                break;
            }
        }
    }
    ordered
}

impl LocalRhs {
    fn value(self, assignment: &[i32]) -> i64 {
        match self {
            Self::Const(value) => value,
            Self::Var(var) => assignment[var.index()] as i64,
        }
    }
}

impl LocalSearchSpec {
    pub(crate) fn add_var(&mut self, var: VarId) {
        self.ensure(var);
    }

    pub(crate) fn add_expr(&mut self, expr: Expr) {
        if let Some(functional) = functional_from_expr(&expr, &self.derived) {
            self.mark_functional(functional);
        }
        self.constraints.push(LocalConstraint::Expr(expr));
    }

    pub(crate) fn add_linear(&mut self, coeffs: Vec<i64>, vars: Vec<VarId>, rel: Relation, rhs: i64) {
        if let Some(functional) = functional_from_linear(&coeffs, &vars, rel, rhs, &self.derived) {
            self.mark_functional(functional);
        }
        self.constraints.push(LocalConstraint::Linear { coeffs, vars, rel, rhs });
    }

    pub(crate) fn add_all_different(&mut self, vars: Vec<VarId>) {
        self.constraints.push(LocalConstraint::AllDifferent(vars));
    }

    pub(crate) fn add_all_different_rows(&mut self, rows: Vec<Vec<VarId>>) {
        self.constraints.push(LocalConstraint::AllDifferentRows(rows));
    }

    pub(crate) fn add_all_different_except(&mut self, vars: Vec<VarId>, except: Vec<i32>) {
        self.constraints.push(LocalConstraint::AllDifferentExcept { vars, except });
    }

    pub(crate) fn add_all_equal(&mut self, vars: Vec<VarId>) {
        self.constraints.push(LocalConstraint::AllEqual(vars));
    }

    pub(crate) fn add_extension(&mut self, vars: Vec<VarId>, tuples: Vec<Vec<i32>>, positive: bool) {
        if positive {
            if let Some(functional) = functional_from_extension(&vars, &tuples) {
                self.mark_functional(functional);
            }
            self.constraints.push(LocalConstraint::Extension { vars, tuples });
        } else {
            self.constraints.push(LocalConstraint::NegExtension { vars, tuples });
        }
    }

    pub(crate) fn add_lex_chain(&mut self, rows: Vec<Vec<VarId>>, strict: bool) {
        self.constraints.push(LocalConstraint::Lex { rows, strict });
    }

    pub(crate) fn add_count(&mut self, vars: Vec<VarId>, values: Vec<i32>, rel: Relation, rhs: LocalRhs) {
        self.constraints.push(LocalConstraint::Count { vars, values, rel, rhs });
    }

    pub(crate) fn add_count_allowed(&mut self, vars: Vec<VarId>, values: Vec<i32>, allowed: Vec<i32>) {
        self.constraints.push(LocalConstraint::CountAllowed { vars, values, allowed });
    }

    pub(crate) fn add_n_values(&mut self, vars: Vec<VarId>, rel: Relation, rhs: LocalRhs) {
        self.constraints.push(LocalConstraint::NValues { vars, rel, rhs });
    }

    pub(crate) fn add_cardinality(&mut self, vars: Vec<VarId>, values: Vec<i32>, low: Vec<i64>, high: Vec<i64>, closed: bool) {
        self.constraints.push(LocalConstraint::Cardinality { vars, values, low, high, closed });
    }

    pub(crate) fn add_extremum(&mut self, vars: Vec<VarId>, is_min: bool, rel: Relation, rhs: LocalRhs) {
        self.constraints.push(LocalConstraint::Extremum { vars, is_min, rel, rhs });
    }

    pub(crate) fn add_element_member(&mut self, array: Vec<VarId>, value: i32) {
        self.constraints.push(LocalConstraint::ElementMember { array, value });
    }

    pub(crate) fn add_element(&mut self, array: Vec<VarId>, index: VarId, target: VarId, start_index: i32) {
        self.mark_functional(Functional::Element { target, array, index, start_index });
    }

    pub(crate) fn add_cumulative(&mut self, starts: Vec<VarId>, durations: Vec<VarId>, heights: Vec<VarId>, cap: LocalRhs) {
        self.constraints.push(LocalConstraint::Cumulative { starts, durations, heights, cap });
    }

    pub(crate) fn add_channel_inverse(&mut self, xs: Vec<VarId>, x_start: i32, ys: Vec<VarId>, y_start: i32) {
        self.constraints.push(LocalConstraint::ChannelInverse { xs, x_start, ys, y_start });
    }

    pub(crate) fn add_channel_onehot(&mut self, xs: Vec<VarId>, value: VarId, start_index: i32) {
        self.constraints.push(LocalConstraint::ChannelOneHot { xs, value, start_index });
    }

    pub(crate) fn add_precedence(&mut self, vars: Vec<VarId>, values: Vec<i32>, covered: bool) {
        self.constraints.push(LocalConstraint::Precedence { vars, values, covered });
    }

    pub(crate) fn add_circuit(&mut self, vars: Vec<VarId>) {
        self.constraints.push(LocalConstraint::Circuit(vars));
    }

    pub(crate) fn add_bin_packing(&mut self, items: Vec<VarId>, sizes: Vec<i64>, limits: Vec<LocalRhs>, exact: bool) {
        self.constraints.push(LocalConstraint::BinPacking { items, sizes, limits, exact });
    }

    pub(crate) fn add_no_overlap(&mut self, origins: Vec<Vec<VarId>>, lengths: Vec<Vec<Expr>>, zero_ignored: bool) {
        self.constraints.push(LocalConstraint::NoOverlap { origins, lengths, zero_ignored });
    }

    pub(crate) fn add_regular(&mut self, vars: Vec<VarId>, dfa: Dfa) {
        self.constraints.push(LocalConstraint::Regular { vars, dfa });
    }

    pub(crate) fn add_mdd(&mut self, vars: Vec<VarId>, mdd: Mdd) {
        self.constraints.push(LocalConstraint::Mdd { vars, mdd });
    }

    pub(crate) fn unsupported(&self) -> usize {
        self.unsupported
    }

    fn ensure(&mut self, var: VarId) {
        if self.derived.len() <= var.index() {
            self.derived.resize(var.index() + 1, false);
        }
    }

    fn mark_functional(&mut self, functional: Functional) {
        let target = match &functional {
            Functional::Expr { target, .. }
            | Functional::Linear { target, .. }
            | Functional::Element { target, .. }
            | Functional::BoolTable { target, .. } => *target,
        };
        self.ensure(target);
        self.derived[target.index()] = true;
        self.functionals.push(functional);
    }
}

impl LocalModel {
    fn new(problem: Problem, spec: LocalSearchSpec) -> Result<Self, usize> {
        let objective = problem.objective.clone().ok_or(1usize)?;
        let mut domains = Vec::with_capacity(problem.solver.store.num_vars());
        for i in 0..problem.solver.store.num_vars() {
            let var = VarId(i as u32);
            let min = problem.solver.store.min(var);
            let max = problem.solver.store.max(var);
            let values = if problem.solver.store.size(var) <= MAX_DOMAIN_VALUES {
                problem.solver.store.values(var).collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            if values.is_empty() && min > max {
                return Err(1);
            }
            domains.push(LocalDomain { min, max, values });
        }
        let mutable = problem
            .search
            .iter()
            .copied()
            .filter(|&var| domains[var.index()].values.len() > 1 && !spec.derived.get(var.index()).copied().unwrap_or(false))
            .collect();
        let constraints = spec.constraints;
        let functionals = order_functionals(spec.functionals);
        let bool_tables = bool_tables(&functionals);
        let exact_covers = exact_cover_rows(&constraints, &domains);
        let affected = build_affected(domains.len(), &constraints, &functionals);
        Ok(Self { domains, mutable, search: problem.search, objective, constraints, functionals, bool_tables, exact_covers, affected })
    }

    fn random_assignment(&self, seed: u64) -> Vec<i32> {
        self.domains.iter().enumerate().map(|(i, domain)| domain.initial_value(seed ^ i as u64)).collect()
    }

    fn min_assignment(&self) -> Vec<i32> {
        self.domains.iter().map(LocalDomain::min_value).collect()
    }

    fn constructive_assignment(&self, seed: u64) -> Vec<i32> {
        let mut assignment = self.min_assignment();
        self.greedy_exact_cover(&mut assignment);
        let _ = self.complete(&mut assignment);
        self.place_guarded_elements(assignment, seed)
    }

    fn has_extension(&self) -> bool {
        self.constraints.iter().any(|constraint| matches!(constraint, LocalConstraint::Extension { .. }))
    }

    fn objective_kicks(&self) -> Vec<(VarId, i32)> {
        let mut kicks = Vec::new();
        let minimize = self.objective.minimizing();
        for (coeff, var) in self.objective_terms() {
            if coeff != 0 {
                self.push_objective_kick(&mut kicks, var, (minimize && coeff < 0) || (!minimize && coeff > 0));
            }
        }
        kicks
    }

    fn push_objective_kick(&self, kicks: &mut Vec<(VarId, i32)>, var: VarId, prefer_max: bool) {
        if self.mutable.contains(&var) {
            let domain = &self.domains[var.index()];
            kicks.push((var, if prefer_max { domain.max_value() } else { domain.min_value() }));
        }
    }

    fn objective_weight(&self, var: VarId) -> Option<i64> {
        let minimize = self.objective.minimizing();
        self.objective_terms().into_iter().find_map(|(coeff, candidate)| {
            (candidate == var && ((!minimize && coeff > 0) || (minimize && coeff < 0))).then_some(coeff.abs())
        })
    }

    fn objective_terms(&self) -> Vec<(i64, VarId)> {
        match &self.objective {
            Objective::Var(_, var) => self.expanded_var_terms(*var).unwrap_or_else(|| vec![(1, *var)]),
            Objective::Linear(_, coeffs, vars) => coeffs.iter().copied().zip(vars.iter().copied()).collect(),
            Objective::Expr(_, _) => Vec::new(),
        }
    }

    fn expanded_var_terms(&self, objective: VarId) -> Option<Vec<(i64, VarId)>> {
        for functional in &self.functionals {
            let Functional::Linear { target, coeff, terms, .. } = functional else {
                continue;
            };
            if *target != objective || *coeff == 0 {
                continue;
            }
            let mut expanded = Vec::with_capacity(terms.len());
            for &(c, var) in terms {
                if c % coeff != 0 {
                    return None;
                }
                expanded.push((-c / coeff, var));
            }
            return Some(expanded);
        }
        None
    }

    fn objective_costs(&self) -> HashMap<VarId, i64> {
        let minimizing = self.objective.minimizing();
        self.objective_terms().into_iter().map(|(coeff, var)| (var, if minimizing { coeff } else { -coeff })).collect()
    }

    fn greedy_exact_cover(&self, assignment: &mut [i32]) {
        let covers = self.exact_cover_rows();
        if covers.is_empty() {
            return;
        }
        let mut memberships: HashMap<VarId, Vec<usize>> = HashMap::new();
        for (row, vars) in covers.iter().enumerate() {
            for &var in vars {
                memberships.entry(var).or_default().push(row);
            }
        }

        let costs = self.objective_costs();
        let mut counts = vec![0usize; covers.len()];
        let mut selected = vec![false; self.domains.len()];
        while let Some(row) = counts.iter().position(|&count| count == 0) {
            let Some(var) = self.best_cover_var(&covers[row], &memberships, &counts, &selected, &costs) else {
                return;
            };
            selected[var.index()] = true;
            assignment[var.index()] = 1;
            if let Some(rows) = memberships.get(&var) {
                for &r in rows {
                    counts[r] += 1;
                }
            }
        }
    }

    fn exact_cover_rows(&self) -> Vec<Vec<VarId>> {
        self.exact_covers.clone()
    }

    fn best_cover_var(
        &self,
        row: &[VarId],
        memberships: &HashMap<VarId, Vec<usize>>,
        counts: &[usize],
        selected: &[bool],
        costs: &HashMap<VarId, i64>,
    ) -> Option<VarId> {
        let mut best = None;
        for &var in row {
            if selected[var.index()] {
                continue;
            }
            let rows = memberships.get(&var).map(Vec::as_slice).unwrap_or(&[]);
            let conflicts = rows.iter().filter(|&&r| counts[r] > 0).count();
            let uncovered = rows.iter().filter(|&&r| counts[r] == 0).count();
            let key = (conflicts, usize::MAX - uncovered, *costs.get(&var).unwrap_or(&1), var.index());
            if best.is_none_or(|(old, _)| key < old) {
                best = Some((key, var));
            }
        }
        best.map(|(_, var)| var)
    }

    fn place_guarded_elements(&self, mut assignment: Vec<i32>, seed: u64) -> Vec<i32> {
        let elements = self.element_views();
        let mut words = self.guarded_words(&elements);
        if seed == 0 {
            words.sort_by(|a, b| b.weight.cmp(&a.weight).then_with(|| a.guard.cmp(&b.guard)));
        } else {
            words.sort_by(|a, b| {
                let ak = mix64(seed ^ a.guard.index() as u64);
                let bk = mix64(seed ^ b.guard.index() as u64);
                let aw = a.weight * 64 + (ak & 511) as i64;
                let bw = b.weight * 64 + (bk & 511) as i64;
                bw.cmp(&aw).then_with(|| a.guard.cmp(&b.guard))
            });
        }
        let mut fixed = vec![None; self.domains.len()];

        for word in words {
            let Some((trial, placements)) = self.try_place_word(&assignment, &fixed, &word) else {
                continue;
            };
            let mut scored = trial;
            if self.score(&mut scored).violation == 0 {
                for (var, value) in placements {
                    fixed[var.index()] = Some(value);
                }
                assignment = scored;
            }
        }
        assignment
    }

    fn element_views(&self) -> ElementViews<'_> {
        let mut elements = HashMap::new();
        for functional in &self.functionals {
            if let Functional::Element { target, array, index, start_index } = functional {
                elements.insert(*target, (array.as_slice(), *index, *start_index));
            }
        }
        elements
    }

    fn guarded_words<'a>(&self, elements: &ElementViews<'a>) -> Vec<GuardedWord<'a>> {
        let mut requirements: BTreeMap<VarId, Vec<(VarId, i32)>> = BTreeMap::new();
        for constraint in &self.constraints {
            match constraint {
                LocalConstraint::Extension { vars, tuples } => {
                    if let Some((guard, target, value)) = guarded_value(vars, tuples) {
                        requirements.entry(guard).or_default().push((target, value));
                    }
                }
                LocalConstraint::Expr(expr) => {
                    if let Some((guard, target, value)) = guarded_expr_value(expr) {
                        requirements.entry(guard).or_default().push((target, value));
                    }
                }
                _ => {}
            }
        }

        let mut words = Vec::new();
        for (guard, reqs) in requirements {
            let Some(weight) = self.objective_weight(guard) else {
                continue;
            };
            let mut letters = Vec::with_capacity(reqs.len());
            let mut array = None;
            for (target, value) in reqs {
                let Some(&(candidate_array, index, start_index)) = elements.get(&target) else {
                    letters.clear();
                    break;
                };
                if start_index != 0 {
                    letters.clear();
                    break;
                }
                if let Some(array) = array {
                    if array != candidate_array {
                        letters.clear();
                        break;
                    }
                } else {
                    array = Some(candidate_array);
                }
                letters.push((index, value));
            }
            if let Some(array) = array {
                if !letters.is_empty() && self.domains[guard.index()].contains(1) {
                    letters.sort_by_key(|&(index, _)| index.index());
                    words.push(GuardedWord { guard, weight, array, letters });
                }
            }
        }
        words
    }

    fn try_place_word(&self, assignment: &[i32], fixed: &[Option<i32>], word: &GuardedWord<'_>) -> Option<TrialPlacement> {
        let mut state = WordPlacementState {
            values: vec![None; word.array.len()],
            cells: vec![0; word.letters.len()],
            nodes: WORD_PLACEMENT_NODE_LIMIT,
        };
        for (cell, &var) in word.array.iter().enumerate() {
            state.values[cell] = fixed[var.index()];
        }
        self.place_word_letters(assignment, word, 0, None, &mut state)
    }

    fn word_trial(&self, assignment: &[i32], word: &GuardedWord<'_>, values: &[Option<i32>], cells: &[usize]) -> Option<TrialPlacement> {
        let mut trial = assignment.to_vec();
        let mut placements = Vec::new();
        trial[word.guard.index()] = 1;
        for (cell, &value) in values.iter().enumerate() {
            if let Some(value) = value {
                let x = word.array[cell];
                trial[x.index()] = value;
                placements.push((x, value));
            }
        }
        for (&cell, &(index, _)) in cells.iter().zip(&word.letters) {
            trial[index.index()] = cell as i32;
        }
        self.lex_constraints_ok(&trial).then_some((trial, placements))
    }

    fn lex_constraints_ok(&self, assignment: &[i32]) -> bool {
        self.constraints.iter().all(|constraint| match constraint {
            LocalConstraint::Lex { rows, strict } => lex_chain_violation(rows, *strict, assignment) == 0,
            _ => true,
        })
    }

    fn pair_allowed(&self, left: VarId, right: VarId, left_value: i32, right_value: i32) -> bool {
        if let Some(true_pairs) = self.bool_tables.get(&(left, right)) {
            return true_pairs.binary_search(&(left_value, right_value)).is_ok();
        }
        self.bool_tables.get(&(right, left)).is_none_or(|true_pairs| true_pairs.binary_search(&(right_value, left_value)).is_ok())
    }

    fn place_word_letters(
        &self,
        assignment: &[i32],
        word: &GuardedWord<'_>,
        pos: usize,
        previous: Option<usize>,
        state: &mut WordPlacementState,
    ) -> Option<TrialPlacement> {
        if state.nodes == 0 {
            return None;
        }
        state.nodes -= 1;
        if pos == word.letters.len() {
            return self.word_trial(assignment, word, &state.values, &state.cells);
        }
        let (index, value) = word.letters[pos];
        for cell in 0..word.array.len() {
            if previous == Some(cell) || !self.domains[index.index()].contains(cell as i32) {
                continue;
            }
            if let Some(previous) = previous {
                let previous_index = word.letters[pos - 1].0;
                if !self.pair_allowed(previous_index, index, previous as i32, cell as i32) {
                    continue;
                }
            }
            match state.values[cell] {
                Some(old) if old != value => continue,
                Some(_) => {
                    state.cells[pos] = cell;
                    if let Some(trial) = self.place_word_letters(assignment, word, pos + 1, Some(cell), state) {
                        return Some(trial);
                    }
                }
                None => {
                    let x = word.array[cell];
                    if !self.domains[x.index()].contains(value) {
                        continue;
                    }
                    state.values[cell] = Some(value);
                    state.cells[pos] = cell;
                    if let Some(trial) = self.place_word_letters(assignment, word, pos + 1, Some(cell), state) {
                        return Some(trial);
                    }
                    state.values[cell] = None;
                }
            }
        }
        None
    }

    fn focused_repair_vars(&self, assignment: &[i32], seed: u64, iter: u64) -> Vec<VarId> {
        let mut vars = Vec::new();
        for row in &self.exact_covers {
            let selected = row.iter().filter(|&&var| assignment[var.index()] == 1).count();
            match selected {
                0 => vars.extend(row.iter().copied()),
                2.. => vars.extend(row.iter().copied().filter(|&var| assignment[var.index()] == 1)),
                _ => {}
            }
        }
        vars.sort_unstable();
        vars.dedup();
        candidate_vars(&vars, seed, iter)
    }

    fn complete(&self, assignment: &mut [i32]) -> bool {
        for functional in &self.functionals {
            let Some(value) = self.functional_value(functional, assignment) else {
                return false;
            };
            let target = match functional {
                Functional::Expr { target, .. }
                | Functional::Linear { target, .. }
                | Functional::Element { target, .. }
                | Functional::BoolTable { target, .. } => *target,
            };
            if !self.domains[target.index()].contains(value) {
                return false;
            }
            assignment[target.index()] = value;
        }
        true
    }

    fn functional_value(&self, functional: &Functional, assignment: &[i32]) -> Option<i32> {
        match functional {
            Functional::Expr { expr, .. } => to_i32(expr.eval(&|v| assignment[v.index()] as i64)?),
            Functional::Linear { coeff, terms, rhs, .. } => {
                let used = terms.iter().map(|&(c, v)| c * assignment[v.index()] as i64).sum::<i64>();
                let num = rhs - used;
                if num % coeff != 0 {
                    return None;
                }
                to_i32(num / coeff)
            }
            Functional::Element { array, index, start_index, .. } => {
                let offset = assignment[index.index()] as i64 - *start_index as i64;
                if !(0..array.len() as i64).contains(&offset) {
                    return None;
                }
                Some(assignment[array[offset as usize].index()])
            }
            Functional::BoolTable { left, right, true_pairs, .. } => {
                Some(i32::from(true_pairs.binary_search(&(assignment[left.index()], assignment[right.index()])).is_ok()))
            }
        }
    }

    fn objective_value(&self, assignment: &[i32]) -> Option<i64> {
        match &self.objective {
            Objective::Var(_, var) => Some(assignment[var.index()] as i64),
            Objective::Linear(_, coeffs, vars) => Some(coeffs.iter().zip(vars).map(|(&c, &v)| c * assignment[v.index()] as i64).sum()),
            Objective::Expr(_, expr) => expr.eval(&|v| assignment[v.index()] as i64),
        }
    }

    /// Unweighted score, used for feasibility checks (violation == 0). Independent
    /// of any GLS weights, since a weighted sum is zero iff the raw sum is.
    fn score(&self, assignment: &mut [i32]) -> Score {
        let mut violation = i64::from(!self.complete(assignment)) * 1_000_000;
        for constraint in &self.constraints {
            violation = violation.saturating_add(self.violation(constraint, assignment));
        }
        let objective = self.objective_value(assignment).unwrap_or(i64::MAX / 4);
        Score { violation, objective: if self.objective.minimizing() { objective } else { -objective } }
    }

    /// Score plus the per-constraint violation vector and completeness flag, used
    /// to seed and maintain the incremental delta-scoring caches. The reported
    /// `violation` is the GLS-**weighted** constraint sum (`Σ wᵢ·violᵢ`) plus the
    /// incompleteness penalty; pass all-ones weights for the unweighted sum.
    fn score_breakdown(&self, assignment: &mut [i32], weights: &[i64]) -> (Score, Vec<i64>, bool) {
        let complete = self.complete(assignment);
        let mut con_viol = Vec::with_capacity(self.constraints.len());
        let mut weighted: i128 = 0;
        for (i, constraint) in self.constraints.iter().enumerate() {
            let v = self.violation(constraint, assignment);
            weighted += i128::from(v) * i128::from(weights[i]);
            con_viol.push(v);
        }
        let penalty = i128::from(!complete) * 1_000_000;
        let objective = self.objective_value(assignment).unwrap_or(i64::MAX / 4);
        let objective = if self.objective.minimizing() { objective } else { -objective };
        (Score { violation: combine_violation(penalty, weighted), objective }, con_viol, complete)
    }

    fn violation(&self, constraint: &LocalConstraint, assignment: &[i32]) -> i64 {
        match constraint {
            LocalConstraint::Expr(expr) => match expr.eval(&|v| assignment[v.index()] as i64) {
                Some(v) if v != 0 => 0,
                Some(_) => 1,
                None => 1_000,
            },
            LocalConstraint::Linear { coeffs, vars, rel, rhs } => {
                let lhs = coeffs.iter().zip(vars).map(|(&c, &v)| c * assignment[v.index()] as i64).sum();
                relation_violation(lhs, *rel, *rhs)
            }
            LocalConstraint::AllDifferent(vars) => {
                let mut values = vars.iter().map(|&v| assignment[v.index()]).collect::<Vec<_>>();
                values.sort_unstable();
                values.windows(2).filter(|w| w[0] == w[1]).count() as i64
            }
            LocalConstraint::AllDifferentRows(rows) => all_different_rows_violation(rows, assignment),
            LocalConstraint::AllDifferentExcept { vars, except } => all_different_except_violation(vars, except, assignment),
            LocalConstraint::AllEqual(vars) => vars.split_first().map_or(0, |(&first, rest)| {
                rest.iter().filter(|&&var| assignment[var.index()] != assignment[first.index()]).count() as i64
            }),
            LocalConstraint::Extension { vars, tuples } => {
                if tuples.iter().any(|tuple| tuple_matches(vars, tuple, assignment)) {
                    0
                } else {
                    1
                }
            }
            LocalConstraint::NegExtension { vars, tuples } => {
                // Conflict table: every forbidden tuple the assignment matches is a
                // violation (a gradient, since wildcard tuples may match several).
                tuples.iter().filter(|tuple| tuple_matches(vars, tuple, assignment)).count() as i64
            }
            LocalConstraint::Lex { rows, strict } => lex_chain_violation(rows, *strict, assignment),
            LocalConstraint::Count { vars, values, rel, rhs } => {
                let count = vars.iter().filter(|&&var| values.contains(&assignment[var.index()])).count() as i64;
                relation_violation(count, *rel, rhs.value(assignment))
            }
            LocalConstraint::CountAllowed { vars, values, allowed } => {
                let count = vars.iter().filter(|&&var| values.contains(&assignment[var.index()])).count() as i32;
                i64::from(!allowed.contains(&count))
            }
            LocalConstraint::NValues { vars, rel, rhs } => {
                let mut values = vars.iter().map(|&var| assignment[var.index()]).collect::<Vec<_>>();
                values.sort_unstable();
                values.dedup();
                relation_violation(values.len() as i64, *rel, rhs.value(assignment))
            }
            LocalConstraint::Cardinality { vars, values, low, high, closed } => {
                cardinality_violation(vars, values, low, high, *closed, assignment)
            }
            LocalConstraint::Extremum { vars, is_min, rel, rhs } => {
                let Some(value) = extremum(vars, *is_min, assignment) else { return 1 };
                relation_violation(value as i64, *rel, rhs.value(assignment))
            }
            LocalConstraint::ElementMember { array, value } => element_member_violation(array, *value, assignment),
            LocalConstraint::Cumulative { starts, durations, heights, cap } => {
                cumulative_violation(starts, durations, heights, cap.value(assignment), assignment)
            }
            LocalConstraint::ChannelInverse { xs, x_start, ys, y_start } => {
                channel_inverse_violation(xs, *x_start, ys, *y_start, assignment)
            }
            LocalConstraint::ChannelOneHot { xs, value, start_index } => channel_onehot_violation(xs, *value, *start_index, assignment),
            LocalConstraint::Precedence { vars, values, covered } => precedence_violation(vars, values, *covered, assignment),
            LocalConstraint::Circuit(vars) => circuit_violation(vars, assignment),
            LocalConstraint::BinPacking { items, sizes, limits, exact } => bin_packing_violation(items, sizes, limits, *exact, assignment),
            LocalConstraint::NoOverlap { origins, lengths, zero_ignored } => {
                no_overlap_violation(origins, lengths, *zero_ignored, assignment)
            }
            LocalConstraint::Regular { vars, dfa } => regular_violation(vars, dfa, assignment),
            LocalConstraint::Mdd { vars, mdd } => mdd_violation(vars, mdd, assignment),
        }
    }

    fn search_solution(&self, assignment: &[i32]) -> Vec<i32> {
        self.search.iter().map(|&v| assignment[v.index()]).collect()
    }
}

fn to_i32(value: i64) -> Option<i32> {
    (i32::MIN as i64..=i32::MAX as i64).contains(&value).then_some(value as i32)
}

fn tuple_matches(vars: &[VarId], tuple: &[i32], assignment: &[i32]) -> bool {
    vars.iter().zip(tuple).all(|(&var, &value)| value == STAR || assignment[var.index()] == value)
}

fn lex_chain_violation(rows: &[Vec<VarId>], strict: bool, assignment: &[i32]) -> i64 {
    rows.windows(2).filter(|pair| !lex_le(&pair[0], &pair[1], strict, assignment)).count() as i64
}

fn lex_le(a: &[VarId], b: &[VarId], strict: bool, assignment: &[i32]) -> bool {
    for (&x, &y) in a.iter().zip(b) {
        let xv = assignment[x.index()];
        let yv = assignment[y.index()];
        if xv != yv {
            return xv < yv;
        }
    }
    !strict
}

fn extremum(vars: &[VarId], is_min: bool, assignment: &[i32]) -> Option<i32> {
    if is_min {
        vars.iter().map(|&var| assignment[var.index()]).min()
    } else {
        vars.iter().map(|&var| assignment[var.index()]).max()
    }
}

fn cardinality_violation(vars: &[VarId], values: &[i32], low: &[i64], high: &[i64], closed: bool, assignment: &[i32]) -> i64 {
    let mut counts = vec![0i64; values.len()];
    let mut violation = 0;
    for &var in vars {
        let value = assignment[var.index()];
        if let Some(pos) = values.iter().position(|&v| v == value) {
            counts[pos] += 1;
        } else if closed {
            violation += 1;
        }
    }
    for ((count, &lo), &hi) in counts.iter().zip(low).zip(high) {
        violation += (lo - *count).max(0) + (*count - hi).max(0);
    }
    violation
}

fn all_different_rows_violation(rows: &[Vec<VarId>], assignment: &[i32]) -> i64 {
    let mut seen: HashMap<Vec<i32>, i64> = HashMap::new();
    let mut violation = 0;
    for row in rows {
        let tuple = row.iter().map(|&v| assignment[v.index()]).collect::<Vec<_>>();
        let count = seen.entry(tuple).or_insert(0);
        violation += *count;
        *count += 1;
    }
    violation
}

fn all_different_except_violation(vars: &[VarId], except: &[i32], assignment: &[i32]) -> i64 {
    let mut counts: HashMap<i32, i64> = HashMap::new();
    let mut violation = 0;
    for &var in vars {
        let value = assignment[var.index()];
        if except.contains(&value) {
            continue;
        }
        let count = counts.entry(value).or_insert(0);
        violation += *count;
        *count += 1;
    }
    violation
}

fn element_member_violation(array: &[VarId], value: i32, assignment: &[i32]) -> i64 {
    if array.iter().any(|&var| assignment[var.index()] == value) {
        0
    } else {
        array.iter().map(|&var| (i64::from(assignment[var.index()]) - i64::from(value)).abs()).min().unwrap_or(1).max(1)
    }
}

fn cumulative_violation(starts: &[VarId], durations: &[VarId], heights: &[VarId], cap: i64, assignment: &[i32]) -> i64 {
    let mut tasks = Vec::new();
    let mut points = Vec::new();
    for ((&start, &duration), &height) in starts.iter().zip(durations).zip(heights) {
        let s = assignment[start.index()] as i64;
        let d = assignment[duration.index()] as i64;
        let h = assignment[height.index()] as i64;
        if d <= 0 || h <= 0 {
            continue;
        }
        let e = s + d;
        tasks.push((s, e, h));
        points.push(s);
        points.push(e);
    }
    points.sort_unstable();
    points.dedup();
    let mut violation = (-cap).max(0);
    for window in points.windows(2) {
        let lo = window[0];
        let hi = window[1];
        if lo >= hi {
            continue;
        }
        let t = lo;
        let usage: i64 = tasks.iter().filter(|&&(s, e, _)| s <= t && t < e).map(|&(_, _, h)| h).sum();
        violation = violation.saturating_add((usage - cap).max(0).saturating_mul((hi - lo).max(1)));
    }
    violation
}

fn channel_inverse_violation(xs: &[VarId], x_start: i32, ys: &[VarId], y_start: i32, assignment: &[i32]) -> i64 {
    let mut violation = 0;
    for (i, &x) in xs.iter().enumerate() {
        let y_pos = assignment[x.index()] - y_start;
        if (0..ys.len() as i32).contains(&y_pos) {
            let expected = x_start + i as i32;
            violation += i64::from(assignment[ys[y_pos as usize].index()] != expected);
        } else {
            violation += 1;
        }
    }
    for (j, &y) in ys.iter().enumerate() {
        let x_pos = assignment[y.index()] - x_start;
        if (0..xs.len() as i32).contains(&x_pos) {
            let expected = y_start + j as i32;
            violation += i64::from(assignment[xs[x_pos as usize].index()] != expected);
        } else {
            violation += 1;
        }
    }
    violation
}

fn channel_onehot_violation(xs: &[VarId], value: VarId, start_index: i32, assignment: &[i32]) -> i64 {
    let target = assignment[value.index()];
    let mut violation = 0;
    let mut ones = 0;
    for (i, &x) in xs.iter().enumerate() {
        let xv = assignment[x.index()];
        let selected = target == start_index + i as i32;
        violation += match (xv, selected) {
            (1, true) | (0, false) => 0,
            _ => 1,
        };
        ones += i64::from(xv == 1);
    }
    violation + (ones - 1).abs()
}

fn precedence_violation(vars: &[VarId], values: &[i32], covered: bool, assignment: &[i32]) -> i64 {
    let first_pos = |value| vars.iter().position(|&var| assignment[var.index()] == value);
    let mut violation = 0;
    for pair in values.windows(2) {
        let prev = first_pos(pair[0]);
        let next = first_pos(pair[1]);
        if let Some(next_pos) = next {
            if prev.is_none_or(|prev_pos| prev_pos > next_pos) {
                violation += 1;
            }
        }
    }
    if covered {
        violation += values.iter().filter(|&&value| first_pos(value).is_none()).count() as i64;
    }
    violation
}

fn circuit_violation(vars: &[VarId], assignment: &[i32]) -> i64 {
    let n = vars.len();
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return i64::from(assignment[vars[0].index()] != 0);
    }
    let mut indeg = vec![0i64; n];
    let mut violation = 0;
    for (i, &var) in vars.iter().enumerate() {
        let succ = assignment[var.index()];
        if succ < 0 || succ as usize >= n {
            violation += 1;
            continue;
        }
        if succ as usize == i {
            violation += 1;
        }
        indeg[succ as usize] += 1;
    }
    violation += indeg.iter().map(|&count| (count - 1).abs()).sum::<i64>();
    if violation > 0 {
        return violation;
    }
    let mut seen = vec![false; n];
    let mut cycles = 0;
    for start in 0..n {
        if seen[start] {
            continue;
        }
        cycles += 1;
        let mut cur = start;
        while !seen[cur] {
            seen[cur] = true;
            cur = assignment[vars[cur].index()] as usize;
        }
    }
    violation + (cycles - 1).max(0) as i64
}

fn bin_packing_violation(items: &[VarId], sizes: &[i64], limits: &[LocalRhs], exact: bool, assignment: &[i32]) -> i64 {
    let mut loads = vec![0i64; limits.len()];
    let mut violation = 0;
    for (&item, &size) in items.iter().zip(sizes) {
        let bin = assignment[item.index()];
        if (0..loads.len() as i32).contains(&bin) {
            loads[bin as usize] += size;
        } else {
            violation += size.abs().max(1);
        }
    }
    for (load, limit) in loads.into_iter().zip(limits) {
        let cap = limit.value(assignment);
        violation += if exact { (load - cap).abs() } else { (load - cap).max(0) };
    }
    violation
}

fn no_overlap_violation(origins: &[Vec<VarId>], lengths: &[Vec<Expr>], zero_ignored: bool, assignment: &[i32]) -> i64 {
    let mut boxes = Vec::new();
    for (origin, length) in origins.iter().zip(lengths) {
        let mut dims = Vec::with_capacity(origin.len());
        let mut active = true;
        for (&start_var, len_expr) in origin.iter().zip(length) {
            let Some(len) = len_expr.eval(&|v| assignment[v.index()] as i64) else {
                active = false;
                break;
            };
            if len <= 0 {
                if zero_ignored {
                    active = false;
                    break;
                }
                return 1;
            }
            let start = assignment[start_var.index()] as i64;
            dims.push((start, start + len));
        }
        if active {
            boxes.push(dims);
        }
    }
    let mut violation = 0;
    for i in 0..boxes.len() {
        for j in (i + 1)..boxes.len() {
            let mut overlap = i64::MAX;
            for (&(a0, a1), &(b0, b1)) in boxes[i].iter().zip(&boxes[j]) {
                let amount = a1.min(b1) - a0.max(b0);
                if amount <= 0 {
                    overlap = 0;
                    break;
                }
                overlap = overlap.min(amount);
            }
            violation += overlap.max(0);
        }
    }
    violation
}

fn regular_violation(vars: &[VarId], dfa: &Dfa, assignment: &[i32]) -> i64 {
    let mut delta = vec![HashMap::new(); dfa.n_states];
    for &(src, value, dst) in &dfa.transitions {
        if src < dfa.n_states {
            delta[src].insert(value, dst);
        }
    }
    let mut state = dfa.start;
    for (i, &var) in vars.iter().enumerate() {
        if state >= dfa.n_states {
            return (vars.len() - i + 1) as i64;
        }
        let value = assignment[var.index()];
        let Some(&next) = delta[state].get(&value) else {
            return (vars.len() - i) as i64 + 1;
        };
        state = next;
    }
    i64::from(!dfa.accept.contains(&state))
}

fn mdd_violation(vars: &[VarId], mdd: &Mdd, assignment: &[i32]) -> i64 {
    let mut frontier = HashSet::from([0usize]);
    for (layer, &var) in vars.iter().enumerate() {
        let value = assignment[var.index()];
        let Some(arcs) = mdd.layers.get(layer) else {
            return (vars.len() - layer + 1) as i64;
        };
        let mut next = HashSet::new();
        for arc in arcs {
            if frontier.contains(&arc.from) && arc.value == value {
                next.insert(arc.to);
            }
        }
        if next.is_empty() {
            return (vars.len() - layer) as i64 + 1;
        }
        frontier = next;
    }
    let final_nodes = mdd.nodes_per_layer.last().copied().unwrap_or(0);
    i64::from(!frontier.iter().any(|&node| node < final_nodes))
}

fn exact_cover_rows(constraints: &[LocalConstraint], domains: &[LocalDomain]) -> Vec<Vec<VarId>> {
    constraints
        .iter()
        .filter_map(|constraint| {
            let LocalConstraint::Linear { coeffs, vars, rel: Relation::Eq, rhs: 1 } = constraint else {
                return None;
            };
            (coeffs.iter().all(|&c| c == 1) && vars.iter().all(|&var| domains[var.index()].is_bool())).then_some(vars.clone())
        })
        .collect()
}

fn bool_tables(functionals: &[Functional]) -> HashMap<(VarId, VarId), Vec<(i32, i32)>> {
    functionals
        .iter()
        .filter_map(|functional| {
            let Functional::BoolTable { left, right, true_pairs, .. } = functional else {
                return None;
            };
            Some(((*left, *right), true_pairs.clone()))
        })
        .collect()
}

fn guarded_value(vars: &[VarId], tuples: &[Vec<i32>]) -> Option<(VarId, VarId, i32)> {
    if vars.len() != 2 {
        return None;
    }
    let mut inactive_free = false;
    let mut active_value = None;
    for tuple in tuples {
        if tuple.len() != 2 {
            return None;
        }
        match (tuple[0], tuple[1]) {
            (0, STAR) => inactive_free = true,
            (1, value) if value != STAR => active_value = Some(value),
            _ => return None,
        }
    }
    inactive_free.then_some((vars[0], vars[1], active_value?))
}

fn guarded_expr_value(expr: &Expr) -> Option<(VarId, VarId, i32)> {
    let Expr::Or(terms) = expr else {
        return None;
    };
    let [a, b] = terms.as_slice() else {
        return None;
    };
    guarded_expr_parts(a, b).or_else(|| guarded_expr_parts(b, a))
}

fn guarded_expr_parts(guard: &Expr, target: &Expr) -> Option<(VarId, VarId, i32)> {
    let guard = eq_var_const(guard, 0)?;
    let (target, value) = eq_var_any_const(target)?;
    Some((guard, target, value))
}

fn eq_var_const(expr: &Expr, expected: i64) -> Option<VarId> {
    let Expr::Eq(a, b) = expr else {
        return None;
    };
    match (&**a, &**b) {
        (Expr::Var(var), Expr::Const(value)) | (Expr::Const(value), Expr::Var(var)) if *value == expected => Some(*var),
        _ => None,
    }
}

fn eq_var_any_const(expr: &Expr) -> Option<(VarId, i32)> {
    let Expr::Eq(a, b) = expr else {
        return None;
    };
    match (&**a, &**b) {
        (Expr::Var(var), Expr::Const(value)) | (Expr::Const(value), Expr::Var(var)) => to_i32(*value).map(|value| (*var, value)),
        _ => None,
    }
}

fn better_value(minimizing: bool, new_value: i64, old: Option<i64>) -> bool {
    old.is_none_or(|old| if minimizing { new_value < old } else { new_value > old })
}

fn candidate_vars(vars: &[VarId], seed: u64, iter: u64) -> Vec<VarId> {
    if vars.len() <= MAX_SAMPLED_VARS {
        return vars.to_vec();
    }
    (0..MAX_SAMPLED_VARS).map(|i| vars[mix64(seed ^ iter ^ i as u64) as usize % vars.len()]).collect()
}

fn objective_kick_trial(model: &LocalModel, assignment: &[i32], kicks: &[(VarId, i32)], seed: u64, iter: u64) -> Option<Vec<i32>> {
    if kicks.is_empty() {
        return None;
    }
    let start = mix64(seed ^ iter) as usize % kicks.len();
    for offset in 0..kicks.len() {
        let (var, value) = kicks[(start + offset) % kicks.len()];
        if assignment[var.index()] != value {
            let mut trial = assignment.to_vec();
            trial[var.index()] = value;
            if model.complete(&mut trial) {
                return Some(trial);
            }
        }
    }
    None
}

fn objective_kick(model: &LocalModel, assignment: &[i32], kicks: &[(VarId, i32)], seed: u64, iter: u64) -> Option<Vec<i32>> {
    if iter.is_multiple_of(RANDOM_WALK_PERIOD) {
        objective_kick_trial(model, assignment, kicks, seed, iter)
    } else {
        None
    }
}

fn signed_log_delta(delta: i128) -> f64 {
    let magnitude = (delta.unsigned_abs().min(u128::from(u64::MAX)) as f64).ln_1p();
    if delta >= 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn kick_reward(before: Score, after: Score) -> f64 {
    let violation = i128::from(before.violation) - i128::from(after.violation);
    let objective = i128::from(before.objective) - i128::from(after.objective);
    100.0 * signed_log_delta(violation) + signed_log_delta(objective)
}

#[allow(clippy::too_many_arguments)]
fn best_single_variable_move(
    model: &LocalModel,
    assignment: &[i32],
    work: &mut [i32],
    weights: &[i64],
    con_viol: &[i64],
    viol_sum: i128,
    minimizing: bool,
    config: LsConfig,
    value_sets: &[Option<HashSet<i32>>],
    cand_values: &mut Vec<i32>,
    seed: u64,
    iter: u64,
) -> Option<(Score, usize, i32)> {
    let focused = model.focused_repair_vars(assignment, seed, iter);
    let candidates = if focused.is_empty() { candidate_vars(&model.mutable, seed, iter) } else { focused };
    let mut best_move: Option<(Score, usize, i32)> = None;
    for var in candidates {
        let j = var.index();
        let old_val = assignment[j];
        let affected = &model.affected[j];
        let candidate_values: &[i32] = if config.min_conflicts && value_sets.get(j).and_then(Option::as_ref).is_some() {
            min_conflict_candidates(model, var, assignment, value_sets[j].as_ref().unwrap(), seed, iter, cand_values);
            cand_values
        } else {
            &model.domains[j].values
        };
        for &value in candidate_values {
            if value == old_val {
                continue;
            }
            work.copy_from_slice(assignment);
            work[j] = value;
            let trial_complete = model.complete(work);
            let mut delta: i128 = 0;
            for &c in affected {
                delta += (i128::from(model.violation(&model.constraints[c], work)) - i128::from(con_viol[c])) * i128::from(weights[c]);
            }
            let trial_penalty = i128::from(!trial_complete) * 1_000_000;
            let violation = combine_violation(trial_penalty, viol_sum + delta);
            let objective = model.objective_value(work).unwrap_or(i64::MAX / 4);
            let objective = if minimizing { objective } else { -objective };
            let score = Score { violation, objective };
            if best_move.as_ref().is_none_or(|&(best, _, _)| score < best) {
                best_move = Some((score, j, value));
            }
        }
    }
    best_move
}

#[allow(clippy::too_many_arguments)]
fn apply_single_variable_move(
    model: &LocalModel,
    assignment: &mut [i32],
    con_viol: &mut [i64],
    viol_sum: &mut i128,
    complete_now: &mut bool,
    current: &mut Score,
    weights: &[i64],
    score: Score,
    j: usize,
    value: i32,
) {
    assignment[j] = value;
    *complete_now = model.complete(assignment);
    for &c in &model.affected[j] {
        let updated = model.violation(&model.constraints[c], assignment);
        *viol_sum += (i128::from(updated) - i128::from(con_viol[c])) * i128::from(weights[c]);
        con_viol[c] = updated;
    }
    *current = score;
    #[cfg(debug_assertions)]
    {
        let mut check = assignment.to_vec();
        let (full, _, full_complete) = model.score_breakdown(&mut check, weights);
        debug_assert_eq!(full, *current, "incremental LS score drifted from full recompute");
        debug_assert_eq!(full_complete, *complete_now, "LS completeness cache drifted");
        debug_assert_eq!(
            full.violation,
            combine_violation(i128::from(!*complete_now) * 1_000_000, *viol_sum),
            "LS violation-sum cache drifted"
        );
    }
}

fn refresh_score(
    model: &LocalModel,
    assignment: &mut [i32],
    weights: &[i64],
    current: &mut Score,
    con_viol: &mut Vec<i64>,
    complete_now: &mut bool,
    viol_sum: &mut i128,
) {
    let (score, next_con_viol, complete) = model.score_breakdown(assignment, weights);
    *current = score;
    *con_viol = next_con_viol;
    *complete_now = complete;
    *viol_sum = weighted_sum(con_viol, weights);
}

fn bump_gls_weights(weights: &mut [i64], con_viol: &[i64], viol_sum: &mut i128, current: &mut Score, complete_now: bool) -> bool {
    if !complete_now || current.violation <= 0 {
        return false;
    }
    let mut bumped = false;
    for (c, weight) in weights.iter_mut().enumerate() {
        if con_viol[c] > 0 {
            *weight += 1;
            bumped = true;
        }
    }
    if bumped {
        *viol_sum = weighted_sum(con_viol, weights);
        *current = Score { violation: combine_violation(0, *viol_sum), objective: current.objective };
    }
    bumped
}

pub(crate) fn solve_fast_cop<F>(
    problem: Problem,
    spec: LocalSearchSpec,
    stop: &AtomicBool,
    seed: u64,
    config: LsConfig,
    mut on_improve: F,
) -> LocalSearchOutcome
where
    F: FnMut(i64, &[i32], &'static str),
{
    let unsupported = spec.unsupported();
    let constraints = spec.constraints.len();
    let functionals = spec.functionals.len();
    let Ok(model) = LocalModel::new(problem, spec) else {
        return LocalSearchOutcome {
            best: None,
            iterations: 0,
            moves: 0,
            restarts: 0,
            constraints,
            functionals,
            unsupported: unsupported + 1,
        };
    };
    if unsupported > 0 {
        return LocalSearchOutcome { best: None, iterations: 0, moves: 0, restarts: 0, constraints, functionals, unsupported };
    }
    if model.mutable.is_empty() {
        let mut assignment = model.min_assignment();
        let best = if model.score(&mut assignment).violation == 0 {
            model.objective_value(&assignment).map(|value| {
                let solution = model.search_solution(&assignment);
                on_improve(value, &solution, "local-search");
                (solution, value)
            })
        } else {
            None
        };
        return LocalSearchOutcome { best, iterations: 0, moves: 0, restarts: 0, constraints, functionals, unsupported };
    }

    let minimizing = model.objective.minimizing();
    let objective_kicks = model.objective_kicks();
    let constructive_start = model.has_extension();
    // GLS penalty weights, one per constraint (all 1 = unweighted min-conflicts).
    // Only mutated when `config.gls` is on (i.e. `--turbo`); they persist across
    // restarts so effort keeps accumulating on the genuinely hard constraints.
    let mut weights: Vec<i64> = vec![1; model.constraints.len()];
    let mut assignment = if constructive_start { model.constructive_assignment(seed) } else { model.random_assignment(seed) };
    let (mut current, mut con_viol, mut complete_now) = model.score_breakdown(&mut assignment, &weights);
    let mut viol_sum: i128 = weighted_sum(&con_viol, &weights);
    // Reusable scratch for trial moves; never reallocated inside the loop.
    let mut work = assignment.clone();
    // Min-conflicts (#1b): O(1) membership set per large domain, built once. Empty
    // when the toggle is off. `cand_values` is the reused candidate buffer.
    let value_sets: Vec<Option<HashSet<i32>>> = if config.min_conflicts {
        model.domains.iter().map(|d| (d.values.len() > MIN_CONFLICTS_FULL).then(|| d.values.iter().copied().collect())).collect()
    } else {
        Vec::new()
    };
    let mut cand_values: Vec<i32> = Vec::new();
    let mut source = if constructive_start { "constructive" } else { "local-search" };
    let mut best_solution: Option<(Vec<i32>, i64)> = None;
    let mut iterations = 0;
    let mut moves = 0;
    let mut restarts = 0;
    let mut stagnant = 0;
    let mut kick_bandit = KickBandit::new();

    while !stop.load(Ordering::Relaxed) {
        iterations += 1;
        if complete_now && current.violation == 0 {
            if let Some(value) = model.objective_value(&assignment) {
                if better_value(minimizing, value, best_solution.as_ref().map(|(_, v)| *v)) {
                    let solution = model.search_solution(&assignment);
                    on_improve(value, &solution, source);
                    best_solution = Some((solution, value));
                }
            }
        }

        if config.kick_bandit {
            // Restart is a stagnation fallback, not a bandit arm: as a reward-
            // competing operator it always "moves" onto a fresh optimum and gets
            // over-selected, discarding accumulated search (it regressed coverage
            // 9 -> 5/7). Fire it only when genuinely stuck; the bandit chooses
            // among the refine operators (Repair / Objective / Constructive).
            if stagnant >= RESTART_AFTER {
                restarts += 1;
                let restart_seed = seed ^ mix64(iterations);
                let mut trial = if constructive_start {
                    model.constructive_assignment(restart_seed)
                } else {
                    model.random_assignment(restart_seed)
                };
                let (mut trial_score, mut trial_con_viol, mut trial_complete) = model.score_breakdown(&mut trial, &weights);
                let mut min_trial = model.min_assignment();
                let (min_score, min_con_viol, min_complete) = model.score_breakdown(&mut min_trial, &weights);
                if min_score < trial_score {
                    trial = min_trial;
                    trial_score = min_score;
                    trial_con_viol = min_con_viol;
                    trial_complete = min_complete;
                }
                assignment = trial;
                current = trial_score;
                con_viol = trial_con_viol;
                complete_now = trial_complete;
                viol_sum = weighted_sum(&con_viol, &weights);
                source = if constructive_start { "constructive" } else { "local-search" };
                stagnant = 0;
                continue;
            }

            let mut available = Vec::with_capacity(KICK_OPERATOR_COUNT);
            available.push(KickOperator::Repair);
            if !objective_kicks.is_empty() {
                available.push(KickOperator::Objective);
            }
            if constructive_start {
                available.push(KickOperator::Constructive);
            }

            let op = kick_bandit.select(&available, seed, iterations);
            let before = current;
            let mut moved = false;
            match op {
                KickOperator::Repair => {
                    // Descent-only: apply the best single-variable move only when it
                    // improves. Otherwise leave it to the escape operators (and the
                    // GLS bump below). Applying worsening moves here made `Repair`
                    // look bad to the bandit and degenerated it into restart-spam.
                    if let Some((score, j, value)) = best_single_variable_move(
                        &model,
                        &assignment,
                        &mut work,
                        &weights,
                        &con_viol,
                        viol_sum,
                        minimizing,
                        config,
                        &value_sets,
                        &mut cand_values,
                        seed,
                        iterations,
                    ) {
                        if score < current {
                            apply_single_variable_move(
                                &model,
                                &mut assignment,
                                &mut con_viol,
                                &mut viol_sum,
                                &mut complete_now,
                                &mut current,
                                &weights,
                                score,
                                j,
                                value,
                            );
                            source = op.source();
                            moves += 1;
                            moved = true;
                        }
                    }
                }
                KickOperator::Objective => {
                    if let Some(mut trial) = objective_kick_trial(&model, &assignment, &objective_kicks, seed, iterations) {
                        refresh_score(&model, &mut trial, &weights, &mut current, &mut con_viol, &mut complete_now, &mut viol_sum);
                        assignment = trial;
                        source = op.source();
                        moves += 1;
                        moved = true;
                    }
                }
                KickOperator::Constructive => {
                    let mut trial = model.constructive_assignment(seed ^ mix64(iterations));
                    let (score, trial_con_viol, trial_complete) = model.score_breakdown(&mut trial, &weights);
                    if score < current {
                        assignment = trial;
                        con_viol = trial_con_viol;
                        viol_sum = weighted_sum(&con_viol, &weights);
                        complete_now = trial_complete;
                        current = score;
                        source = op.source();
                        moves += 1;
                        moved = true;
                    }
                }
            }
            kick_bandit.record(op, kick_reward(before, current));
            if moved {
                stagnant = 0;
            } else {
                stagnant += 1;
                if config.gls {
                    bump_gls_weights(&mut weights, &con_viol, &mut viol_sum, &mut current, complete_now);
                }
            }
            continue;
        }

        if stagnant >= RESTART_AFTER {
            restarts += 1;
            let restart_seed = seed ^ mix64(iterations);
            assignment =
                if constructive_start { model.constructive_assignment(restart_seed) } else { model.random_assignment(restart_seed) };
            (current, con_viol, complete_now) = model.score_breakdown(&mut assignment, &weights);
            viol_sum = weighted_sum(&con_viol, &weights);
            source = if constructive_start { "constructive" } else { "local-search" };
            stagnant = 0;
            continue;
        }

        if constructive_start && iterations.is_multiple_of(CONSTRUCTIVE_KICK_PERIOD) {
            let mut trial = model.constructive_assignment(seed ^ mix64(iterations));
            let (score, trial_con_viol, trial_complete) = model.score_breakdown(&mut trial, &weights);
            if score < current {
                assignment = trial;
                con_viol = trial_con_viol;
                viol_sum = weighted_sum(&con_viol, &weights);
                complete_now = trial_complete;
                current = score;
                source = "constructive";
                moves += 1;
                stagnant = 0;
                continue;
            }
        }

        if let Some(trial) = objective_kick(&model, &assignment, &objective_kicks, seed, iterations) {
            assignment = trial;
            (current, con_viol, complete_now) = model.score_breakdown(&mut assignment, &weights);
            viol_sum = weighted_sum(&con_viol, &weights);
            source = "local-search";
            moves += 1;
            stagnant = 0;
            continue;
        }

        // Incremental delta scoring: evaluate `x_j := value` by re-running only the
        // constraints in `affected[j]` against a reused `work` buffer, instead of
        // cloning the assignment and rescoring every constraint. `complete()` still
        // runs in full (functional targets are folded into `affected[j]`, so the
        // delta over those constraints is exact). See TURBO.md §4.1 / first PR.
        let best_move = best_single_variable_move(
            &model,
            &assignment,
            &mut work,
            &weights,
            &con_viol,
            viol_sum,
            minimizing,
            config,
            &value_sets,
            &mut cand_values,
            seed,
            iterations,
        );

        let random_walk = iterations.is_multiple_of(RANDOM_WALK_PERIOD);
        let mut moved = false;
        if let Some((score, j, value)) = best_move {
            if score < current || random_walk {
                apply_single_variable_move(
                    &model,
                    &mut assignment,
                    &mut con_viol,
                    &mut viol_sum,
                    &mut complete_now,
                    &mut current,
                    &weights,
                    score,
                    j,
                    value,
                );
                source = "local-search";
                moves += 1;
                moved = true;
            }
        }

        if moved {
            stagnant = 0;
        } else {
            stagnant += 1;
            // Guided Local Search: stuck at a local minimum while still infeasible.
            // Penalise every currently-violated constraint, reshaping the weighted
            // landscape so the next descent is pushed toward the hard constraints.
            if config.gls {
                bump_gls_weights(&mut weights, &con_viol, &mut viol_sum, &mut current, complete_now);
            }
        }
    }

    LocalSearchOutcome { best: best_solution, iterations, moves, restarts, constraints, functionals, unsupported }
}
