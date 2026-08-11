//! Incumbent-only local search for the `--ls` COP engine.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::constraints::linear::Relation;
use crate::constraints::table::{Dfa, Mdd, STAR};
use crate::expr::Expr;
use crate::ids::VarId;
use crate::mix64;
use crate::problem::{Objective, Problem};

pub const MAX_DOMAIN_VALUES: usize = 4096;
/// Move-candidate samples drawn from a range-only domain (one too large to
/// materialise), alongside the two endpoints, per candidate evaluation.
const RANGE_SAMPLE_VALUES: usize = 32;
const MAX_SAMPLED_VARS: usize = 48;
const RANDOM_WALK_PERIOD: u64 = 17;
const RESTART_AFTER: u64 = 200;
const CONSTRUCTIVE_KICK_PERIOD: u64 = 5;
const SEQUENCE_PLACEMENT_NODE_LIMIT: usize = 20_000;
const SEQUENCE_PLACEMENT_EVALUATION_NODE_LIMIT: usize = 1_000;
const DYNAMIC_SEQUENCE_ORDER_LIMIT: usize = 20;
const LARGE_SHARED_ARRAY_DYNAMIC_SEQUENCE_CANDIDATES: usize = 8;
const MAX_SHARED_ARRAY_CELLS: usize = u64::BITS as usize;
const MAX_SHARED_ARRAY_PAIR_CELLS: usize = 16;
const SHARED_ARRAY_ANNEAL_STEPS: usize = 50_000;
const SHARED_ARRAY_ANNEAL_CYCLE: usize = 5_000;
const SHARED_ARRAY_HILL_CLIMB_ROUNDS: usize = 2;
const SHARED_ARRAY_PAIR_ROUNDS: usize = 4;
/// Min-conflicts (#1b): domains with at most this many values are scanned in full
/// (cheap and optimal); larger ones use the bounded candidate set.
const MIN_CONFLICTS_FULL: usize = 24;
/// Min-conflicts (#1b): random domain samples added to the candidate set on a
/// large domain, alongside the current value and any structural suggestions.
const MIN_CONFLICTS_SAMPLES: usize = 8;

#[derive(Clone)]
pub(crate) enum LocalConstraint {
    Selected {
        selector: VarId,
        constraint: Box<LocalConstraint>,
    },
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
        durations: Vec<LocalRhs>,
        heights: Vec<LocalRhs>,
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
pub enum LocalRhs {
    Const(i64),
    Var(VarId),
}

#[derive(Clone, Default)]
pub struct LocalSearchSpec {
    constraints: Vec<LocalConstraint>,
    functionals: Vec<Functional>,
    derived: Vec<bool>,
    /// Variables owned by the semantic model, as opposed to physical helper
    /// variables introduced while posting constraints.
    decisions: Vec<bool>,
    unsupported: usize,
    suppress_functionals: bool,
}

/// Behaviour toggles for the local-search engine selected by `--ls`.
#[derive(Clone, Copy, Default)]
pub struct LsConfig {
    /// Guided Local Search: at a local minimum, penalise the still-violated
    /// constraints (bump their weights) so search is pushed off the plateau and
    /// toward the genuinely hard constraints.
    pub(crate) gls: bool,
    /// Min-conflicts value selection: on large domains, evaluate a small candidate
    /// set (current value + structure-suggested values + random samples) instead
    /// of scanning the whole domain, so more variables are tried per iteration.
    /// Only bites when a domain exceeds `MIN_CONFLICTS_FULL`.
    pub(crate) min_conflicts: bool,
    /// Adaptive operator selection over the existing LS kicks.
    pub(crate) kick_bandit: bool,
}

pub struct LocalSearchOutcome {
    pub(crate) best: Option<(Vec<i32>, i64)>,
    #[allow(dead_code, reason = "retained for the structured LS report migration")]
    pub(crate) iterations: u64,
    #[allow(dead_code, reason = "retained for the structured LS report migration")]
    pub(crate) moves: u64,
    #[allow(dead_code, reason = "retained for the structured LS report migration")]
    pub(crate) restarts: u64,
    #[allow(dead_code, reason = "retained for the structured LS report migration")]
    pub(crate) constraints: usize,
    #[allow(dead_code, reason = "retained for the structured LS report migration")]
    pub(crate) functionals: usize,
    #[allow(dead_code, reason = "retained for the structured LS report migration")]
    pub(crate) unsupported: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Score {
    violation: i64,
    objective: i64,
}

// Restart is deliberately NOT a bandit operator - see the stagnation fallback in
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

    fn first_value_except(&self, excluded: i32) -> Option<i32> {
        if self.values.is_empty() {
            [self.min, self.max].into_iter().find(|&value| value != excluded)
        } else {
            self.values.iter().copied().find(|&value| value != excluded)
        }
    }

    fn is_bool(&self) -> bool {
        self.contains(0) && self.contains(1) && self.min_value() == 0 && self.max_value() == 1
    }

    /// Whether this variable offers more than one value to try: an explicit
    /// domain with >1 member, or a range-only domain (too large to materialise)
    /// spanning more than a single point.
    fn is_searchable(&self) -> bool {
        if self.values.is_empty() {
            self.min < self.max
        } else {
            self.values.len() > 1
        }
    }

    /// Move-candidate values for a range-only domain (`values` empty): both
    /// endpoints plus a bounded set of uniform interior draws over `[min,max]`.
    /// Every emitted value lies in `[min,max]`, which for a contiguous range
    /// (verified in [`LocalModel::new`]) is a genuine domain member. Fills `out`;
    /// callers with a materialised domain iterate `values` directly instead.
    fn sample_range(&self, seed: u64, out: &mut Vec<i32>) {
        out.clear();
        out.push(self.min);
        out.push(self.max);
        let span = i64::from(self.max) - i64::from(self.min);
        if span > 0 {
            let span = span as u64;
            for k in 0..RANGE_SAMPLE_VALUES as u64 {
                let pick = mix64(seed.wrapping_add(k.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
                out.push((i64::from(self.min) + (pick % (span + 1)) as i64) as i32);
            }
        }
        out.sort_unstable();
        out.dedup();
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

/// A strictly recognized sum of squared bilinear-product groups over signs.
/// The plan stores only the algebraic incidence needed by incremental deltas;
/// it assumes no ordering, distance pattern, or completeness of the graph.
struct SignedProductSquaresPlan {
    signs: Vec<VarId>,
    groups: Vec<Vec<(usize, usize)>>,
    incidence: Vec<Vec<GroupIncidence>>,
}

struct GroupIncidence {
    group: usize,
    neighbors: Vec<usize>,
}

// Specialized shared-array sequence-placement descriptor. The orchestrator's
// cheap prefilter may request this path, but descriptors are emitted only from
// an exact local-IR reconstruction. Near-matches stay on generic LS, and every
// candidate is canonically scored and replayed by CP before publication.
struct GuardedSequence<'a> {
    guard: VarId,
    weight: i64,
    shared_array: &'a [VarId],
    symbols: Vec<(VarId, i32)>,
    max_mismatches: usize,
}

type ElementViews<'a> = HashMap<VarId, (&'a [VarId], VarId, i32)>;
type Placement = (VarId, i32);

struct TrialPlacement {
    assignment: Vec<i32>,
    placements: Vec<Placement>,
    new_cells: usize,
}

#[derive(Clone, Copy)]
enum SequencePlacementPriority {
    Reuse,
    Weighted { cell_penalty: i64 },
}

struct CompiledGuardedSequence {
    weight: i64,
    max_mismatches: usize,
    expected: Vec<i32>,
    allowed_cells: Vec<u64>,
    transitions: Vec<Vec<u64>>,
}

struct GuardedSequenceEvaluator {
    values: Vec<i32>,
    sequences: Vec<CompiledGuardedSequence>,
    sequences_by_symbol: HashMap<i32, Vec<usize>>,
    active: Vec<bool>,
    score: i64,
}

impl GuardedSequenceEvaluator {
    fn new(model: &LocalModel, assignment: &[i32], sequences: &[GuardedSequence<'_>]) -> Option<Self> {
        let shared_array = sequences.first()?.shared_array;
        if shared_array.is_empty()
            || shared_array.len() > MAX_SHARED_ARRAY_CELLS
            || sequences
                .iter()
                .any(|sequence| sequence.shared_array != shared_array || sequence.max_mismatches > 1 || sequence.symbols.is_empty())
        {
            return None;
        }

        let values = shared_array.iter().map(|var| assignment[var.index()]).collect::<Vec<_>>();
        let mut compiled_sequences = Vec::with_capacity(sequences.len());
        let mut sequences_by_symbol: HashMap<i32, Vec<usize>> = HashMap::new();
        for (sequence_index, sequence) in sequences.iter().enumerate() {
            let expected = sequence.symbols.iter().map(|&(_, value)| value).collect::<Vec<_>>();
            for &value in &expected {
                sequences_by_symbol.entry(value).or_default().push(sequence_index);
            }
            let allowed_cells = sequence
                .symbols
                .iter()
                .map(|&(index, _)| {
                    (0..shared_array.len()).fold(0u64, |mask, cell| {
                        if model.domains[index.index()].contains(cell as i32) {
                            mask | (1u64 << cell)
                        } else {
                            mask
                        }
                    })
                })
                .collect::<Vec<_>>();
            let mut transitions = vec![Vec::new(); sequence.symbols.len()];
            for (position, transition) in transitions.iter_mut().enumerate().skip(1) {
                let left = sequence.symbols[position - 1].0;
                let right = sequence.symbols[position].0;
                *transition = (0..shared_array.len())
                    .map(|from| {
                        (0..shared_array.len()).fold(0u64, |mask, to| {
                            if from != to && model.pair_allowed(left, right, from as i32, to as i32) {
                                mask | (1u64 << to)
                            } else {
                                mask
                            }
                        })
                    })
                    .collect();
            }
            compiled_sequences.push(CompiledGuardedSequence {
                weight: sequence.weight,
                max_mismatches: sequence.max_mismatches,
                expected,
                allowed_cells,
                transitions,
            });
        }
        for indices in sequences_by_symbol.values_mut() {
            indices.sort_unstable();
            indices.dedup();
        }

        let mut evaluator =
            Self { values, sequences: compiled_sequences, sequences_by_symbol, active: vec![false; sequences.len()], score: 0 };
        evaluator.rebuild();
        Some(evaluator)
    }

    fn rebuild(&mut self) {
        self.score = 0;
        for sequence_index in 0..self.sequences.len() {
            let present = self.sequence_present(sequence_index);
            self.active[sequence_index] = present;
            if present {
                self.score = self.score.saturating_add(self.sequences[sequence_index].weight);
            }
        }
    }

    fn set_values(&mut self, values: &[i32]) {
        self.values.copy_from_slice(values);
        self.rebuild();
    }

    fn score_for_values(&self, values: &[i32]) -> i64 {
        (0..self.sequences.len())
            .filter(|&sequence_index| self.sequence_present_in(sequence_index, values))
            .fold(0i64, |score, sequence_index| score.saturating_add(self.sequences[sequence_index].weight))
    }

    fn apply_value(&mut self, cell: usize, value: i32) -> i64 {
        let old_value = self.values[cell];
        if old_value == value {
            return 0;
        }
        let mut affected = self.sequences_by_symbol.get(&old_value).cloned().unwrap_or_default();
        if let Some(indices) = self.sequences_by_symbol.get(&value) {
            affected.extend(indices.iter().copied());
        }
        affected.sort_unstable();
        affected.dedup();

        self.values[cell] = value;
        let mut delta = 0i64;
        for sequence_index in affected {
            let present = self.sequence_present(sequence_index);
            if present == self.active[sequence_index] {
                continue;
            }
            let weight = self.sequences[sequence_index].weight;
            delta = if present { delta.saturating_add(weight) } else { delta.saturating_sub(weight) };
            self.active[sequence_index] = present;
        }
        self.score = self.score.saturating_add(delta);
        delta
    }

    fn sequence_present(&self, sequence_index: usize) -> bool {
        self.sequence_present_in(sequence_index, &self.values)
    }

    fn sequence_present_in(&self, sequence_index: usize, values: &[i32]) -> bool {
        let sequence = &self.sequences[sequence_index];
        let matching = sequence.allowed_cells[0]
            & values
                .iter()
                .enumerate()
                .fold(0u64, |mask, (cell, &value)| if value == sequence.expected[0] { mask | (1u64 << cell) } else { mask });
        let mut exact = matching;
        let mut mismatch = if sequence.max_mismatches == 1 { sequence.allowed_cells[0] & !matching } else { 0 };
        for position in 1..sequence.expected.len() {
            let expanded_exact = expand_cells(exact, &sequence.transitions[position]);
            let expanded_mismatch = expand_cells(mismatch, &sequence.transitions[position]);
            let matching = sequence.allowed_cells[position]
                & values.iter().enumerate().fold(
                    0u64,
                    |mask, (cell, &value)| {
                        if value == sequence.expected[position] {
                            mask | (1u64 << cell)
                        } else {
                            mask
                        }
                    },
                );
            let mismatching = sequence.allowed_cells[position] & !matching;
            mismatch = (expanded_mismatch & matching) | if sequence.max_mismatches == 1 { expanded_exact & mismatching } else { 0 };
            exact = expanded_exact & matching;
            if exact | mismatch == 0 {
                return false;
            }
        }
        exact | mismatch != 0
    }
}

fn expand_cells(mut cells: u64, transitions: &[u64]) -> u64 {
    let mut expanded = 0u64;
    while cells != 0 {
        let from = cells.trailing_zeros() as usize;
        expanded |= transitions.get(from).copied().unwrap_or(0);
        cells &= cells - 1;
    }
    expanded
}

struct GuardedLexPlan {
    shared_array: Vec<VarId>,
    deferred_constraints: Vec<usize>,
    permutations: Vec<Vec<usize>>,
    guarded_indices: Vec<(VarId, VarId)>,
}

struct SequencePlacementState<'run, 'model> {
    assignment: &'run [i32],
    sequence: &'run GuardedSequence<'model>,
    ignored_constraints: &'run [usize],
    values: Vec<Option<i32>>,
    cells: Vec<usize>,
    nodes: usize,
    placement_seed: u64,
    stop: &'run AtomicBool,
    best: Option<TrialPlacement>,
}

struct SequenceTraceState<'a> {
    cells: Vec<usize>,
    nodes: usize,
    stop: &'a AtomicBool,
}

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
        LocalConstraint::Selected { selector, constraint } => {
            out.push(*selector);
            constraint_vars(constraint, out);
        }
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
            out.extend(durations.iter().filter_map(|value| match value {
                LocalRhs::Var(variable) => Some(*variable),
                LocalRhs::Const(_) => None,
            }));
            out.extend(heights.iter().filter_map(|value| match value {
                LocalRhs::Var(variable) => Some(*variable),
                LocalRhs::Const(_) => None,
            }));
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

fn constraint_has_lex_touching(constraint: &LocalConstraint, variables: &HashSet<VarId>) -> bool {
    match constraint {
        LocalConstraint::Lex { rows, .. } => rows.iter().flatten().any(|var| variables.contains(var)),
        LocalConstraint::Selected { constraint, .. } => constraint_has_lex_touching(constraint, variables),
        _ => false,
    }
}

fn mismatch_symbols(functionals: &[Functional], counter: VarId) -> Option<Vec<(VarId, i32)>> {
    let Functional::Linear { coeff, terms, rhs, .. } = functionals.iter().find(|functional| functional_target(functional) == counter)?
    else {
        return None;
    };
    if *coeff != -1 || *rhs != 0 || terms.is_empty() || terms.iter().any(|&(coefficient, _)| coefficient != 1) {
        return None;
    }

    let mut symbols = Vec::with_capacity(terms.len());
    for &(_, mismatch) in terms {
        let Functional::Expr { expr, .. } = functionals.iter().find(|functional| functional_target(functional) == mismatch)? else {
            return None;
        };
        symbols.push(ne_var_any_const(expr)?);
    }
    Some(symbols)
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
    pub(crate) fn may_have_signed_product_square_objective(problem: &Problem, stop: &AtomicBool) -> bool {
        problem
            .objective
            .as_ref()
            .filter(|objective| objective.minimizing())
            .and_then(|objective| squared_objective_variables(objective, Some(stop)))
            .is_some()
    }

    pub(crate) fn begin_guarded_constraints(&mut self) -> usize {
        self.suppress_functionals = true;
        self.constraints.len()
    }

    pub(crate) fn finish_guarded_constraints(&mut self, start: usize, selector: VarId) {
        self.suppress_functionals = false;
        for constraint in &mut self.constraints[start..] {
            let inner = constraint.clone();
            *constraint = LocalConstraint::Selected { selector, constraint: Box::new(inner) };
        }
    }

    pub(crate) fn is_derived(&self, variable: VarId) -> bool {
        self.derived.get(variable.index()).copied().unwrap_or(false)
    }

    pub(crate) fn is_decision(&self, variable: VarId) -> bool {
        self.decisions.get(variable.index()).copied().unwrap_or(false)
    }

    pub(crate) fn has_guarded_mismatch_structure(&self) -> bool {
        self.constraints.iter().any(|constraint| {
            let LocalConstraint::Expr(expr) = constraint else {
                return false;
            };
            guarded_mismatch_bound(expr).is_some_and(|(_, counter, _)| mismatch_symbols(&self.functionals, counter).is_some())
        })
    }

    pub(crate) fn has_guarded_sequence_primitives(&self) -> bool {
        let has_element = self.functionals.iter().any(|functional| matches!(functional, Functional::Element { .. }));
        let has_guard = self.constraints.iter().any(|constraint| match constraint {
            LocalConstraint::Extension { vars, tuples } => guarded_value(vars, tuples).is_some(),
            LocalConstraint::Expr(expr) => {
                guarded_expr_value(expr).is_some()
                    || guarded_mismatch_bound(expr).is_some_and(|(_, counter, _)| mismatch_symbols(&self.functionals, counter).is_some())
            }
            _ => false,
        });
        has_element && has_guard
    }

    pub(crate) fn has_signed_product_square_structure(&self, problem: &Problem) -> bool {
        self.has_signed_product_square_structure_interruptible(problem, &AtomicBool::new(false))
    }

    pub(crate) fn has_signed_product_square_structure_interruptible(&self, problem: &Problem, stop: &AtomicBool) -> bool {
        recognizes_signed_product_square_structure(problem, self, stop)
    }

    pub fn add_var(&mut self, var: VarId) {
        self.ensure(var);
        self.decisions[var.index()] = true;
    }

    pub fn add_expr(&mut self, expr: Expr) {
        if !self.suppress_functionals {
            if let Some(functional) = functional_from_expr(&expr, &self.derived) {
                self.mark_functional(functional);
            }
        }
        self.constraints.push(LocalConstraint::Expr(expr));
    }

    pub fn add_linear(&mut self, coeffs: Vec<i64>, vars: Vec<VarId>, rel: Relation, rhs: i64) {
        if !self.suppress_functionals {
            if let Some(functional) = functional_from_linear(&coeffs, &vars, rel, rhs, &self.derived) {
                self.mark_functional(functional);
            }
        }
        self.constraints.push(LocalConstraint::Linear { coeffs, vars, rel, rhs });
    }

    pub fn add_all_different(&mut self, vars: Vec<VarId>) {
        self.constraints.push(LocalConstraint::AllDifferent(vars));
    }

    pub fn add_all_different_rows(&mut self, rows: Vec<Vec<VarId>>) {
        self.constraints.push(LocalConstraint::AllDifferentRows(rows));
    }

    pub fn add_all_different_except(&mut self, vars: Vec<VarId>, except: Vec<i32>) {
        self.constraints.push(LocalConstraint::AllDifferentExcept { vars, except });
    }

    pub fn add_all_equal(&mut self, vars: Vec<VarId>) {
        self.constraints.push(LocalConstraint::AllEqual(vars));
    }

    pub fn add_extension(&mut self, vars: Vec<VarId>, tuples: Vec<Vec<i32>>, positive: bool) {
        if positive {
            if !self.suppress_functionals {
                if let Some(functional) = functional_from_extension(&vars, &tuples) {
                    self.mark_functional(functional);
                }
            }
            self.constraints.push(LocalConstraint::Extension { vars, tuples });
        } else {
            self.constraints.push(LocalConstraint::NegExtension { vars, tuples });
        }
    }

    pub fn add_lex_chain(&mut self, rows: Vec<Vec<VarId>>, strict: bool) {
        self.constraints.push(LocalConstraint::Lex { rows, strict });
    }

    pub fn add_count(&mut self, vars: Vec<VarId>, values: Vec<i32>, rel: Relation, rhs: LocalRhs) {
        self.constraints.push(LocalConstraint::Count { vars, values, rel, rhs });
    }

    pub fn add_count_allowed(&mut self, vars: Vec<VarId>, values: Vec<i32>, allowed: Vec<i32>) {
        self.constraints.push(LocalConstraint::CountAllowed { vars, values, allowed });
    }

    pub fn add_n_values(&mut self, vars: Vec<VarId>, rel: Relation, rhs: LocalRhs) {
        self.constraints.push(LocalConstraint::NValues { vars, rel, rhs });
    }

    pub fn add_cardinality(&mut self, vars: Vec<VarId>, values: Vec<i32>, low: Vec<i64>, high: Vec<i64>, closed: bool) {
        self.constraints.push(LocalConstraint::Cardinality { vars, values, low, high, closed });
    }

    pub fn add_extremum(&mut self, vars: Vec<VarId>, is_min: bool, rel: Relation, rhs: LocalRhs) {
        self.constraints.push(LocalConstraint::Extremum { vars, is_min, rel, rhs });
    }

    pub fn add_element_member(&mut self, array: Vec<VarId>, value: i32) {
        self.constraints.push(LocalConstraint::ElementMember { array, value });
    }

    pub fn add_element(&mut self, array: Vec<VarId>, index: VarId, target: VarId, start_index: i32) {
        if self.suppress_functionals {
            self.mark_unsupported();
        } else {
            self.mark_functional(Functional::Element { target, array, index, start_index });
        }
    }

    pub fn add_cumulative(&mut self, starts: Vec<VarId>, durations: Vec<VarId>, heights: Vec<VarId>, cap: LocalRhs) {
        self.add_cumulative_rhs(
            starts,
            durations.into_iter().map(LocalRhs::Var).collect(),
            heights.into_iter().map(LocalRhs::Var).collect(),
            cap,
        );
    }

    pub fn add_cumulative_rhs(&mut self, starts: Vec<VarId>, durations: Vec<LocalRhs>, heights: Vec<LocalRhs>, cap: LocalRhs) {
        self.constraints.push(LocalConstraint::Cumulative { starts, durations, heights, cap });
    }

    pub fn add_channel_inverse(&mut self, xs: Vec<VarId>, x_start: i32, ys: Vec<VarId>, y_start: i32) {
        self.constraints.push(LocalConstraint::ChannelInverse { xs, x_start, ys, y_start });
    }

    pub fn add_channel_onehot(&mut self, xs: Vec<VarId>, value: VarId, start_index: i32) {
        self.constraints.push(LocalConstraint::ChannelOneHot { xs, value, start_index });
    }

    pub fn add_precedence(&mut self, vars: Vec<VarId>, values: Vec<i32>, covered: bool) {
        self.constraints.push(LocalConstraint::Precedence { vars, values, covered });
    }

    pub fn add_circuit(&mut self, vars: Vec<VarId>) {
        self.constraints.push(LocalConstraint::Circuit(vars));
    }

    pub fn add_bin_packing(&mut self, items: Vec<VarId>, sizes: Vec<i64>, limits: Vec<LocalRhs>, exact: bool) {
        self.constraints.push(LocalConstraint::BinPacking { items, sizes, limits, exact });
    }

    pub fn add_no_overlap(&mut self, origins: Vec<Vec<VarId>>, lengths: Vec<Vec<Expr>>, zero_ignored: bool) {
        self.constraints.push(LocalConstraint::NoOverlap { origins, lengths, zero_ignored });
    }

    pub fn add_regular(&mut self, vars: Vec<VarId>, dfa: Dfa) {
        self.constraints.push(LocalConstraint::Regular { vars, dfa });
    }

    pub fn add_mdd(&mut self, vars: Vec<VarId>, mdd: Mdd) {
        self.constraints.push(LocalConstraint::Mdd { vars, mdd });
    }

    pub(crate) fn unsupported(&self) -> usize {
        self.unsupported
    }

    /// Flag a constraint with no local-search encoding. The engine then declines to
    /// run (returns no incumbent) rather than searching an incomplete model. Only
    /// the Python front-end currently posts constraints the LS model can't represent
    /// (channel, knapsack); the XCSP builder covers everything it posts.
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn mark_unsupported(&mut self) {
        self.unsupported += 1;
    }

    fn ensure(&mut self, var: VarId) {
        if self.derived.len() <= var.index() {
            self.derived.resize(var.index() + 1, false);
        }
        if self.decisions.len() <= var.index() {
            self.decisions.resize(var.index() + 1, false);
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
    fn new(problem: &Problem, spec: LocalSearchSpec) -> Result<Self, usize> {
        let objective = problem.objective.clone().ok_or(1usize)?;
        let mut domains = Vec::with_capacity(problem.solver.store.num_vars());
        for i in 0..problem.solver.store.num_vars() {
            let var = VarId(i as u32);
            let min = problem.solver.store.min(var);
            let max = problem.solver.store.max(var);
            let size = problem.solver.store.size(var);
            let values = if size <= MAX_DOMAIN_VALUES { problem.solver.store.values(var).collect::<Vec<_>>() } else { Vec::new() };
            if values.is_empty() && min > max {
                return Err(1);
            }
            // A range-only (unmaterialised) domain is sampled over [min,max] during
            // search, which is sound only if the range has no holes. `store.values`
            // filters holes, so any holey domain small enough is materialised
            // exactly; a large one is only left range-only when it is contiguous.
            debug_assert!(
                !values.is_empty() || i64::from(max) - i64::from(min) + 1 == size as i64,
                "range-only LS domain has holes; [min,max] sampling would emit removed values"
            );
            domains.push(LocalDomain { min, max, values });
        }
        let mutable = problem
            .search
            .iter()
            .copied()
            .filter(|&var| domains[var.index()].is_searchable() && !spec.derived.get(var.index()).copied().unwrap_or(false))
            .collect();
        let constraints = spec.constraints;
        let functionals = order_functionals(spec.functionals);
        let bool_tables = bool_tables(&functionals);
        let exact_covers = exact_cover_rows(&constraints, &domains);
        let affected = build_affected(domains.len(), &constraints, &functionals);
        Ok(Self {
            domains,
            mutable,
            search: problem.search.clone(),
            objective,
            constraints,
            functionals,
            bool_tables,
            exact_covers,
            affected,
        })
    }

    fn random_assignment(&self, seed: u64) -> Vec<i32> {
        self.domains.iter().enumerate().map(|(i, domain)| domain.initial_value(seed ^ i as u64)).collect()
    }

    fn min_assignment(&self) -> Vec<i32> {
        self.domains.iter().map(LocalDomain::min_value).collect()
    }

    fn constructive_assignment(&self, seed: u64, stop: &AtomicBool) -> Vec<i32> {
        let mut assignment = self.min_assignment();
        if stop.load(Ordering::Relaxed) {
            return assignment;
        }
        self.greedy_exact_cover(&mut assignment);
        if stop.load(Ordering::Relaxed) {
            return assignment;
        }
        let _ = self.complete(&mut assignment);
        self.place_guarded_sequences(assignment, seed, stop)
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
            Objective::VarWithAffine(_, _, coeffs, vars) | Objective::Linear(_, coeffs, vars) => {
                coeffs.iter().copied().zip(vars.iter().copied()).collect()
            }
            Objective::Expr(_, expr) => expr
                .affine_form_interruptible(&AtomicBool::new(false))
                .map_or_else(Vec::new, |(_, coeffs, vars)| coeffs.into_iter().zip(vars).collect()),
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

    fn place_guarded_sequences(&self, assignment: Vec<i32>, seed: u64, stop: &AtomicBool) -> Vec<i32> {
        let elements = self.element_views();
        let mut sequences = self.guarded_sequences(&elements);
        if sequences.is_empty() {
            return assignment;
        }
        if seed == 0 {
            sequences.sort_by(|a, b| b.weight.cmp(&a.weight).then_with(|| a.guard.cmp(&b.guard)));
        } else {
            sequences.sort_by(|a, b| {
                let ak = mix64(seed ^ a.guard.index() as u64);
                let bk = mix64(seed ^ b.guard.index() as u64);
                let aw = i128::from(a.weight) * i128::from(512 + (ak & 1023));
                let bw = i128::from(b.weight) * i128::from(512 + (bk & 1023));
                bw.cmp(&aw).then_with(|| a.guard.cmp(&b.guard))
            });
        }
        let fallback = assignment.clone();
        let lex_plan = self.guarded_lex_plan(&sequences);
        let deferred_lex = lex_plan.as_ref().map_or(&[][..], |plan| plan.deferred_constraints.as_slice());
        let mismatch_aware = sequences.iter().any(|sequence| sequence.max_mismatches > 0);
        let priorities = if mismatch_aware && seed != 0 {
            let mut unit_weights = sequences
                .iter()
                .filter(|sequence| sequence.weight > 0)
                .map(|sequence| sequence.weight / i64::try_from(sequence.symbols.len()).unwrap_or(1).max(1))
                .collect::<Vec<_>>();
            unit_weights.sort_unstable();
            let unit = unit_weights.get(unit_weights.len() / 2).copied().unwrap_or(1).max(1);
            let multiplier = [1i64, 3, 6][seed as usize % 3];
            vec![SequencePlacementPriority::Weighted { cell_penalty: unit.saturating_mul(multiplier) }]
        } else {
            vec![SequencePlacementPriority::Reuse]
        };

        let mut best: Option<(i64, Vec<i32>, Vec<i32>)> = None;
        for priority in priorities {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let raw = self.construct_guarded_array(assignment.clone(), &sequences, seed, priority, deferred_lex, stop);
            let Some(finalized) = self.finalize_guarded_candidate(raw.clone(), &sequences, lex_plan.as_ref(), stop) else {
                continue;
            };
            let objective = self.objective_value(&finalized).unwrap_or(if self.objective.minimizing() { i64::MAX } else { i64::MIN });
            if best.as_ref().is_none_or(|(old, _, _)| better_value(self.objective.minimizing(), objective, Some(*old))) {
                best = Some((objective, raw, finalized));
            }
        }

        let Some((_, raw, finalized)) = best else {
            return fallback;
        };
        if stop.load(Ordering::Relaxed) {
            return finalized;
        }
        let improved_raw = self.improve_guarded_array(raw, &sequences, seed, mismatch_aware, stop);
        let Some(improved) = self.finalize_guarded_candidate(improved_raw, &sequences, lex_plan.as_ref(), stop) else {
            return finalized;
        };
        let old_value = self.objective_value(&finalized);
        let new_value = self.objective_value(&improved);
        match (old_value, new_value) {
            (Some(old), Some(new)) if better_value(self.objective.minimizing(), new, Some(old)) => improved,
            _ => finalized,
        }
    }

    fn construct_guarded_array(
        &self,
        mut assignment: Vec<i32>,
        sequences: &[GuardedSequence<'_>],
        seed: u64,
        priority: SequencePlacementPriority,
        deferred_lex: &[usize],
        stop: &AtomicBool,
    ) -> Vec<i32> {
        let mut fixed = vec![None; self.domains.len()];
        let mut remaining = (0..sequences.len()).collect::<Vec<_>>();
        let dynamic_order =
            matches!(priority, SequencePlacementPriority::Weighted { .. }) || sequences.len() <= DYNAMIC_SEQUENCE_ORDER_LIMIT;

        while !remaining.is_empty() {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let candidate_positions = if dynamic_order {
                let mut positions = (0..remaining.len()).collect::<Vec<_>>();
                positions.sort_unstable_by(|&left, &right| {
                    sequences[remaining[right]]
                        .weight
                        .cmp(&sequences[remaining[left]].weight)
                        .then_with(|| sequences[remaining[left]].guard.cmp(&sequences[remaining[right]].guard))
                });
                if sequences.first().is_some_and(|sequence| sequence.shared_array.len() > MAX_SHARED_ARRAY_CELLS) {
                    positions.truncate(LARGE_SHARED_ARRAY_DYNAMIC_SEQUENCE_CANDIDATES);
                }
                positions
            } else {
                vec![0]
            };
            let mut selected: Option<(usize, usize, TrialPlacement)> = None;
            for &remaining_pos in &candidate_positions {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let sequence_index = remaining[remaining_pos];
                let sequence = &sequences[sequence_index];
                let placement_seed = if seed == 0 { 0 } else { mix64(seed ^ sequence.guard.index() as u64 ^ remaining.len() as u64) };
                let Some(candidate) = self.try_place_sequence(&assignment, &fixed, sequence, placement_seed, deferred_lex, stop) else {
                    continue;
                };
                let better = selected.as_ref().is_none_or(|(_, best_sequence_index, best)| {
                    self.guarded_placement_better(priority, sequence, &candidate, &sequences[*best_sequence_index], best)
                });
                if better {
                    selected = Some((remaining_pos, sequence_index, candidate));
                }
            }
            let Some((remaining_pos, _, candidate)) = selected else {
                if dynamic_order {
                    let failed = candidate_positions.first().copied().unwrap_or(0);
                    remaining.remove(failed);
                    continue;
                }
                remaining.remove(0);
                continue;
            };
            for (var, value) in candidate.placements {
                fixed[var.index()] = Some(value);
            }
            assignment = candidate.assignment;
            remaining.remove(remaining_pos);
        }
        assignment
    }

    fn guarded_placement_better(
        &self,
        priority: SequencePlacementPriority,
        sequence: &GuardedSequence<'_>,
        candidate: &TrialPlacement,
        best_sequence: &GuardedSequence<'_>,
        best: &TrialPlacement,
    ) -> bool {
        let primary = match priority {
            SequencePlacementPriority::Reuse => i128::try_from(usize::MAX - candidate.new_cells).unwrap_or(i128::MAX),
            SequencePlacementPriority::Weighted { cell_penalty } => {
                i128::from(sequence.weight) - i128::from(cell_penalty) * candidate.new_cells as i128
            }
        };
        let best_primary = match priority {
            SequencePlacementPriority::Reuse => i128::try_from(usize::MAX - best.new_cells).unwrap_or(i128::MAX),
            SequencePlacementPriority::Weighted { cell_penalty } => {
                i128::from(best_sequence.weight) - i128::from(cell_penalty) * best.new_cells as i128
            }
        };
        primary > best_primary
            || (primary == best_primary
                && (candidate.new_cells < best.new_cells
                    || (candidate.new_cells == best.new_cells
                        && (sequence.weight > best_sequence.weight
                            || (sequence.weight == best_sequence.weight && sequence.guard < best_sequence.guard)))))
    }

    fn finalize_guarded_candidate(
        &self,
        assignment: Vec<i32>,
        sequences: &[GuardedSequence<'_>],
        lex_plan: Option<&GuardedLexPlan>,
        stop: &AtomicBool,
    ) -> Option<Vec<i32>> {
        let ignored = lex_plan.map_or(&[][..], |plan| plan.deferred_constraints.as_slice());
        let materialized = self.materialize_guarded_array(assignment, sequences, ignored, stop)?;
        let mut finalized = match lex_plan {
            Some(plan) => self.canonicalize_guarded_array(&materialized, plan, stop)?,
            None => materialized,
        };
        if finalized.len() != self.domains.len()
            || finalized.iter().zip(&self.domains).any(|(&value, domain)| !domain.contains(value))
            || self.score(&mut finalized).violation != 0
        {
            return None;
        }
        Some(finalized)
    }

    fn guarded_lex_plan(&self, sequences: &[GuardedSequence<'_>]) -> Option<GuardedLexPlan> {
        let shared_array = sequences.first()?.shared_array;
        if shared_array.is_empty()
            || sequences.iter().any(|sequence| sequence.shared_array != shared_array)
            || i32::try_from(shared_array.len() - 1).is_err()
        {
            return None;
        }
        let positions = shared_array.iter().copied().enumerate().map(|(index, var)| (var, index)).collect::<HashMap<_, _>>();
        if positions.len() != shared_array.len() {
            return None;
        }
        let shared_array_vars = positions.keys().copied().collect::<HashSet<_>>();
        let mut deferred_constraints = Vec::new();
        let mut permutations = vec![(0..shared_array.len()).collect::<Vec<_>>()];
        for (constraint_index, constraint) in self.constraints.iter().enumerate() {
            match constraint {
                LocalConstraint::Lex { rows, strict } => {
                    let touches_shared_array = rows.iter().flatten().any(|var| shared_array_vars.contains(var));
                    if !touches_shared_array {
                        continue;
                    }
                    if *strict || rows.len() != 2 || rows[0].as_slice() != shared_array || rows[1].len() != shared_array.len() {
                        return None;
                    }
                    let mut seen = vec![false; shared_array.len()];
                    let mut permutation = Vec::with_capacity(shared_array.len());
                    for var in &rows[1] {
                        let &position = positions.get(var)?;
                        if seen[position] {
                            return None;
                        }
                        seen[position] = true;
                        permutation.push(position);
                    }
                    if !seen.into_iter().all(|present| present) {
                        return None;
                    }
                    deferred_constraints.push(constraint_index);
                    if !permutations.contains(&permutation) {
                        permutations.push(permutation);
                    }
                }
                LocalConstraint::Selected { constraint, .. } if constraint_has_lex_touching(constraint, &shared_array_vars) => return None,
                _ => {}
            }
        }
        if deferred_constraints.is_empty() {
            return None;
        }

        let mut index_guards = HashMap::new();
        for sequence in sequences {
            for &(index, _) in &sequence.symbols {
                if index_guards.insert(index, sequence.guard).is_some_and(|guard| guard != sequence.guard) {
                    return None;
                }
            }
        }
        let mut guarded_indices = Vec::new();
        for functional in &self.functionals {
            let Functional::Element { array, index, start_index, .. } = functional else {
                continue;
            };
            if !array.iter().any(|var| shared_array_vars.contains(var)) {
                continue;
            }
            if array.as_slice() != shared_array || *start_index != 0 {
                return None;
            }
            guarded_indices.push((*index, *index_guards.get(index)?));
        }
        guarded_indices.sort_unstable();
        guarded_indices.dedup();
        if guarded_indices.is_empty()
            || sequences
                .iter()
                .flat_map(|sequence| sequence.symbols.iter().map(|&(index, _)| index))
                .any(|index| guarded_indices.binary_search_by_key(&index, |&(candidate, _)| candidate).is_err())
        {
            return None;
        }

        Some(GuardedLexPlan { shared_array: shared_array.to_vec(), deferred_constraints, permutations, guarded_indices })
    }

    fn canonicalize_guarded_array(&self, assignment: &[i32], plan: &GuardedLexPlan, stop: &AtomicBool) -> Option<Vec<i32>> {
        let old_shared_array = plan.shared_array.iter().map(|var| assignment[var.index()]).collect::<Vec<_>>();
        let mut best: Option<(Vec<i32>, Vec<i32>)> = None;
        for permutation in &plan.permutations {
            if stop.load(Ordering::Relaxed) || permutation.len() != plan.shared_array.len() {
                return None;
            }
            let mut inverse = vec![usize::MAX; permutation.len()];
            let mut trial = assignment.to_vec();
            let mut in_domain = true;
            for (new_position, &old_position) in permutation.iter().enumerate() {
                if old_position >= old_shared_array.len() || inverse[old_position] != usize::MAX {
                    in_domain = false;
                    break;
                }
                inverse[old_position] = new_position;
                let variable = plan.shared_array[new_position];
                let value = old_shared_array[old_position];
                if !self.domains[variable.index()].contains(value) {
                    in_domain = false;
                    break;
                }
                trial[variable.index()] = value;
            }
            if !in_domain || inverse.contains(&usize::MAX) {
                continue;
            }
            for &(index, guard) in &plan.guarded_indices {
                if assignment[guard.index()] != 1 {
                    continue;
                }
                let Ok(old_position) = usize::try_from(assignment[index.index()]) else {
                    in_domain = false;
                    break;
                };
                let Some(&new_position) = inverse.get(old_position) else {
                    in_domain = false;
                    break;
                };
                let Ok(value) = i32::try_from(new_position) else {
                    in_domain = false;
                    break;
                };
                if !self.domains[index.index()].contains(value) {
                    in_domain = false;
                    break;
                }
                trial[index.index()] = value;
            }
            if !in_domain || self.score(&mut trial).violation != 0 {
                continue;
            }
            let shared_array_values = plan.shared_array.iter().map(|var| trial[var.index()]).collect::<Vec<_>>();
            if best.as_ref().is_none_or(|(best_shared_array, _)| shared_array_values < *best_shared_array) {
                best = Some((shared_array_values, trial));
            }
        }
        best.map(|(_, assignment)| assignment)
    }

    fn improve_guarded_array(
        &self,
        mut assignment: Vec<i32>,
        sequences: &[GuardedSequence<'_>],
        seed: u64,
        mismatch_aware: bool,
        stop: &AtomicBool,
    ) -> Vec<i32> {
        let Some(shared_array) = sequences.first().map(|sequence| sequence.shared_array) else {
            return assignment;
        };
        if !mismatch_aware
            || shared_array.len() > MAX_SHARED_ARRAY_CELLS
            || sequences.iter().any(|sequence| sequence.shared_array != shared_array)
            || stop.load(Ordering::Relaxed)
        {
            return assignment;
        }

        let mut alphabet = sequences.iter().flat_map(|sequence| sequence.symbols.iter().map(|&(_, value)| value)).collect::<Vec<_>>();
        alphabet.extend(shared_array.iter().map(|variable| assignment[variable.index()]));
        alphabet.sort_unstable();
        alphabet.dedup();
        if alphabet.is_empty() {
            return assignment;
        }

        let Some(mut evaluator) = GuardedSequenceEvaluator::new(self, &assignment, sequences) else {
            return assignment;
        };
        let mut best_values = evaluator.values.clone();
        let mut best_score = evaluator.score;
        let mut weights = sequences.iter().map(|sequence| sequence.weight.max(1)).collect::<Vec<_>>();
        weights.sort_unstable();
        let temperature_scale = weights.get(weights.len() / 2).copied().unwrap_or(1).max(1);

        // Each cycle starts from the best known shared_array at a high temperature and
        // cools deterministically. This makes portfolio seeds meaningfully
        // different while keeping a seed exactly reproducible.
        for step in 0..SHARED_ARRAY_ANNEAL_STEPS {
            if step.is_multiple_of(256) && stop.load(Ordering::Relaxed) {
                break;
            }
            let cycle_step = step % SHARED_ARRAY_ANNEAL_CYCLE;
            if cycle_step == 0 {
                evaluator.set_values(&best_values);
            }
            let random = mix64(seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA076_1D64_78BD_642F);
            let cell = random as usize % shared_array.len();
            let value = alphabet[mix64(random ^ 0xE703_7ED1_A0B4_28DB) as usize % alphabet.len()];
            if value == evaluator.values[cell] || !self.domains[shared_array[cell].index()].contains(value) {
                continue;
            }

            let old_value = evaluator.values[cell];
            let delta = evaluator.apply_value(cell, value);
            let accept = if delta >= 0 {
                true
            } else {
                let remaining = SHARED_ARRAY_ANNEAL_CYCLE - cycle_step;
                let temperature = temperature_scale.saturating_mul(i64::try_from(remaining).unwrap_or(i64::MAX))
                    / i64::try_from(SHARED_ARRAY_ANNEAL_CYCLE).unwrap_or(1);
                let loss = delta.saturating_neg();
                let range = temperature.saturating_add(loss).max(1) as u64;
                mix64(random ^ 0x8EBC_6AF0_9C88_C6E3) % range < temperature.max(0) as u64
            };
            if !accept {
                evaluator.apply_value(cell, old_value);
                continue;
            }
            if evaluator.score > best_score {
                best_score = evaluator.score;
                best_values.clone_from(&evaluator.values);
            }
        }

        evaluator.set_values(&best_values);
        for _ in 0..SHARED_ARRAY_HILL_CLIMB_ROUNDS {
            let mut changed = false;
            for (cell, &variable) in shared_array.iter().enumerate() {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let old_value = evaluator.values[cell];
                let mut chosen_value = old_value;
                let mut chosen_score = evaluator.score;
                for &value in &alphabet {
                    if value == old_value || !self.domains[variable.index()].contains(value) {
                        continue;
                    }
                    evaluator.apply_value(cell, value);
                    if evaluator.score > chosen_score {
                        chosen_score = evaluator.score;
                        chosen_value = value;
                    }
                    evaluator.apply_value(cell, old_value);
                }
                if chosen_value != old_value {
                    evaluator.apply_value(cell, chosen_value);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // Single-cell local optima are common on the 3x3 and 4x4 variants.
        // Exhaustive two-cell moves are still small at this deliberately tight
        // shared_array limit and recover improvements that a blind random walk misses.
        let pair_rounds = if shared_array.len() <= MAX_SHARED_ARRAY_PAIR_CELLS { SHARED_ARRAY_PAIR_ROUNDS } else { 0 };
        for _ in 0..pair_rounds {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let baseline = evaluator.score;
            let mut best_pair: Option<(i64, usize, i32, usize, i32)> = None;
            for left in 0..shared_array.len() {
                let old_left = evaluator.values[left];
                for &left_value in &alphabet {
                    if left_value == old_left || !self.domains[shared_array[left].index()].contains(left_value) {
                        continue;
                    }
                    evaluator.apply_value(left, left_value);
                    for (right, &right_variable) in shared_array.iter().enumerate().skip(left + 1) {
                        let old_right = evaluator.values[right];
                        for &right_value in &alphabet {
                            if right_value == old_right || !self.domains[right_variable.index()].contains(right_value) {
                                continue;
                            }
                            evaluator.apply_value(right, right_value);
                            if evaluator.score > baseline && best_pair.as_ref().is_none_or(|(score, ..)| evaluator.score > *score) {
                                best_pair = Some((evaluator.score, left, left_value, right, right_value));
                            }
                            evaluator.apply_value(right, old_right);
                        }
                    }
                    evaluator.apply_value(left, old_left);
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                }
                if stop.load(Ordering::Relaxed) {
                    break;
                }
            }
            let Some((_, left, left_value, right, right_value)) = best_pair else {
                break;
            };
            evaluator.apply_value(left, left_value);
            evaluator.apply_value(right, right_value);
        }

        if evaluator.score > best_score {
            best_values.clone_from(&evaluator.values);
        }
        for (position, &variable) in shared_array.iter().enumerate() {
            let value = best_values[position];
            if self.domains[variable.index()].contains(value) {
                assignment[variable.index()] = value;
            }
        }
        assignment
    }

    fn materialize_guarded_array(
        &self,
        mut assignment: Vec<i32>,
        sequences: &[GuardedSequence<'_>],
        ignored_constraints: &[usize],
        stop: &AtomicBool,
    ) -> Option<Vec<i32>> {
        for sequence in sequences {
            if !self.domains[sequence.guard.index()].contains(0) {
                return None;
            }
            assignment[sequence.guard.index()] = 0;
            for &(index, _) in &sequence.symbols {
                let domain = &self.domains[index.index()];
                assignment[index.index()] = if domain.contains(0) { 0 } else { domain.min_value() };
            }
        }
        for sequence in sequences {
            let Some(cells) = self.trace_guarded_sequence(&assignment, sequence, stop) else {
                continue;
            };
            if !self.domains[sequence.guard.index()].contains(1) {
                return None;
            }
            assignment[sequence.guard.index()] = 1;
            for (&cell, &(index, _)) in cells.iter().zip(&sequence.symbols) {
                let value = i32::try_from(cell).ok()?;
                if !self.domains[index.index()].contains(value) {
                    return None;
                }
                assignment[index.index()] = value;
            }
        }
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        (self.score_ignoring_constraints(&mut assignment, ignored_constraints).violation == 0).then_some(assignment)
    }

    fn trace_guarded_sequence(&self, assignment: &[i32], sequence: &GuardedSequence<'_>, stop: &AtomicBool) -> Option<Vec<usize>> {
        let mut state = SequenceTraceState { cells: vec![0; sequence.symbols.len()], nodes: SEQUENCE_PLACEMENT_NODE_LIMIT, stop };
        self.trace_guarded_symbols(assignment, sequence, 0, None, sequence.max_mismatches, &mut state).then_some(state.cells)
    }

    fn trace_guarded_symbols(
        &self,
        assignment: &[i32],
        sequence: &GuardedSequence<'_>,
        pos: usize,
        previous: Option<usize>,
        mismatches_left: usize,
        state: &mut SequenceTraceState<'_>,
    ) -> bool {
        if state.nodes == 0 || state.stop.load(Ordering::Relaxed) {
            return false;
        }
        state.nodes -= 1;
        if pos == sequence.symbols.len() {
            return true;
        }
        let (index, value) = sequence.symbols[pos];
        for cell in (0..sequence.shared_array.len()).rev() {
            if previous == Some(cell) || !self.domains[index.index()].contains(cell as i32) {
                continue;
            }
            if let Some(previous) = previous {
                let previous_index = sequence.symbols[pos - 1].0;
                if !self.pair_allowed(previous_index, index, previous as i32, cell as i32) {
                    continue;
                }
            }
            let mismatch = usize::from(assignment[sequence.shared_array[cell].index()] != value);
            if mismatch > mismatches_left {
                continue;
            }
            state.cells[pos] = cell;
            if self.trace_guarded_symbols(assignment, sequence, pos + 1, Some(cell), mismatches_left - mismatch, state) {
                return true;
            }
        }
        false
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

    // This reconstruction is the authoritative structural check. The semantic
    // orchestration check is intentionally only a cheap compilation prefilter.
    fn guarded_sequences<'a>(&self, elements: &ElementViews<'a>) -> Vec<GuardedSequence<'a>> {
        let mut requirements: BTreeMap<VarId, Vec<(VarId, i32)>> = BTreeMap::new();
        let mut max_mismatches = BTreeMap::new();
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
        for constraint in &self.constraints {
            let LocalConstraint::Expr(expr) = constraint else {
                continue;
            };
            let Some((guard, counter, allowed)) = guarded_mismatch_bound(expr) else {
                continue;
            };
            if requirements.contains_key(&guard) {
                continue;
            }
            let Some(symbols) = mismatch_symbols(&self.functionals, counter) else {
                continue;
            };
            requirements.insert(guard, symbols);
            max_mismatches.insert(guard, allowed);
        }

        let mut sequences = Vec::new();
        for (guard, reqs) in requirements {
            let Some(weight) = self.objective_weight(guard) else {
                continue;
            };
            let mut symbols = Vec::with_capacity(reqs.len());
            let mut array = None;
            for (target, value) in reqs {
                let Some(&(candidate_array, index, start_index)) = elements.get(&target) else {
                    symbols.clear();
                    break;
                };
                if start_index != 0 {
                    symbols.clear();
                    break;
                }
                if let Some(array) = array {
                    if array != candidate_array {
                        symbols.clear();
                        break;
                    }
                } else {
                    array = Some(candidate_array);
                }
                symbols.push((index, value));
            }
            if let Some(array) = array {
                if !symbols.is_empty() && self.domains[guard.index()].contains(1) {
                    symbols.sort_by_key(|&(index, _)| index.index());
                    let max_mismatches = max_mismatches.get(&guard).copied().unwrap_or(0).min(symbols.len());
                    sequences.push(GuardedSequence { guard, weight, shared_array: array, symbols, max_mismatches });
                }
            }
        }
        sequences
    }

    fn try_place_sequence(
        &self,
        assignment: &[i32],
        fixed: &[Option<i32>],
        sequence: &GuardedSequence<'_>,
        placement_seed: u64,
        ignored_constraints: &[usize],
        stop: &AtomicBool,
    ) -> Option<TrialPlacement> {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        let mut state = SequencePlacementState {
            assignment,
            sequence,
            ignored_constraints,
            values: vec![None; sequence.shared_array.len()],
            cells: vec![0; sequence.symbols.len()],
            nodes: SEQUENCE_PLACEMENT_EVALUATION_NODE_LIMIT,
            placement_seed,
            stop,
            best: None,
        };
        for (cell, &var) in sequence.shared_array.iter().enumerate() {
            state.values[cell] = fixed[var.index()];
        }
        self.place_sequence_symbols(0, None, sequence.max_mismatches, 0, &mut state);
        state.best
    }

    fn sequence_trial(
        &self,
        assignment: &[i32],
        sequence: &GuardedSequence<'_>,
        values: &[Option<i32>],
        cells: &[usize],
        new_cells: usize,
        ignored_constraints: &[usize],
    ) -> Option<TrialPlacement> {
        let mut trial = assignment.to_vec();
        let mut placements = Vec::new();
        trial[sequence.guard.index()] = 1;
        for (cell, &value) in values.iter().enumerate() {
            if let Some(value) = value {
                let x = sequence.shared_array[cell];
                if !self.domains[x.index()].contains(value) {
                    return None;
                }
                trial[x.index()] = value;
                placements.push((x, value));
            }
        }
        for (&cell, &(index, _)) in cells.iter().zip(&sequence.symbols) {
            trial[index.index()] = i32::try_from(cell).ok()?;
        }
        (self.score_ignoring_constraints(&mut trial, ignored_constraints).violation == 0).then_some(TrialPlacement {
            assignment: trial,
            placements,
            new_cells,
        })
    }

    fn lex_constraints_ok(&self, assignment: &[i32]) -> bool {
        self.constraints.iter().all(|constraint| match constraint {
            LocalConstraint::Lex { rows, strict } => lex_chain_violation(rows, *strict, assignment) == 0,
            LocalConstraint::Selected { selector, constraint }
                if assignment[selector.index()] == 1 && matches!(constraint.as_ref(), LocalConstraint::Lex { .. }) =>
            {
                let LocalConstraint::Lex { rows, strict } = constraint.as_ref() else { unreachable!() };
                lex_chain_violation(rows, *strict, assignment) == 0
            }
            _ => true,
        })
    }

    fn pair_allowed(&self, left: VarId, right: VarId, left_value: i32, right_value: i32) -> bool {
        if let Some(true_pairs) = self.bool_tables.get(&(left, right)) {
            return true_pairs.binary_search(&(left_value, right_value)).is_ok();
        }
        self.bool_tables.get(&(right, left)).is_none_or(|true_pairs| true_pairs.binary_search(&(right_value, left_value)).is_ok())
    }

    fn place_sequence_symbols(
        &self,
        pos: usize,
        previous: Option<usize>,
        mismatches_left: usize,
        new_cells: usize,
        state: &mut SequencePlacementState<'_, '_>,
    ) {
        if state.nodes == 0
            || state.stop.load(Ordering::Relaxed)
            || state.best.as_ref().is_some_and(|candidate| new_cells >= candidate.new_cells)
        {
            return;
        }
        state.nodes -= 1;
        if pos == state.sequence.symbols.len() {
            if let Some(candidate) =
                self.sequence_trial(state.assignment, state.sequence, &state.values, &state.cells, new_cells, state.ignored_constraints)
            {
                state.best = Some(candidate);
            }
            return;
        }
        let (index, value) = state.sequence.symbols[pos];
        let cell_count = state.sequence.shared_array.len();
        let offset = if state.placement_seed == 0 {
            0
        } else {
            mix64(state.placement_seed ^ pos as u64 ^ previous.unwrap_or(cell_count) as u64) as usize % cell_count
        };

        // Reusing an occupied matching cell has zero marginal cost and never
        // consumes the mismatch allowance. Explore those branches first.
        for rank in 0..cell_count {
            let cell = (offset + rank) % cell_count;
            if state.stop.load(Ordering::Relaxed) {
                return;
            }
            if !self.sequence_cell_allowed(state.sequence, pos, previous, index, cell) || state.values[cell] != Some(value) {
                continue;
            }
            self.descend_sequence_cell(pos, cell, None, mismatches_left, new_cells, state);
            if state.best.as_ref().is_some_and(|candidate| candidate.new_cells == new_cells) {
                return;
            }
        }

        // An already occupied mismatching cell also costs nothing, but spends
        // one unit of an explicitly recognised mismatch budget.
        if mismatches_left > 0 {
            for rank in 0..cell_count {
                let cell = (offset + rank) % cell_count;
                if state.stop.load(Ordering::Relaxed) {
                    return;
                }
                let Some(old) = state.values[cell] else {
                    continue;
                };
                if old == value || !self.sequence_cell_allowed(state.sequence, pos, previous, index, cell) {
                    continue;
                }
                self.descend_sequence_cell(pos, cell, None, mismatches_left - 1, new_cells, state);
                if state.best.as_ref().is_some_and(|candidate| candidate.new_cells == new_cells) {
                    return;
                }
            }
        }

        // A free cell whose current value is already a permitted mismatch can
        // be used without claiming it for later sequences.
        if mismatches_left > 0 {
            for rank in 0..cell_count {
                let cell = (offset + rank) % cell_count;
                if state.stop.load(Ordering::Relaxed) {
                    return;
                }
                let x = state.sequence.shared_array[cell];
                if state.values[cell].is_some()
                    || state.assignment[x.index()] == value
                    || !self.sequence_cell_allowed(state.sequence, pos, previous, index, cell)
                {
                    continue;
                }
                self.descend_sequence_cell(pos, cell, None, mismatches_left - 1, new_cells, state);
                if state.best.as_ref().is_some_and(|candidate| candidate.new_cells == new_cells) {
                    return;
                }
            }
        }

        // Exact placements on free cells add one newly occupied cell. The B&B
        // keeps searching after the first leaf and retains the cheapest one.
        for rank in 0..cell_count {
            let cell = (offset + rank) % cell_count;
            if state.stop.load(Ordering::Relaxed) {
                return;
            }
            let x = state.sequence.shared_array[cell];
            if state.values[cell].is_some()
                || !self.domains[x.index()].contains(value)
                || !self.sequence_cell_allowed(state.sequence, pos, previous, index, cell)
            {
                continue;
            }
            self.descend_sequence_cell(pos, cell, Some(value), mismatches_left, new_cells + 1, state);
        }

        // If the current free-cell value equals the requested symbol, create an
        // explicit alternative only when consuming a mismatch is allowed.
        if mismatches_left > 0 {
            for rank in 0..cell_count {
                let cell = (offset + rank) % cell_count;
                if state.stop.load(Ordering::Relaxed) {
                    return;
                }
                let x = state.sequence.shared_array[cell];
                if state.values[cell].is_some()
                    || state.assignment[x.index()] != value
                    || !self.sequence_cell_allowed(state.sequence, pos, previous, index, cell)
                {
                    continue;
                }
                let Some(other) = self.domains[x.index()].first_value_except(value) else {
                    continue;
                };
                self.descend_sequence_cell(pos, cell, Some(other), mismatches_left - 1, new_cells + 1, state);
            }
        }
    }

    fn sequence_cell_allowed(
        &self,
        sequence: &GuardedSequence<'_>,
        pos: usize,
        previous: Option<usize>,
        index: VarId,
        cell: usize,
    ) -> bool {
        if previous == Some(cell) || !self.domains[index.index()].contains(cell as i32) {
            return false;
        }
        previous.is_none_or(|previous| {
            let previous_index = sequence.symbols[pos - 1].0;
            self.pair_allowed(previous_index, index, previous as i32, cell as i32)
        })
    }

    fn descend_sequence_cell(
        &self,
        pos: usize,
        cell: usize,
        set_value: Option<i32>,
        mismatches_left: usize,
        new_cells: usize,
        state: &mut SequencePlacementState<'_, '_>,
    ) {
        let old = state.values[cell];
        if let Some(value) = set_value {
            state.values[cell] = Some(value);
        }
        state.cells[pos] = cell;
        self.place_sequence_symbols(pos + 1, Some(cell), mismatches_left, new_cells, state);
        state.values[cell] = old;
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
            Objective::Var(_, var) | Objective::VarWithAffine(_, var, _, _) => Some(assignment[var.index()] as i64),
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

    fn score_ignoring_constraints(&self, assignment: &mut [i32], ignored: &[usize]) -> Score {
        if ignored.is_empty() {
            return self.score(assignment);
        }
        let mut violation = i64::from(!self.complete(assignment)) * 1_000_000;
        for (index, constraint) in self.constraints.iter().enumerate() {
            if ignored.binary_search(&index).is_err() {
                violation = violation.saturating_add(self.violation(constraint, assignment));
            }
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
            LocalConstraint::Selected { selector, constraint } => {
                if assignment[selector.index()] == 1 {
                    self.violation(constraint, assignment)
                } else {
                    0
                }
            }
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

    /// Recognize a complete signed-product-square IR structure. Any relevant
    /// near-match is declined so the ordinary local-search loop handles it.
    fn signed_product_squares_plan(&self, stop: Option<&AtomicBool>) -> Option<SignedProductSquaresPlan> {
        if plan_interrupted(stop) || !self.objective.minimizing() || self.constraints.len() != self.functionals.len() {
            return None;
        }
        let square_targets = squared_objective_variables(&self.objective, stop)?;
        let square_set = square_targets.iter().copied().collect::<HashSet<_>>();
        let mutable = self.mutable.iter().copied().collect::<HashSet<_>>();

        // Every non-objective functional must define one bilinear sign variable product.
        // No constraint is silently ignored by this specialized plan.
        let mut products = HashMap::<VarId, (VarId, VarId)>::new();
        let mut signs = HashSet::<VarId>::new();
        for functional in &self.functionals {
            if plan_interrupted(stop) {
                return None;
            }
            let target = functional_target(functional);
            if square_set.contains(&target) {
                continue;
            }
            let Functional::Expr { expr, .. } = functional else {
                return None;
            };
            let (left, right) = binary_product(expr)?;
            if left == right
                || square_set.contains(&left)
                || square_set.contains(&right)
                || !mutable.contains(&left)
                || !mutable.contains(&right)
                || mutable.contains(&target)
                || !is_sign_domain(self.domains.get(left.index())?)
                || !is_sign_domain(self.domains.get(right.index())?)
                || !is_sign_domain(self.domains.get(target.index())?)
                || products.insert(target, (left, right)).is_some()
            {
                return None;
            }
            signs.extend([left, right]);
        }
        if signs.len() < 2 || products.is_empty() {
            return None;
        }

        // Each squared objective variable must be exactly the unit-coefficient
        // sum of one non-empty group of the products above. Requiring the full
        // parity range in its domain proves that the derived variable imposes no
        // hidden restriction on the sign variable assignments searched by the kernel.
        let mut seen_targets = HashSet::new();
        let mut used_products = HashSet::new();
        let mut variable_groups = Vec::<Vec<(VarId, VarId)>>::with_capacity(square_targets.len());
        for functional in &self.functionals {
            if plan_interrupted(stop) {
                return None;
            }
            let target = functional_target(functional);
            if !square_set.contains(&target) {
                continue;
            }
            if mutable.contains(&target) || !seen_targets.insert(target) {
                return None;
            }
            let product_targets = unit_product_sum_targets(functional)?;
            if product_targets.is_empty() || !domain_contains_unit_sum_range(self.domains.get(target.index())?, product_targets.len(), stop)
            {
                return None;
            }
            let mut group = Vec::with_capacity(product_targets.len());
            for product in product_targets {
                if plan_interrupted(stop) {
                    return None;
                }
                group.push(*products.get(&product)?);
                used_products.insert(product);
            }
            variable_groups.push(group);
        }
        if seen_targets != square_set
            || used_products.len() != products.len()
            || !products.keys().all(|target| used_products.contains(target))
        {
            return None;
        }

        let mut signs = signs.into_iter().collect::<Vec<_>>();
        signs.sort_unstable();
        let sign_indices = signs.iter().enumerate().map(|(index, &variable)| (variable, index)).collect::<HashMap<_, _>>();
        let mut groups = Vec::with_capacity(variable_groups.len());
        let mut incidence = (0..signs.len()).map(|_| BTreeMap::<usize, Vec<usize>>::new()).collect::<Vec<_>>();
        let mut incidence_visits = 0usize;
        for (group_index, group) in variable_groups.into_iter().enumerate() {
            if plan_interrupted(stop) {
                return None;
            }
            let mut indexed = Vec::with_capacity(group.len());
            for (left, right) in group {
                incidence_visits = incidence_visits.wrapping_add(1);
                if incidence_visits.is_multiple_of(1_024) && plan_interrupted(stop) {
                    return None;
                }
                let left = *sign_indices.get(&left)?;
                let right = *sign_indices.get(&right)?;
                if left == right {
                    return None;
                }
                indexed.push((left, right));
                incidence[left].entry(group_index).or_default().push(right);
                incidence[right].entry(group_index).or_default().push(left);
            }
            groups.push(indexed);
        }
        let incidence = incidence
            .into_iter()
            .map(|groups| groups.into_iter().map(|(group, neighbors)| GroupIncidence { group, neighbors }).collect())
            .collect();
        Some(SignedProductSquaresPlan { signs, groups, incidence })
    }

    fn search_solution(&self, assignment: &[i32]) -> Vec<i32> {
        self.search.iter().map(|&v| assignment[v.index()]).collect()
    }
}

fn squared_objective_variables(objective: &Objective, stop: Option<&AtomicBool>) -> Option<Vec<VarId>> {
    let Objective::Expr(true, expression) = objective else {
        return None;
    };
    let mut variables = Vec::new();
    let mut pending = vec![expression];
    let mut visits = 0usize;
    while let Some(expression) = pending.pop() {
        visits = visits.wrapping_add(1);
        if visits.is_multiple_of(1_024) && plan_interrupted(stop) {
            return None;
        }
        match expression {
            Expr::Add(terms) if !terms.is_empty() => pending.extend(terms),
            Expr::Mul(terms) => {
                let [Expr::Var(left), Expr::Var(right)] = terms.as_slice() else {
                    return None;
                };
                if left != right {
                    return None;
                }
                variables.push(*left);
            }
            _ => return None,
        }
    }
    if variables.is_empty() {
        return None;
    }
    let mut unique = variables.clone();
    unique.sort_unstable();
    unique.dedup();
    (unique.len() == variables.len()).then_some(unique)
}

fn recognizes_signed_product_square_structure(problem: &Problem, spec: &LocalSearchSpec, stop: &AtomicBool) -> bool {
    if stop.load(Ordering::Acquire) || spec.constraints.len() != spec.functionals.len() {
        return false;
    }
    let Some(objective) = problem.objective.as_ref().filter(|objective| objective.minimizing()) else {
        return false;
    };
    let Some(square_targets) = squared_objective_variables(objective, Some(stop)) else {
        return false;
    };
    let square_set = square_targets.iter().copied().collect::<HashSet<_>>();
    let mutable = problem
        .search
        .iter()
        .copied()
        .filter(|variable| problem.solver.store.size(*variable) > 1 && !spec.derived.get(variable.index()).copied().unwrap_or(false))
        .collect::<HashSet<_>>();

    let mut products = HashMap::<VarId, (VarId, VarId)>::new();
    let mut signs = HashSet::<VarId>::new();
    for functional in &spec.functionals {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        let target = functional_target(functional);
        if square_set.contains(&target) {
            continue;
        }
        let Functional::Expr { expr, .. } = functional else {
            return false;
        };
        let Some((left, right)) = binary_product(expr) else {
            return false;
        };
        if left == right
            || square_set.contains(&left)
            || square_set.contains(&right)
            || !mutable.contains(&left)
            || !mutable.contains(&right)
            || mutable.contains(&target)
            || !is_store_sign_domain(problem, left)
            || !is_store_sign_domain(problem, right)
            || !is_store_sign_domain(problem, target)
            || products.insert(target, (left, right)).is_some()
        {
            return false;
        }
        signs.extend([left, right]);
    }
    if signs.len() < 2 || products.is_empty() {
        return false;
    }

    let mut seen_targets = HashSet::new();
    let mut used_products = HashSet::new();
    for functional in &spec.functionals {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        let target = functional_target(functional);
        if !square_set.contains(&target) {
            continue;
        }
        let Some(product_targets) = unit_product_sum_targets(functional) else {
            return false;
        };
        if mutable.contains(&target)
            || !seen_targets.insert(target)
            || product_targets.is_empty()
            || !store_domain_contains_unit_sum_range(problem, target, product_targets.len(), stop)
        {
            return false;
        }
        for product in product_targets {
            if stop.load(Ordering::Acquire) || !products.contains_key(&product) {
                return false;
            }
            used_products.insert(product);
        }
    }
    seen_targets == square_set && used_products.len() == products.len() && products.keys().all(|target| used_products.contains(target))
}

fn is_store_sign_domain(problem: &Problem, variable: VarId) -> bool {
    problem.solver.store.size(variable) == 2
        && problem.solver.store.min(variable) == -1
        && problem.solver.store.max(variable) == 1
        && problem.solver.store.contains(variable, -1)
        && problem.solver.store.contains(variable, 1)
}

fn store_domain_contains_unit_sum_range(problem: &Problem, variable: VarId, terms: usize, stop: &AtomicBool) -> bool {
    let Ok(width) = i32::try_from(terms) else {
        return false;
    };
    let mut value = -i64::from(width);
    let maximum = i64::from(width);
    let mut visits = 0usize;
    while value <= maximum {
        visits = visits.wrapping_add(1);
        if visits.is_multiple_of(1_024) && stop.load(Ordering::Acquire) {
            return false;
        }
        let Some(converted) = i32::try_from(value).ok() else {
            return false;
        };
        if !problem.solver.store.contains(variable, converted) {
            return false;
        }
        value += 2;
    }
    !stop.load(Ordering::Acquire)
}

fn binary_product(expression: &Expr) -> Option<(VarId, VarId)> {
    let Expr::Mul(terms) = expression else {
        return None;
    };
    let [Expr::Var(left), Expr::Var(right)] = terms.as_slice() else {
        return None;
    };
    Some((*left, *right))
}

fn unit_product_sum_targets(functional: &Functional) -> Option<Vec<VarId>> {
    match functional {
        Functional::Expr { expr: Expr::Var(product), .. } => Some(vec![*product]),
        Functional::Linear { coeff, terms, rhs, .. } if *coeff != 0 && *rhs == 0 => {
            let mut products = Vec::with_capacity(terms.len());
            for &(term_coefficient, variable) in terms {
                let numerator = term_coefficient.checked_neg()?;
                if numerator % coeff != 0 || numerator / coeff != 1 {
                    return None;
                }
                products.push(variable);
            }
            Some(products)
        }
        _ => None,
    }
}

fn is_sign_domain(domain: &LocalDomain) -> bool {
    domain.min_value() == -1 && domain.max_value() == 1 && domain.values.len() == 2 && domain.contains(-1) && domain.contains(1)
}

fn plan_interrupted(stop: Option<&AtomicBool>) -> bool {
    stop.is_some_and(|stop| stop.load(Ordering::Relaxed))
}

fn domain_contains_unit_sum_range(domain: &LocalDomain, terms: usize, stop: Option<&AtomicBool>) -> bool {
    let Ok(width) = i32::try_from(terms) else {
        return false;
    };
    if domain.values.is_empty() {
        return domain.min <= -width && domain.max >= width;
    }
    let values = domain.values.iter().copied().collect::<HashSet<_>>();
    let mut value = -i64::from(width);
    let maximum = i64::from(width);
    while value <= maximum {
        if plan_interrupted(stop) || !i32::try_from(value).ok().is_some_and(|value| values.contains(&value)) {
            return false;
        }
        value += 2;
    }
    true
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

fn cumulative_violation(starts: &[VarId], durations: &[LocalRhs], heights: &[LocalRhs], cap: i64, assignment: &[i32]) -> i64 {
    let mut tasks = Vec::new();
    let mut points = Vec::new();
    for ((&start, &duration), &height) in starts.iter().zip(durations).zip(heights) {
        let s = assignment[start.index()] as i64;
        let d = duration.value(assignment);
        let h = height.value(assignment);
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

fn guarded_mismatch_bound(expr: &Expr) -> Option<(VarId, VarId, usize)> {
    let Expr::Or(terms) = expr else {
        return None;
    };
    let [left, right] = terms.as_slice() else {
        return None;
    };
    guarded_mismatch_parts(left, right).or_else(|| guarded_mismatch_parts(right, left))
}

fn guarded_mismatch_parts(guard: &Expr, bound: &Expr) -> Option<(VarId, VarId, usize)> {
    let guard = eq_var_const(guard, 0)?;
    let Expr::Le(left, right) = bound else {
        return None;
    };
    let (Expr::Var(counter), Expr::Const(limit)) = (&**left, &**right) else {
        return None;
    };
    Some((guard, *counter, usize::try_from(*limit).ok()?))
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

fn ne_var_any_const(expr: &Expr) -> Option<(VarId, i32)> {
    let Expr::Ne(a, b) = expr else {
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
        } else if model.domains[j].values.is_empty() {
            model.domains[j].sample_range(seed ^ iter ^ (j as u64), cand_values);
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

struct SignedProductSquaresState {
    signs: Vec<i32>,
    group_sums: Vec<i128>,
    energy: i128,
}

impl SignedProductSquaresState {
    fn new(plan: &SignedProductSquaresPlan, seed: u64, stop: &AtomicBool) -> Option<Self> {
        let mut signs = Vec::with_capacity(plan.signs.len());
        signs.push(1);
        for index in 1..plan.signs.len() {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let draw = mix64(seed.wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            signs.push(if draw & 1 == 0 { -1 } else { 1 });
        }
        let mut visits = 0usize;
        let mut group_sums = Vec::with_capacity(plan.groups.len());
        for group in &plan.groups {
            let mut sum = 0i128;
            for &(left, right) in group {
                visits = visits.wrapping_add(1);
                if visits.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
                    return None;
                }
                sum = sum.checked_add(i128::from(signs[left] * signs[right]))?;
            }
            group_sums.push(sum);
        }
        let energy = group_sums.iter().try_fold(0i128, |energy, &sum| energy.checked_add(sum.checked_mul(sum)?))?;
        Some(Self { signs, group_sums, energy })
    }

    fn flip_delta(
        &self,
        plan: &SignedProductSquaresPlan,
        index: usize,
        stop: &AtomicBool,
        changes: &mut Vec<(usize, i128)>,
    ) -> Option<i128> {
        changes.clear();
        let mut delta = 0i128;
        let mut visits = 0usize;
        for incidence in plan.incidence.get(index)? {
            let mut affected = 0i128;
            for &neighbor in &incidence.neighbors {
                visits = visits.wrapping_add(1);
                if visits.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
                    return None;
                }
                affected = affected.checked_add(i128::from(self.signs[index] * self.signs[neighbor]))?;
            }
            let change = affected.checked_mul(-2)?;
            let old = *self.group_sums.get(incidence.group)?;
            let group_delta = old.checked_mul(change)?.checked_mul(2)?.checked_add(change.checked_mul(change)?)?;
            delta = delta.checked_add(group_delta)?;
            changes.push((incidence.group, change));
        }
        Some(delta)
    }

    fn flip(&mut self, plan: &SignedProductSquaresPlan, index: usize, stop: &AtomicBool, changes: &mut Vec<(usize, i128)>) -> Option<()> {
        let delta = self.flip_delta(plan, index, stop, changes)?;
        let energy = self.energy.checked_add(delta)?;
        if changes.iter().any(|&(group, change)| self.group_sums.get(group).and_then(|sum| sum.checked_add(change)).is_none()) {
            return None;
        }
        for &(group, change) in changes.iter() {
            self.group_sums[group] = self.group_sums[group].checked_add(change).expect("validated signed-product group update");
        }
        self.signs[index] = -self.signs[index];
        self.energy = energy;
        debug_assert_eq!(
            self.energy,
            self.group_sums.iter().map(|&value| value * value).sum::<i128>(),
            "signed-product-square delta drifted from its group sums"
        );
        debug_assert!(plan.groups.iter().zip(&self.group_sums).all(|(group, &expected)| {
            group.iter().map(|&(left, right)| i128::from(self.signs[left] * self.signs[right])).sum::<i128>() == expected
        }));
        Some(())
    }
}

fn materialize_signed_product_squares(
    model: &LocalModel,
    plan: &SignedProductSquaresPlan,
    signs: &[i32],
    expected_energy: i128,
) -> Option<(Vec<i32>, i64)> {
    let expected_energy = i64::try_from(expected_energy).ok()?;
    let mut assignment = model.min_assignment();
    for (&variable, &value) in plan.signs.iter().zip(signs) {
        assignment[variable.index()] = value;
    }
    if model.score(&mut assignment).violation != 0 {
        return None;
    }
    let objective = model.objective_value(&assignment)?;
    if objective != expected_energy {
        return None;
    }
    Some((model.search_solution(&assignment), objective))
}

fn solve_signed_product_squares<F>(
    model: &LocalModel,
    plan: &SignedProductSquaresPlan,
    stop: &AtomicBool,
    seed: u64,
    max_iterations: u64,
    on_improve: &mut F,
) -> (Option<(Vec<i32>, i64)>, u64, u64, u64)
where
    F: FnMut(i64, &[i32], &'static str),
{
    if max_iterations == 0 || stop.load(Ordering::Relaxed) {
        return (None, 0, 0, 0);
    }

    let Some(mut state) = SignedProductSquaresState::new(plan, seed, stop) else {
        return (None, 0, 0, 0);
    };
    let mut best_energy = state.energy;
    let mut best_signs = state.signs.clone();
    let mut published_energy = None;
    let mut best_solution = None;
    let mut iterations = 0u64;
    let mut moves = 0u64;
    let mut perturbations = 0u64;
    let mut tied = Vec::new();
    let mut kick_indices = (1..state.signs.len()).collect::<Vec<_>>();
    let mut changes = Vec::new();

    'search: while iterations < max_iterations && !stop.load(Ordering::Relaxed) {
        let mut best_delta = None;
        tied.clear();
        // Global sign inversion preserves every bilinear product, so fixing the
        // first sign variable to +1 removes that symmetry without excluding an energy.
        for index in 1..state.signs.len() {
            if stop.load(Ordering::Relaxed) {
                break 'search;
            }
            let Some(delta) = state.flip_delta(plan, index, stop, &mut changes) else {
                break 'search;
            };
            match best_delta {
                None => {
                    best_delta = Some(delta);
                    tied.push(index);
                }
                Some(best) if delta < best => {
                    best_delta = Some(delta);
                    tied.clear();
                    tied.push(index);
                }
                Some(best) if delta == best => tied.push(index),
                Some(_) => {}
            }
        }
        let Some(best_delta) = best_delta else {
            break;
        };

        if best_delta < 0 {
            let draw = mix64(seed ^ iterations ^ (state.energy as u64).rotate_left(17));
            let index = tied[draw as usize % tied.len()];
            if state.flip(plan, index, stop, &mut changes).is_none() {
                break;
            }
            moves = moves.saturating_add(1);
            if state.energy < best_energy {
                best_energy = state.energy;
                best_signs.clone_from(&state.signs);
            }
        } else {
            if published_energy.is_none_or(|published| best_energy < published) {
                if let Some((solution, objective)) = materialize_signed_product_squares(model, plan, &best_signs, best_energy) {
                    on_improve(objective, &solution, "signed-product-squares");
                    published_energy = Some(best_energy);
                    best_solution = Some((solution, objective));
                }
            }

            let draw = mix64(seed ^ iterations.rotate_left(23) ^ perturbations.rotate_left(41));
            // Scale perturbations with the algebraic dimension. Logarithmic
            // kicks preserve most of a local minimum while still growing on
            // larger incidence networks.
            let available = state.signs.len() - 1;
            let kick_floor = usize::from(available > 1) + 1;
            let kick_ceiling = (state.signs.len().ilog2() as usize + 1).clamp(kick_floor, available);
            let kick_count = kick_floor + draw as usize % (kick_ceiling - kick_floor + 1);
            for offset in 0..kick_count {
                if stop.load(Ordering::Relaxed) {
                    break 'search;
                }
                let pick =
                    offset + mix64(draw ^ (offset as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)) as usize % (kick_indices.len() - offset);
                kick_indices.swap(offset, pick);
                if state.flip(plan, kick_indices[offset], stop, &mut changes).is_none() {
                    break 'search;
                }
                moves = moves.saturating_add(1);
                if state.energy < best_energy {
                    best_energy = state.energy;
                    best_signs.clone_from(&state.signs);
                }
            }
            perturbations = perturbations.saturating_add(1);
        }
        iterations = iterations.saturating_add(1);
    }

    if published_energy.is_none_or(|published| best_energy < published) {
        if let Some((solution, objective)) = materialize_signed_product_squares(model, plan, &best_signs, best_energy) {
            on_improve(objective, &solution, "signed-product-squares");
            best_solution = Some((solution, objective));
        }
    }
    (best_solution, iterations, moves, perturbations)
}

pub fn solve_ls<F>(
    problem: Problem,
    spec: LocalSearchSpec,
    stop: &AtomicBool,
    seed: u64,
    config: LsConfig,
    on_improve: F,
) -> LocalSearchOutcome
where
    F: FnMut(i64, &[i32], &'static str),
{
    solve_ls_capped(problem, spec, stop, seed, config, u64::MAX, on_improve)
}

pub(crate) fn solve_ls_capped<F>(
    problem: Problem,
    spec: LocalSearchSpec,
    stop: &AtomicBool,
    seed: u64,
    config: LsConfig,
    max_iterations: u64,
    on_improve: F,
) -> LocalSearchOutcome
where
    F: FnMut(i64, &[i32], &'static str),
{
    solve_ls_capped_borrowed(&problem, spec, stop, seed, config, max_iterations, on_improve)
}

pub(crate) fn solve_ls_capped_borrowed<F>(
    problem: &Problem,
    spec: LocalSearchSpec,
    stop: &AtomicBool,
    seed: u64,
    config: LsConfig,
    max_iterations: u64,
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

    if let Some(plan) = model.signed_product_squares_plan(Some(stop)) {
        let (best, iterations, moves, restarts) = solve_signed_product_squares(&model, &plan, stop, seed, max_iterations, &mut on_improve);
        return LocalSearchOutcome { best, iterations, moves, restarts, constraints, functionals, unsupported };
    }

    let minimizing = model.objective.minimizing();
    let objective_kicks = model.objective_kicks();
    let constructive_start = model.has_extension();
    // GLS penalty weights, one per constraint (all 1 = unweighted min-conflicts).
    // Only mutated when `config.gls` is on; they persist across
    // restarts so effort keeps accumulating on the genuinely hard constraints.
    let mut weights: Vec<i64> = vec![1; model.constraints.len()];
    let mut assignment = if constructive_start { model.constructive_assignment(seed, stop) } else { model.random_assignment(seed) };
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

    while iterations < max_iterations && !stop.load(Ordering::Relaxed) {
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
                    model.constructive_assignment(restart_seed, stop)
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
                    // improves. (Adding random-walk sideways moves here was tested and
                    // both diluted the exploration that helps timetabling and did not
                    // recover the descent-bound regression - net worse on both.)
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
                    let mut trial = model.constructive_assignment(seed ^ mix64(iterations), stop);
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
                if constructive_start { model.constructive_assignment(restart_seed, stop) } else { model.random_assignment(restart_seed) };
            (current, con_viol, complete_now) = model.score_breakdown(&mut assignment, &weights);
            viol_sum = weighted_sum(&con_viol, &weights);
            source = if constructive_start { "constructive" } else { "local-search" };
            stagnant = 0;
            continue;
        }

        if constructive_start && iterations.is_multiple_of(CONSTRUCTIVE_KICK_PERIOD) {
            let mut trial = model.constructive_assignment(seed ^ mix64(iterations), stop);
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
        // delta over those constraints is exact).
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
