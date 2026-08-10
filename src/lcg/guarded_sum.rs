//! Compact evaluation of sums of Boolean conjunctions.
//!
//! Generated models sometimes expose an objective as a very large
//! `Add(And(...), ...)` expression. Walking that boxed expression tree for
//! every objective-bound probe is needlessly expensive. This module recognizes
//! a deliberately small, exact subset and stores its literals and term offsets
//! in flat arrays.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::expr::Expr;
use crate::ids::VarId;
use crate::mix64;
use crate::store::Store;

#[cfg_attr(not(test), allow(dead_code))]
const MAX_SMALL_HINT_KICKS: usize = 2;
#[cfg_attr(not(test), allow(dead_code))]
const MAX_HINT_STARTS: usize = 4;
#[cfg_attr(not(test), allow(dead_code))]
const SMALL_HINT_VARIABLES: usize = 64;

/// A sum of conjunctions whose atoms compare one integer variable with a
/// constant.
///
/// `offsets[i]..offsets[i + 1]` is the literal slice of term `i`. Conjunctions
/// known to be false are discarded, while conjunctions known to be true are
/// accumulated in `constant`.
#[derive(Clone, Debug)]
pub(crate) struct GuardedSum {
    literals: Box<[Literal]>,
    offsets: Box<[u32]>,
    variables: Box<[VarId]>,
    #[cfg_attr(not(test), allow(dead_code))]
    incidence_offsets: Box<[u32]>,
    #[cfg_attr(not(test), allow(dead_code))]
    incidences: Box<[Incidence]>,
    constant: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Literal {
    variable: VarId,
    value: i32,
    equality: bool,
}

/// One variable's contiguous literal group inside one conjunction.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
struct Incidence {
    term: u32,
    start: u32,
    end: u32,
}

type IncidenceIndex = (Box<[u32]>, Box<[Incidence]>);

#[cfg_attr(not(test), allow(dead_code))]
struct HintState {
    values: Vec<i32>,
    group_true: Vec<bool>,
    false_groups: Vec<u32>,
    objective: i64,
}

#[cfg_attr(not(test), allow(dead_code))]
struct HintBudget<'a> {
    remaining: usize,
    stop: &'a AtomicBool,
}

#[cfg_attr(not(test), allow(dead_code))]
struct HintRng {
    state: u64,
}

#[cfg_attr(not(test), allow(dead_code))]
struct MoveEvaluation {
    delta: i64,
    group_true: Vec<bool>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Truth {
    False,
    Unknown,
    True,
}

enum ParsedLiteral {
    Constant(bool),
    Predicate(Literal),
}

enum ParsedTerm {
    False,
    True,
    Literals(Vec<Literal>),
}

#[cfg_attr(not(test), allow(dead_code))]
impl GuardedSum {
    /// Compile the exact expression subset handled by this representation.
    ///
    /// The root must be an [`Expr::Add`] and each summand must be an
    /// [`Expr::And`]. Accepted atoms are constants, `Var`, `Not(Var)`, and
    /// `Eq` or `Ne` between a variable and an integer constant in either
    /// operand order. Plain variables use the expression language's usual
    /// non-zero truth semantics. Any other shape returns `None` without a
    /// partial lowering.
    pub(crate) fn compile(expression: &Expr) -> Option<Self> {
        let Expr::Add(terms) = expression else {
            return None;
        };

        let mut literals = Vec::new();
        let mut offsets = vec![0u32];
        let mut variables = Vec::new();
        let mut constant = 0i64;

        for term in terms {
            let Expr::And(atoms) = term else {
                return None;
            };
            match compile_term(atoms)? {
                ParsedTerm::False => {}
                ParsedTerm::True => constant = constant.saturating_add(1),
                ParsedTerm::Literals(term_literals) => {
                    let next_len = literals.len().checked_add(term_literals.len())?;
                    let next_offset = u32::try_from(next_len).ok()?;
                    variables.extend(term_literals.iter().map(|literal| literal.variable));
                    literals.extend(term_literals);
                    offsets.push(next_offset);
                }
            }
        }

        variables.sort_unstable();
        variables.dedup();
        let (incidence_offsets, incidences) = build_incidences(&literals, &offsets, &variables)?;
        Some(Self {
            literals: literals.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
            variables: variables.into_boxed_slice(),
            incidence_offsets,
            incidences,
            constant,
        })
    }

    /// Unique variables mentioned by the retained conjunctions, sorted by id.
    pub(crate) fn vars(&self) -> &[VarId] {
        &self.variables
    }

    /// Sound interval bounds under the domains currently in `store`.
    ///
    /// A conjunction contributes zero when one literal is impossible, one when
    /// all literals are entailed, and `[0, 1]` otherwise. Correlations between
    /// distinct conjunctions are intentionally not inferred.
    pub(crate) fn bounds(&self, store: &Store) -> (i64, i64) {
        self.bounds_for(store, None)
    }

    /// Sound interval bounds when `variable` is hypothetically fixed to
    /// `value`. Callers normally pass a value still contained in its domain.
    pub(crate) fn bounds_with_value(&self, store: &Store, variable: VarId, value: i32) -> (i64, i64) {
        self.bounds_for(store, Some((variable, value)))
    }

    /// Find a small deterministic minimization hint over the current domains.
    ///
    /// The result assigns only variables referenced by this guarded sum. Every
    /// chosen value is supported by `store`, and the returned objective is the
    /// exact guarded-sum value of that assignment. Other model constraints are
    /// deliberately ignored, so this is a phase hint and never a feasibility or
    /// optimality claim.
    ///
    /// One unit of `work_limit` is one evaluation of a stored guarded literal
    /// against a tentative value. Initial-state construction, move scoring, and
    /// perturbations all debit the same budget. Candidate-value extraction is a
    /// single bounded preprocessing pass and is not charged.
    pub(crate) fn minimize_hint(&self, store: &Store, seed: u64, stop: &AtomicBool, work_limit: usize) -> Option<(i64, Vec<(VarId, i32)>)> {
        if work_limit == 0 || stop.load(Ordering::Acquire) {
            return None;
        }
        if self.variables.is_empty() {
            return Some((self.constant, Vec::new()));
        }

        let candidates = self.candidate_values(store, stop)?;
        let mut budget = HintBudget { remaining: work_limit, stop };
        let mut rng = HintRng::new(seed);
        let mut best: Option<(i64, Vec<i32>)> = None;
        let mut restart = 0usize;

        while budget.available() && restart < MAX_HINT_STARTS {
            let values = initial_values(&candidates, restart, &mut rng);
            let Some(mut state) = self.initialize_hint_state(values, &mut budget) else {
                break;
            };
            retain_best(&mut best, &state);
            if state.objective == self.constant {
                break;
            }

            let mut order = (0..self.variables.len()).collect::<Vec<_>>();
            rng.shuffle(&mut order);
            // First settle cheap local auxiliaries, then reconsider variables
            // that couple many terms. The initial shuffle preserves seeded
            // diversity between variables with the same incidence count.
            order.sort_by_key(|&variable_index| self.variable_incidences(variable_index).len());
            let max_kicks = if self.variables.len() <= SMALL_HINT_VARIABLES { MAX_SMALL_HINT_KICKS } else { 0 };
            let mut kicks = 0usize;
            loop {
                let completed = self.descend(&mut state, &candidates, &order, &mut rng, &mut budget, &mut best);
                if !completed || state.objective == self.constant {
                    break;
                }
                if kicks == max_kicks || !self.perturb(&mut state, &candidates, &order, &mut rng, &mut budget) {
                    break;
                }
                retain_best(&mut best, &state);
                kicks += 1;
            }
            if state.objective == self.constant || !budget.available() {
                break;
            }
            restart = restart.saturating_add(1);
        }

        best.map(|(objective, values)| {
            let assignment = self.variables.iter().copied().zip(values).collect();
            (objective, assignment)
        })
    }

    fn bounds_for(&self, store: &Store, forced: Option<(VarId, i32)>) -> (i64, i64) {
        let mut lower = self.constant;
        let mut upper = self.constant;

        for offsets in self.offsets.windows(2) {
            let start = offsets[0] as usize;
            let end = offsets[1] as usize;
            match conjunction_truth(&self.literals[start..end], store, forced) {
                Truth::False => {}
                Truth::Unknown => upper = upper.saturating_add(1),
                Truth::True => {
                    lower = lower.saturating_add(1);
                    upper = upper.saturating_add(1);
                }
            }
        }

        (lower, upper)
    }

    fn candidate_values(&self, store: &Store, stop: &AtomicBool) -> Option<Vec<Vec<i32>>> {
        let mut all = Vec::with_capacity(self.variables.len());
        for (variable_index, &variable) in self.variables.iter().enumerate() {
            if stop.load(Ordering::Acquire) || store.size(variable) == 0 {
                return None;
            }
            let mut values = Vec::new();
            for incidence in self.variable_incidences(variable_index) {
                if stop.load(Ordering::Acquire) {
                    return None;
                }
                for literal in &self.literals[incidence.start as usize..incidence.end as usize] {
                    if stop.load(Ordering::Acquire) {
                        return None;
                    }
                    if store.contains(variable, literal.value) {
                        values.push(literal.value);
                    }
                }
            }
            values.sort_unstable();
            values.dedup();

            // All values absent from the literals are objective-equivalent. One
            // supported representative is therefore sufficient.
            if values.len() < store.size(variable) {
                for value in store.values(variable) {
                    if stop.load(Ordering::Acquire) {
                        return None;
                    }
                    if values.binary_search(&value).is_err() {
                        values.push(value);
                        break;
                    }
                }
                values.sort_unstable();
            }
            debug_assert!(!values.is_empty());
            all.push(values);
        }
        Some(all)
    }

    fn variable_incidences(&self, variable_index: usize) -> &[Incidence] {
        let start = self.incidence_offsets[variable_index] as usize;
        let end = self.incidence_offsets[variable_index + 1] as usize;
        &self.incidences[start..end]
    }

    fn initialize_hint_state(&self, values: Vec<i32>, budget: &mut HintBudget<'_>) -> Option<HintState> {
        let mut group_true = vec![false; self.incidences.len()];
        let mut false_groups = vec![0u32; self.offsets.len().saturating_sub(1)];
        for (variable_index, &value) in values.iter().enumerate() {
            let start = self.incidence_offsets[variable_index] as usize;
            let end = self.incidence_offsets[variable_index + 1] as usize;
            for (relative_index, truth) in group_true[start..end].iter_mut().enumerate() {
                let incidence_index = start + relative_index;
                let incidence = self.incidences[incidence_index];
                let is_true = self.group_matches(incidence, value, budget)?;
                *truth = is_true;
                if !is_true {
                    false_groups[incidence.term as usize] = false_groups[incidence.term as usize].saturating_add(1);
                }
            }
        }
        let satisfied = false_groups.iter().filter(|&&count| count == 0).count();
        let objective = self.constant.saturating_add(i64::try_from(satisfied).unwrap_or(i64::MAX));
        Some(HintState { values, group_true, false_groups, objective })
    }

    fn descend(
        &self,
        state: &mut HintState,
        candidates: &[Vec<i32>],
        order: &[usize],
        rng: &mut HintRng,
        budget: &mut HintBudget<'_>,
        best: &mut Option<(i64, Vec<i32>)>,
    ) -> bool {
        loop {
            let mut improved = false;
            let offset = rng.index(order.len());
            for order_index in 0..order.len() {
                let variable_index = order[(offset + order_index) % order.len()];
                let choices = &candidates[variable_index];
                if choices.len() < 2 {
                    continue;
                }

                let choice_offset = rng.index(choices.len());
                let mut selected: Option<(i32, i64)> = None;
                for choice_index in 0..choices.len() {
                    let value = choices[(choice_offset + choice_index) % choices.len()];
                    if value == state.values[variable_index] {
                        continue;
                    }
                    let Some(delta) = self.evaluate_delta(state, variable_index, value, budget) else {
                        return false;
                    };
                    if delta < 0 && selected.as_ref().is_none_or(|(_, selected_delta)| delta < *selected_delta) {
                        selected = Some((value, delta));
                    }
                }
                if let Some((value, expected_delta)) = selected {
                    let Some(evaluation) = self.evaluate_move(state, variable_index, value, budget) else {
                        return false;
                    };
                    debug_assert_eq!(evaluation.delta, expected_delta);
                    self.apply_move(state, variable_index, value, evaluation);
                    retain_best(best, state);
                    improved = true;
                    if state.objective == self.constant {
                        return true;
                    }
                }
            }
            if !improved {
                return true;
            }
        }
    }

    fn perturb(
        &self,
        state: &mut HintState,
        candidates: &[Vec<i32>],
        order: &[usize],
        rng: &mut HintRng,
        budget: &mut HintBudget<'_>,
    ) -> bool {
        let movable = order.iter().copied().filter(|&index| candidates[index].len() > 1).collect::<Vec<_>>();
        if movable.is_empty() {
            return false;
        }
        let first = rng.index(movable.len());
        let count = movable.len().min(2);
        for step in 0..count {
            let variable_index = movable[(first + step) % movable.len()];
            let choices = &candidates[variable_index];
            let current = choices.binary_search(&state.values[variable_index]).expect("hint value remains a supported candidate");
            let shift = 1 + rng.index(choices.len() - 1);
            let value = choices[(current + shift) % choices.len()];
            let Some(evaluation) = self.evaluate_move(state, variable_index, value, budget) else {
                return false;
            };
            self.apply_move(state, variable_index, value, evaluation);
        }
        true
    }

    fn evaluate_move(&self, state: &HintState, variable_index: usize, value: i32, budget: &mut HintBudget<'_>) -> Option<MoveEvaluation> {
        let start = self.incidence_offsets[variable_index] as usize;
        let end = self.incidence_offsets[variable_index + 1] as usize;
        let mut delta = 0i64;
        let mut group_true = Vec::with_capacity(end - start);
        for incidence_index in start..end {
            let incidence = self.incidences[incidence_index];
            let old = state.group_true[incidence_index];
            let new = self.group_matches(incidence, value, budget)?;
            group_true.push(new);
            if old && !new && state.false_groups[incidence.term as usize] == 0 {
                delta = delta.saturating_sub(1);
            } else if !old && new && state.false_groups[incidence.term as usize] == 1 {
                delta = delta.saturating_add(1);
            }
        }
        Some(MoveEvaluation { delta, group_true })
    }

    fn evaluate_delta(&self, state: &HintState, variable_index: usize, value: i32, budget: &mut HintBudget<'_>) -> Option<i64> {
        let start = self.incidence_offsets[variable_index] as usize;
        let end = self.incidence_offsets[variable_index + 1] as usize;
        let mut delta = 0i64;
        for incidence_index in start..end {
            let incidence = self.incidences[incidence_index];
            let old = state.group_true[incidence_index];
            let new = self.group_matches(incidence, value, budget)?;
            if old && !new && state.false_groups[incidence.term as usize] == 0 {
                delta = delta.saturating_sub(1);
            } else if !old && new && state.false_groups[incidence.term as usize] == 1 {
                delta = delta.saturating_add(1);
            }
        }
        Some(delta)
    }

    fn apply_move(&self, state: &mut HintState, variable_index: usize, value: i32, evaluation: MoveEvaluation) {
        let start = self.incidence_offsets[variable_index] as usize;
        let end = self.incidence_offsets[variable_index + 1] as usize;
        debug_assert_eq!(evaluation.group_true.len(), end - start);
        for (incidence_index, new) in (start..end).zip(evaluation.group_true) {
            let old = state.group_true[incidence_index];
            if old == new {
                continue;
            }
            let term = self.incidences[incidence_index].term as usize;
            if old {
                state.false_groups[term] = state.false_groups[term].saturating_add(1);
            } else {
                debug_assert!(state.false_groups[term] > 0);
                state.false_groups[term] -= 1;
            }
            state.group_true[incidence_index] = new;
        }
        state.values[variable_index] = value;
        state.objective = state.objective.saturating_add(evaluation.delta);
    }

    fn group_matches(&self, incidence: Incidence, value: i32, budget: &mut HintBudget<'_>) -> Option<bool> {
        for literal in &self.literals[incidence.start as usize..incidence.end as usize] {
            budget.consume_literal()?;
            if (value == literal.value) != literal.equality {
                return Some(false);
            }
        }
        Some(true)
    }
}

#[cfg_attr(not(test), allow(dead_code))]
impl HintBudget<'_> {
    fn available(&self) -> bool {
        self.remaining > 0 && !self.stop.load(Ordering::Acquire)
    }

    fn consume_literal(&mut self) -> Option<()> {
        if !self.available() {
            return None;
        }
        self.remaining -= 1;
        Some(())
    }
}

#[cfg_attr(not(test), allow(dead_code))]
impl HintRng {
    fn new(seed: u64) -> Self {
        Self { state: seed ^ 0x9e37_79b9_7f4a_7c15 }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.state)
    }

    fn index(&mut self, len: usize) -> usize {
        debug_assert!(len > 0);
        (self.next() % len as u64) as usize
    }

    fn shuffle(&mut self, values: &mut [usize]) {
        for end in (1..values.len()).rev() {
            let selected = self.index(end + 1);
            values.swap(selected, end);
        }
    }
}

fn build_incidences(literals: &[Literal], offsets: &[u32], variables: &[VarId]) -> Option<IncidenceIndex> {
    let mut by_variable = vec![Vec::new(); variables.len()];
    for (term, term_offsets) in offsets.windows(2).enumerate() {
        let term = u32::try_from(term).ok()?;
        let mut start = term_offsets[0] as usize;
        let term_end = term_offsets[1] as usize;
        while start < term_end {
            let variable = literals[start].variable;
            let end = literals[start..term_end]
                .iter()
                .position(|literal| literal.variable != variable)
                .map_or(term_end, |relative| start + relative);
            let variable_index = variables.binary_search(&variable).ok()?;
            by_variable[variable_index].push(Incidence { term, start: u32::try_from(start).ok()?, end: u32::try_from(end).ok()? });
            start = end;
        }
    }

    let mut incidence_offsets = Vec::with_capacity(variables.len() + 1);
    let mut incidences = Vec::new();
    incidence_offsets.push(0u32);
    for variable_incidences in by_variable {
        incidences.extend(variable_incidences);
        incidence_offsets.push(u32::try_from(incidences.len()).ok()?);
    }
    Some((incidence_offsets.into_boxed_slice(), incidences.into_boxed_slice()))
}

#[cfg_attr(not(test), allow(dead_code))]
fn initial_values(candidates: &[Vec<i32>], restart: usize, rng: &mut HintRng) -> Vec<i32> {
    candidates.iter().map(|values| if restart == 0 { values[0] } else { values[rng.index(values.len())] }).collect()
}

#[cfg_attr(not(test), allow(dead_code))]
fn retain_best(best: &mut Option<(i64, Vec<i32>)>, state: &HintState) {
    if best.as_ref().is_none_or(|(objective, _)| state.objective < *objective) {
        *best = Some((state.objective, state.values.clone()));
    }
}

fn compile_term(atoms: &[Expr]) -> Option<ParsedTerm> {
    let mut literals = Vec::with_capacity(atoms.len());
    for atom in atoms {
        match parse_literal(atom)? {
            ParsedLiteral::Constant(false) => return Some(ParsedTerm::False),
            ParsedLiteral::Constant(true) => {}
            ParsedLiteral::Predicate(literal) => literals.push(literal),
        }
    }

    literals.sort_unstable();
    literals.dedup();
    let mut normalized = Vec::with_capacity(literals.len());
    let mut index = 0;
    while index < literals.len() {
        let variable = literals[index].variable;
        let end =
            literals[index..].iter().position(|literal| literal.variable != variable).map_or(literals.len(), |relative| index + relative);
        let group = &literals[index..end];

        if let Some(required) = group.iter().find(|literal| literal.equality) {
            if group.iter().any(|literal| {
                (literal.equality && literal.value != required.value) || (!literal.equality && literal.value == required.value)
            }) {
                return Some(ParsedTerm::False);
            }
            // Once x == c is required, every x != d with d != c is redundant.
            normalized.push(*required);
        } else {
            normalized.extend_from_slice(group);
        }
        index = end;
    }

    if normalized.is_empty() {
        Some(ParsedTerm::True)
    } else {
        Some(ParsedTerm::Literals(normalized))
    }
}

fn parse_literal(expression: &Expr) -> Option<ParsedLiteral> {
    match expression {
        Expr::Const(value) => Some(ParsedLiteral::Constant(*value != 0)),
        Expr::Var(variable) => Some(ParsedLiteral::Predicate(Literal { variable: *variable, value: 0, equality: false })),
        Expr::Not(value) => match value.as_ref() {
            Expr::Var(variable) => Some(ParsedLiteral::Predicate(Literal { variable: *variable, value: 0, equality: true })),
            Expr::Const(value) => Some(ParsedLiteral::Constant(*value == 0)),
            _ => None,
        },
        Expr::Eq(left, right) => parse_comparison(left, right, true),
        Expr::Ne(left, right) => parse_comparison(left, right, false),
        _ => None,
    }
}

fn parse_comparison(left: &Expr, right: &Expr, equality: bool) -> Option<ParsedLiteral> {
    match (left, right) {
        (Expr::Var(variable), Expr::Const(value)) | (Expr::Const(value), Expr::Var(variable)) => {
            let Ok(value) = i32::try_from(*value) else {
                // Store variables are i32-valued, so an out-of-range constant
                // can never be equal to one.
                return Some(ParsedLiteral::Constant(!equality));
            };
            Some(ParsedLiteral::Predicate(Literal { variable: *variable, value, equality }))
        }
        (Expr::Const(left), Expr::Const(right)) => Some(ParsedLiteral::Constant((*left == *right) == equality)),
        _ => None,
    }
}

fn conjunction_truth(literals: &[Literal], store: &Store, forced: Option<(VarId, i32)>) -> Truth {
    let mut result = Truth::True;
    for literal in literals {
        match literal_truth(*literal, store, forced) {
            Truth::False => return Truth::False,
            Truth::Unknown => result = Truth::Unknown,
            Truth::True => {}
        }
    }
    result
}

fn literal_truth(literal: Literal, store: &Store, forced: Option<(VarId, i32)>) -> Truth {
    if let Some((variable, value)) = forced.filter(|(variable, _)| *variable == literal.variable) {
        debug_assert_eq!(variable, literal.variable);
        return if (value == literal.value) == literal.equality { Truth::True } else { Truth::False };
    }

    let contains = store.contains(literal.variable, literal.value);
    if literal.equality {
        if !contains {
            Truth::False
        } else if store.size(literal.variable) == 1 {
            Truth::True
        } else {
            Truth::Unknown
        }
    } else if !contains {
        Truth::True
    } else if store.size(literal.variable) == 1 {
        Truth::False
    } else {
        Truth::Unknown
    }
}
