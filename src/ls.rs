//! Incumbent-only local search for `--fast-cop` runs.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::constraints::linear::Relation;
use crate::constraints::table::STAR;
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

#[derive(Clone)]
pub(crate) enum LocalConstraint {
    Expr(Expr),
    Linear { coeffs: Vec<i64>, vars: Vec<VarId>, rel: Relation, rhs: i64 },
    AllDifferent(Vec<VarId>),
    AllEqual(Vec<VarId>),
    Extension { vars: Vec<VarId>, tuples: Vec<Vec<i32>> },
    Lex { rows: Vec<Vec<VarId>>, strict: bool },
    Count { vars: Vec<VarId>, values: Vec<i32>, rel: Relation, rhs: LocalRhs },
    CountAllowed { vars: Vec<VarId>, values: Vec<i32>, allowed: Vec<i32> },
    NValues { vars: Vec<VarId>, rel: Relation, rhs: LocalRhs },
    Cardinality { vars: Vec<VarId>, values: Vec<i32>, low: Vec<i64>, high: Vec<i64>, closed: bool },
    Extremum { vars: Vec<VarId>, is_min: bool, rel: Relation, rhs: LocalRhs },
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

pub(crate) struct LocalSearchOutcome {
    pub(crate) best: Option<(Vec<i32>, i64)>,
    pub(crate) iterations: u64,
    pub(crate) moves: u64,
    pub(crate) restarts: u64,
    pub(crate) constraints: usize,
    pub(crate) functionals: usize,
    pub(crate) unsupported: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Score {
    violation: i64,
    objective: i64,
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
            self.mark_unsupported();
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

    pub(crate) fn add_element(&mut self, array: Vec<VarId>, index: VarId, target: VarId, start_index: i32) {
        self.mark_functional(Functional::Element { target, array, index, start_index });
    }

    pub(crate) fn mark_unsupported(&mut self) {
        self.unsupported += 1;
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
        let functionals = spec.functionals;
        let bool_tables = bool_tables(&functionals);
        let exact_covers = exact_cover_rows(&constraints, &domains);
        Ok(Self { domains, mutable, search: problem.search, objective, constraints, functionals, bool_tables, exact_covers })
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

    fn score(&self, assignment: &mut [i32]) -> Score {
        let mut violation = i64::from(!self.complete(assignment)) * 1_000_000;
        for constraint in &self.constraints {
            violation = violation.saturating_add(self.violation(constraint, assignment));
        }
        let objective = self.objective_value(assignment).unwrap_or(i64::MAX / 4);
        Score { violation, objective: if self.objective.minimizing() { objective } else { -objective } }
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

fn objective_kick(model: &LocalModel, assignment: &[i32], kicks: &[(VarId, i32)], seed: u64, iter: u64) -> Option<Vec<i32>> {
    if !iter.is_multiple_of(RANDOM_WALK_PERIOD) || kicks.is_empty() {
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

pub(crate) fn solve_fast_cop<F>(
    problem: Problem,
    spec: LocalSearchSpec,
    stop: &AtomicBool,
    seed: u64,
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
    if unsupported > 0 || model.mutable.is_empty() {
        return LocalSearchOutcome { best: None, iterations: 0, moves: 0, restarts: 0, constraints, functionals, unsupported };
    }

    let minimizing = model.objective.minimizing();
    let objective_kicks = model.objective_kicks();
    let constructive_start = model.has_extension();
    let mut assignment = if constructive_start { model.constructive_assignment(seed) } else { model.random_assignment(seed) };
    let mut current = model.score(&mut assignment);
    let mut source = if constructive_start { "constructive" } else { "local-search" };
    let mut best_solution: Option<(Vec<i32>, i64)> = None;
    let mut iterations = 0;
    let mut moves = 0;
    let mut restarts = 0;
    let mut stagnant = 0;

    while !stop.load(Ordering::Relaxed) {
        iterations += 1;
        if current.violation == 0 {
            if let Some(value) = model.objective_value(&assignment) {
                if better_value(minimizing, value, best_solution.as_ref().map(|(_, v)| *v)) {
                    let solution = model.search_solution(&assignment);
                    on_improve(value, &solution, source);
                    best_solution = Some((solution, value));
                }
            }
        }

        if stagnant >= RESTART_AFTER {
            restarts += 1;
            let restart_seed = seed ^ mix64(iterations);
            assignment =
                if constructive_start { model.constructive_assignment(restart_seed) } else { model.random_assignment(restart_seed) };
            current = model.score(&mut assignment);
            source = if constructive_start { "constructive" } else { "local-search" };
            stagnant = 0;
            continue;
        }

        if constructive_start && iterations.is_multiple_of(CONSTRUCTIVE_KICK_PERIOD) {
            let mut trial = model.constructive_assignment(seed ^ mix64(iterations));
            let score = model.score(&mut trial);
            if score < current {
                assignment = trial;
                current = score;
                source = "constructive";
                moves += 1;
                stagnant = 0;
                continue;
            }
        }

        if let Some(trial) = objective_kick(&model, &assignment, &objective_kicks, seed, iterations) {
            assignment = trial;
            current = model.score(&mut assignment);
            source = "local-search";
            moves += 1;
            stagnant = 0;
            continue;
        }

        let focused = model.focused_repair_vars(&assignment, seed, iterations);
        let candidates = if focused.is_empty() { candidate_vars(&model.mutable, seed, iterations) } else { focused };
        let mut best_move: Option<(Score, Vec<i32>)> = None;
        for var in candidates {
            let domain = &model.domains[var.index()].values;
            for &value in domain {
                if value == assignment[var.index()] {
                    continue;
                }
                let mut trial = assignment.clone();
                trial[var.index()] = value;
                let score = model.score(&mut trial);
                if best_move.as_ref().is_none_or(|(best, _)| score < *best) {
                    best_move = Some((score, trial));
                }
            }
        }

        let random_walk = iterations.is_multiple_of(RANDOM_WALK_PERIOD);
        if let Some((score, trial)) = best_move {
            if score < current || random_walk {
                assignment = trial;
                current = score;
                source = "local-search";
                moves += 1;
                stagnant = 0;
            } else {
                stagnant += 1;
            }
        } else {
            stagnant += 1;
        }
    }

    LocalSearchOutcome { best: best_solution, iterations, moves, restarts, constraints, functionals, unsupported }
}
