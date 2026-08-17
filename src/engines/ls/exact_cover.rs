//! Bounded constructive search for Boolean exact-cover models.
//!
//! The plan is compiled from the semantic IR and produces only a candidate.
//! The orchestrator remains responsible for canonical CP replay and transfer
//! verification before the candidate can become an incumbent.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::model::{Constraint, IntDomain, IntExpr, IntGlobalConstraint, IntVarRef, Model, Objective, Relation};

const MAX_EXACT_COVER_ROWS: usize = 4_096;

#[derive(Clone)]
pub(crate) struct ExactCoverPlan {
    rows: Vec<Vec<usize>>,
    columns: Vec<Vec<usize>>,
    binary: Vec<bool>,
    variable_count: usize,
    estimated_bytes: u64,
}

pub(crate) struct ExactCoverSolution {
    pub(crate) values: Vec<Option<i32>>,
    pub(crate) selected: usize,
    pub(crate) nodes: u64,
}

impl ExactCoverPlan {
    pub(crate) fn compile(model: &Model, stop: &AtomicBool, memory_allowance: u64) -> Option<Self> {
        let [objective] = model.objectives() else {
            return None;
        };
        if model.int_vars().is_empty() {
            return None;
        }
        let variable_count = model.int_vars().len();
        if estimate_plan_bytes(variable_count, 0, 0) > memory_allowance {
            return None;
        }
        let binary = model.int_vars().iter().map(is_binary_domain).collect::<Vec<_>>();
        let materialized_objective = match objective {
            Objective::IntExpr { expr: IntExpr::Variable(variable), .. } => match binary.get(variable.0).copied()? {
                false => Some(*variable),
                true if binary.iter().all(|&value| value) => None,
                true => return None,
            },
            Objective::IntExpr { .. } if binary.iter().all(|&value| value) => None,
            Objective::IntExpr { .. } | Objective::ListTerms { .. } | Objective::Makespan { .. } => return None,
        };
        if binary.iter().enumerate().any(|(variable, &is_binary)| !is_binary && Some(IntVarRef(variable)) != materialized_objective) {
            return None;
        }

        let mut rows = Vec::new();
        let mut memberships = 0usize;
        let mut objective_definition = false;
        for constraint in model.constraints() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            let Some(row_variables) = exact_one_variables(constraint) else {
                let objective = materialized_objective?;
                if objective_definition || !is_objective_definition(constraint, objective, &binary) {
                    return None;
                }
                objective_definition = true;
                continue;
            };
            let next_memberships = memberships.checked_add(row_variables.len())?;
            if rows.len() == MAX_EXACT_COVER_ROWS
                || estimate_plan_bytes(variable_count, rows.len() + 1, next_memberships) > memory_allowance
            {
                return None;
            }
            let mut row = row_variables.indices();
            row.sort_unstable();
            if row.is_empty()
                || row.windows(2).any(|pair| pair[0] == pair[1])
                || row.iter().any(|&variable| variable >= binary.len() || !binary[variable])
            {
                return None;
            }
            rows.push(row);
            memberships = next_memberships;
        }
        if rows.is_empty() || materialized_objective.is_some() != objective_definition {
            return None;
        }

        let estimated_bytes = estimate_plan_bytes(variable_count, rows.len(), memberships);
        let mut columns = vec![Vec::new(); variable_count];
        for (row, variables) in rows.iter().enumerate() {
            for &variable in variables {
                columns[variable].push(row);
            }
        }
        Some(Self { rows, columns, binary, variable_count, estimated_bytes })
    }

    pub(crate) fn estimated_bytes(&self) -> u64 {
        self.estimated_bytes
    }

    pub(crate) fn construct(&self, seed: u64, stop: &AtomicBool) -> Option<ExactCoverSolution> {
        let mut state = SearchState {
            plan: self,
            covered: vec![false; self.rows.len()],
            selected: vec![false; self.variable_count],
            covered_count: 0,
            nodes: 0,
            seed,
            stop,
        };
        let found = state.search();
        let selected = state.selected.iter().filter(|&&value| value).count();
        found.then(|| ExactCoverSolution {
            values: state.selected.into_iter().zip(&self.binary).map(|(value, &binary)| binary.then_some(i32::from(value))).collect(),
            selected,
            nodes: state.nodes,
        })
    }
}

fn is_binary_domain(domain: &IntDomain) -> bool {
    match domain {
        IntDomain::Bool | IntDomain::Range { lo: 0, hi: 1 } => true,
        IntDomain::Set(values) => values.len() == 2 && values.contains(&0) && values.contains(&1),
        IntDomain::Range { .. } => false,
    }
}

enum ExactOneVariables<'a> {
    Count(&'a [IntVarRef]),
    Linear(&'a [(i64, IntVarRef)]),
}

impl ExactOneVariables<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Count(variables) => variables.len(),
            Self::Linear(terms) => terms.len(),
        }
    }

    fn indices(&self) -> Vec<usize> {
        match self {
            Self::Count(variables) => variables.iter().map(|variable| variable.0).collect(),
            Self::Linear(terms) => terms.iter().map(|(_, variable)| variable.0).collect(),
        }
    }
}

fn exact_one_variables(constraint: &Constraint) -> Option<ExactOneVariables<'_>> {
    match constraint {
        Constraint::IntegerGlobal(IntGlobalConstraint::Count { variables, value: 1, relation: Relation::Eq, count: 1 }) => {
            Some(ExactOneVariables::Count(variables))
        }
        Constraint::Linear { terms, relation: Relation::Eq, rhs: 1 } if terms.iter().all(|(coefficient, _)| *coefficient == 1) => {
            Some(ExactOneVariables::Linear(terms))
        }
        _ => None,
    }
}

fn estimate_plan_bytes(variable_count: usize, row_count: usize, memberships: usize) -> u64 {
    let vector_headers = row_count.saturating_add(variable_count);
    let bytes = (vector_headers as u128)
        .saturating_mul(std::mem::size_of::<Vec<usize>>() as u128)
        .saturating_add((memberships as u128).saturating_mul(2).saturating_mul(std::mem::size_of::<usize>() as u128))
        .saturating_add(variable_count as u128)
        .saturating_add(4_096);
    u64::try_from(bytes).unwrap_or(u64::MAX)
}

fn is_objective_definition(constraint: &Constraint, objective: IntVarRef, binary: &[bool]) -> bool {
    let Constraint::Linear { terms, relation: Relation::Eq, .. } = constraint else {
        return false;
    };
    let mut objective_coefficient = None;
    for &(coefficient, variable) in terms {
        if variable == objective {
            if objective_coefficient.replace(coefficient).is_some() {
                return false;
            }
        } else if variable.0 >= binary.len() || !binary[variable.0] {
            return false;
        }
    }
    objective_coefficient.is_some_and(|coefficient| coefficient.unsigned_abs() == 1)
}

struct SearchState<'a> {
    plan: &'a ExactCoverPlan,
    covered: Vec<bool>,
    selected: Vec<bool>,
    covered_count: usize,
    nodes: u64,
    seed: u64,
    stop: &'a AtomicBool,
}

impl SearchState<'_> {
    fn search(&mut self) -> bool {
        self.nodes = self.nodes.saturating_add(1);
        if (self.nodes == 1 || self.nodes.is_multiple_of(256)) && self.stop.load(Ordering::Acquire) {
            return false;
        }
        if self.covered_count == self.covered.len() {
            return true;
        }

        let Some(mut candidates) = self.tightest_row_candidates() else {
            return false;
        };
        candidates.sort_unstable_by_key(|&column| {
            let gain = self.plan.columns[column].len();
            (usize::MAX - gain, crate::mix64(self.seed ^ self.nodes ^ column as u64))
        });

        for column in candidates {
            if self.stop.load(Ordering::Acquire) {
                return false;
            }
            let rows = &self.plan.columns[column];
            if rows.iter().any(|&row| self.covered[row]) {
                continue;
            }
            for &row in rows {
                self.covered[row] = true;
            }
            self.covered_count += rows.len();
            self.selected[column] = true;
            if self.search() {
                return true;
            }
            self.selected[column] = false;
            self.covered_count -= rows.len();
            for &row in rows {
                self.covered[row] = false;
            }
        }
        false
    }

    fn tightest_row_candidates(&self) -> Option<Vec<usize>> {
        let mut best: Option<Vec<usize>> = None;
        for (row, columns) in self.plan.rows.iter().enumerate() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            if self.covered[row] {
                continue;
            }
            let mut candidates = Vec::new();
            for (index, &column) in columns.iter().enumerate() {
                if index.is_multiple_of(1_024) && self.stop.load(Ordering::Acquire) {
                    return None;
                }
                if self.plan.columns[column].iter().all(|&member| !self.covered[member]) {
                    candidates.push(column);
                }
            }
            if candidates.is_empty() {
                return None;
            }
            if best.as_ref().is_none_or(|current| candidates.len() < current.len()) {
                best = Some(candidates);
                if best.as_ref().is_some_and(|current| current.len() == 1) {
                    break;
                }
            }
        }
        best
    }
}
