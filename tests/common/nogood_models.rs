//! Shared fixed models and independent witness checks for the I6 experiment.

use std::collections::BTreeSet;
use std::sync::Arc;

use qayd::constraints::linear::{linear, Relation as LinearRelation};
use qayd::constraints::table::extension;
use qayd::model::{Constraint, IntExpr, IntGlobalConstraint, Model, ModelPackage, Objective, Relation};
use qayd::{Solver, VarId};

#[derive(Clone, Copy, Debug)]
enum Family {
    Permutation(usize),
    Pigeonhole(usize),
}

pub struct ModelCase {
    pub package: ModelPackage,
    pub variable_count: usize,
    pub feasible: bool,
    /// Minimum of the first variable, independently known from the family.
    pub optimum: Option<i64>,
    pub objective: bool,
    family: Family,
    impossible: bool,
}

impl ModelCase {
    /// Check a complete semantic assignment without invoking a Qayd solver.
    pub fn accepts(&self, values: &[i32]) -> bool {
        if values.len() != self.variable_count {
            return false;
        }
        match self.family {
            Family::Permutation(n) => {
                let support = permutation_support(n);
                values.iter().all(|value| support.contains(value))
                    && values.iter().copied().collect::<BTreeSet<_>>().len() == n
                    && values.iter().map(|&value| i64::from(value)).sum::<i64>()
                        >= support.iter().map(|&value| i64::from(value)).sum::<i64>() + i64::from(self.impossible)
            }
            Family::Pigeonhole(holes) => {
                values.iter().all(|&value| value == 0 || value == 1)
                    && values.chunks_exact(holes).all(|row| row.iter().sum::<i32>() == 1)
                    && (0..holes).all(|hole| values.iter().skip(hole).step_by(holes).sum::<i32>() <= 1)
            }
        }
    }
}

pub fn permutation_support(n: usize) -> Vec<i32> {
    assert!((1..=6).contains(&n));
    (0..n as i32).map(|value| 2 * value - 3).collect()
}

/// Enumerate permutations directly, then evaluate the sum requirement. This
/// oracle does not reuse the positive-table or linear propagators.
pub fn permutation_solutions(n: usize, impossible: bool) -> Vec<Vec<i32>> {
    fn visit(values: &[i32], used: &mut [bool], prefix: &mut Vec<i32>, out: &mut Vec<Vec<i32>>) {
        if prefix.len() == values.len() {
            out.push(prefix.clone());
            return;
        }
        for index in 0..values.len() {
            if !used[index] {
                used[index] = true;
                prefix.push(values[index]);
                visit(values, used, prefix, out);
                prefix.pop();
                used[index] = false;
            }
        }
    }
    let values = permutation_support(n);
    let mut out = Vec::new();
    visit(&values, &mut vec![false; n], &mut Vec::new(), &mut out);
    let rhs = values.iter().map(|&value| i64::from(value)).sum::<i64>() + i64::from(impossible);
    out.retain(|assignment| assignment.iter().map(|&value| i64::from(value)).sum::<i64>() >= rhs);
    out
}

fn different_pairs(values: &[i32]) -> Vec<Vec<i32>> {
    values.iter().flat_map(|&left| values.iter().filter(move |&&right| left != right).map(move |&right| vec![left, right])).collect()
}

pub fn permutation_solver(n: usize, impossible: bool) -> (Solver, Vec<VarId>) {
    let mut solver = Solver::new();
    let support = permutation_support(n);
    let variables = (0..n).map(|_| solver.new_var_set(&support)).collect::<Vec<_>>();
    let tuples = different_pairs(&support);
    for left in 0..n {
        for right in left + 1..n {
            extension(&mut solver, &[variables[left], variables[right]], &tuples, true);
        }
    }
    let rhs = support.iter().map(|&value| i64::from(value)).sum::<i64>() + i64::from(impossible);
    linear(&mut solver, &vec![1; n], &variables, LinearRelation::Ge, rhs);
    (solver, variables)
}

pub fn permutation(n: usize, impossible: bool, objective: bool) -> ModelCase {
    let mut model = Model::new();
    let support = permutation_support(n);
    let variables = (0..n).map(|_| model.int_set(support.clone())).collect::<Vec<_>>();
    let tuples: Arc<[Vec<i32>]> = different_pairs(&support).into();
    for left in 0..n {
        for right in left + 1..n {
            model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table {
                variables: vec![variables[left], variables[right]],
                tuples: Arc::clone(&tuples),
                positive: true,
            }));
        }
    }
    model.add_constraint(Constraint::Linear {
        terms: variables.iter().map(|&variable| (1, variable)).collect(),
        relation: Relation::Ge,
        rhs: support.iter().map(|&value| i64::from(value)).sum::<i64>() + i64::from(impossible),
    });
    if objective {
        model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(variables[0]) });
    }
    ModelCase {
        package: ModelPackage::new(model),
        variable_count: n,
        feasible: !impossible,
        optimum: (!impossible).then_some(i64::from(support[0])),
        objective,
        family: Family::Permutation(n),
        impossible,
    }
}

pub fn pigeonhole(holes: usize, impossible: bool, objective: bool) -> ModelCase {
    assert!((2..=4).contains(&holes));
    let pigeons = holes + usize::from(impossible);
    let mut model = Model::new();
    let variables = (0..holes * pigeons).map(|_| model.int_range(0, 1)).collect::<Vec<_>>();
    for row in variables.chunks_exact(holes) {
        model.add_constraint(Constraint::Linear {
            terms: row.iter().map(|&variable| (1, variable)).collect(),
            relation: Relation::Eq,
            rhs: 1,
        });
    }
    for hole in 0..holes {
        model.add_constraint(Constraint::Linear {
            terms: (0..pigeons).map(|pigeon| (1, variables[pigeon * holes + hole])).collect(),
            relation: Relation::Le,
            rhs: 1,
        });
    }
    if objective {
        model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(variables[0]) });
    }
    ModelCase {
        package: ModelPackage::new(model),
        variable_count: variables.len(),
        feasible: !impossible,
        optimum: (!impossible).then_some(0),
        objective,
        family: Family::Pigeonhole(holes),
        impossible,
    }
}
