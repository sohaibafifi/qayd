//! Frozen I7 pilot models with exhaustive oracles independent of Qayd search.

use std::sync::Arc;

use qayd::model::{Constraint, IntExpr, IntGlobalConstraint, Model, ModelPackage, Objective, Relation};
use qayd::orchestrator::{SolveResult, SolveStatus, VerificationLevel};

#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub family: &'static str,
    pub size: usize,
}

pub fn all_cases() -> Vec<Case> {
    [("parity", [4, 6, 8]), ("odd_cycle", [3, 5, 7]), ("table_path", [4, 6, 8]), ("linear_min", [4, 6, 8]), ("linear_max", [4, 6, 8])]
        .into_iter()
        .flat_map(|(family, sizes)| sizes.into_iter().map(move |size| Case { family, size }))
        .collect()
}

impl Case {
    pub fn package(self) -> ModelPackage {
        let mut model = Model::new();
        let variables = (0..self.size).map(|_| model.int_range(0, 1)).collect::<Vec<_>>();
        match self.family {
            "parity" => {
                model.add_constraint(Constraint::Linear {
                    terms: variables.iter().map(|&variable| (2, variable)).collect(),
                    relation: Relation::Eq,
                    rhs: self.size as i64 + 1,
                });
            }
            "odd_cycle" | "table_path" => {
                let edges = if self.family == "odd_cycle" { self.size } else { self.size - 1 };
                let tuples: Arc<[Vec<i32>]> = vec![vec![0, 1], vec![1, 0]].into();
                for edge in 0..edges {
                    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table {
                        variables: vec![variables[edge], variables[(edge + 1) % self.size]],
                        tuples: Arc::clone(&tuples),
                        positive: true,
                    }));
                }
            }
            "linear_min" | "linear_max" => {
                let minimizing = self.family == "linear_min";
                let objective = model.int_range(0, (self.size * (self.size + 1) / 2) as i32);
                let mut terms = variables.iter().enumerate().map(|(index, &variable)| (index as i64 + 1, variable)).collect::<Vec<_>>();
                terms.push((-1, objective));
                model.add_constraint(Constraint::Linear { terms, relation: Relation::Eq, rhs: 0 });
                model.add_constraint(Constraint::Linear {
                    terms: variables.iter().map(|&variable| (1, variable)).collect(),
                    relation: if minimizing { Relation::Ge } else { Relation::Le },
                    rhs: if minimizing { self.size.div_ceil(2) as i64 } else { (self.size / 2) as i64 },
                });
                model.add_objective(Objective::IntExpr { minimize: minimizing, expr: IntExpr::Variable(objective) });
            }
            family => panic!("unknown certificate pilot family {family}"),
        }
        ModelPackage::new(model)
    }

    /// Evaluate the declared family directly, without using model replay or
    /// any solver propagator. The final coordinate is z for optimization.
    pub fn accepts(self, values: &[i64]) -> bool {
        let optimization = matches!(self.family, "linear_min" | "linear_max");
        if values.len() != self.size + usize::from(optimization) || values[..self.size].iter().any(|&value| value != 0 && value != 1) {
            return false;
        }
        let count = values[..self.size].iter().sum::<i64>();
        match self.family {
            "parity" => 2 * count == self.size as i64 + 1,
            "odd_cycle" => (0..self.size).all(|index| values[index] != values[(index + 1) % self.size]),
            "table_path" => values.windows(2).all(|pair| pair[0] != pair[1]),
            "linear_min" | "linear_max" => {
                let weighted = values[..self.size].iter().enumerate().map(|(index, &value)| (index as i64 + 1) * value).sum::<i64>();
                values[self.size] == weighted
                    && if self.family == "linear_min" { count >= self.size.div_ceil(2) as i64 } else { count <= (self.size / 2) as i64 }
            }
            _ => false,
        }
    }

    /// Exhaust the Boolean cube, computing z directly instead of enumerating
    /// its domain. The largest pilot oracle visits only 256 assignments.
    pub fn solutions(self) -> Vec<Vec<i64>> {
        assert!((1..=8).contains(&self.size));
        (0..1usize << self.size)
            .map(|bits| {
                let mut values = (0..self.size).map(|index| ((bits >> index) & 1) as i64).collect::<Vec<_>>();
                if matches!(self.family, "linear_min" | "linear_max") {
                    values.push(values.iter().enumerate().map(|(index, &value)| (index as i64 + 1) * value).sum());
                }
                values
            })
            .filter(|values| self.accepts(values))
            .collect()
    }

    pub fn expected_objective(self) -> Option<i64> {
        match self.family {
            "linear_min" => self.solutions().iter().map(|values| values[self.size]).min(),
            "linear_max" => self.solutions().iter().map(|values| values[self.size]).max(),
            _ => None,
        }
    }

    pub fn expected_status(self) -> SolveStatus {
        if self.solutions().is_empty() {
            SolveStatus::Unsatisfiable
        } else if self.expected_objective().is_some() {
            SolveStatus::Optimal
        } else {
            SolveStatus::Satisfiable
        }
    }

    /// Call after timing. This validates a full witness and its optimum with
    /// the independent exhaustive family oracle, in addition to the contract.
    pub fn validate(self, result: &SolveResult) -> Result<(), String> {
        result.validate_contract().map_err(|error| format!("{self:?}: {error}"))?;
        if result.status() != self.expected_status() {
            return Err(format!("{self:?}: expected {:?}, got {:?}", self.expected_status(), result.status()));
        }
        if result.status() == SolveStatus::Unsatisfiable {
            if result.primal().is_some() {
                return Err(format!("{self:?}: UNSAT has a witness"));
            }
            return Ok(());
        }
        let candidate = result.primal().ok_or_else(|| format!("{self:?}: missing witness"))?;
        let values = candidate
            .assignment()
            .integers
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format!("{self:?}: partial witness"))?;
        if candidate.verification() != VerificationLevel::Final || !self.accepts(&values) {
            return Err(format!("{self:?}: invalid witness {values:?}"));
        }
        let objective = self.expected_objective().into_iter().collect::<Vec<_>>();
        if candidate.objectives() != objective {
            return Err(format!("{self:?}: expected objective {objective:?}, got {:?}", candidate.objectives()));
        }
        Ok(())
    }
}
