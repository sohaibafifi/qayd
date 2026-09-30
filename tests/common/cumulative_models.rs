//! Frozen I5 search models and independent integer-slot occupancy oracle.

use std::collections::{BTreeMap, BTreeSet};

use qayd::constraints::interval;
use qayd::model::{Constraint, IntGlobalConstraint, Model, ModelPackage};
use qayd::{Solver, VarId};

#[derive(Clone, Debug)]
pub struct Case {
    pub family: String,
    pub n: usize,
    pub capacity: i32,
    pub domains: Vec<Vec<i32>>,
    pub durations: Vec<i32>,
    pub demands: Vec<i32>,
}

impl Case {
    pub fn new(family: &str, n: usize, capacity: i32) -> Self {
        assert!(matches!(family, "dense" | "anchored"));
        assert!((4..=6).contains(&n));
        assert!(matches!(capacity, 2 | 3));
        let domains = (0..n)
            .map(|index| match (family, index) {
                ("dense", _) => vec![0, 1, 2, 3, 4],
                (_, 0) => vec![0, 1],
                (_, 1) => vec![-1, 0],
                _ => vec![-2, 0, 1, 3, 4],
            })
            .collect();
        let mut durations = [2, 3, 1, 2, 1, 3][..n].to_vec();
        if family == "anchored" {
            durations[0] = 3;
        }
        Self { family: family.to_owned(), n, capacity, domains, durations, demands: [1, 1, 2, 1, 2, 1][..n].to_vec() }
    }

    /// Every occurrence contributes its demand independently at each occupied
    /// integer slot. No production profile or endpoint algorithm is reused.
    pub fn accepts(&self, values: &[i32]) -> bool {
        if values.len() != self.n || values.iter().zip(&self.domains).any(|(value, domain)| !domain.contains(value)) {
            return false;
        }
        let mut loads = BTreeMap::<i64, i64>::new();
        for (index, &start) in values.iter().enumerate() {
            for offset in 0..self.durations[index] {
                let load = loads.entry(i64::from(start) + i64::from(offset)).or_default();
                *load += i64::from(self.demands[index]);
                if *load > i64::from(self.capacity) {
                    return false;
                }
            }
        }
        true
    }

    pub fn solutions(&self) -> BTreeSet<Vec<i32>> {
        fn visit(case: &Case, values: &mut Vec<i32>, result: &mut BTreeSet<Vec<i32>>) {
            if values.len() == case.n {
                if case.accepts(values) {
                    result.insert(values.clone());
                }
                return;
            }
            for &value in &case.domains[values.len()] {
                values.push(value);
                visit(case, values, result);
                values.pop();
            }
        }
        let mut result = BTreeSet::new();
        visit(self, &mut Vec::new(), &mut result);
        result
    }

    pub fn solver(&self, incremental: bool, lifted: bool, metrics: bool) -> (Solver, Vec<VarId>) {
        let mut solver = Solver::new();
        solver.set_incremental_cumulative(incremental);
        solver.set_lifted_cumulative_explanations(lifted);
        solver.set_cumulative_metrics(metrics);
        let mut intervals = Vec::new();
        let mut variables = Vec::new();
        for (domain, &duration) in self.domains.iter().zip(&self.durations) {
            let lower = *domain.iter().min().unwrap();
            let upper = *domain.iter().max().unwrap();
            let interval = solver.store.new_interval(lower, upper, duration);
            let start = solver.store.interval_start_var(interval);
            for value in lower..=upper {
                if !domain.contains(&value) {
                    solver.store.remove(start, value).unwrap();
                }
            }
            intervals.push(interval);
            variables.push(start);
        }
        interval::cumulative(&mut solver, &intervals, &self.demands, self.capacity);
        (solver, variables)
    }

    pub fn model(&self) -> ModelPackage {
        let mut model = Model::new();
        let starts = self.domains.iter().map(|domain| model.int_set(domain.clone())).collect();
        model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Cumulative {
            starts,
            durations: self.durations.iter().map(|&value| i64::from(value)).collect(),
            demands: self.demands.iter().map(|&value| i64::from(value)).collect(),
            capacity: i64::from(self.capacity),
        }));
        ModelPackage::new(model)
    }
}
