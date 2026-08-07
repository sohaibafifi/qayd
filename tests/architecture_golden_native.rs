//! Native Rust fixture used by the architecture golden harness.

use std::collections::HashMap;

use qayd::model::{Constraint, IntExpr, IntVarRef, Model, ModelPackage, Objective, Relation};
use qayd::orchestrator::{IgnoreEvents, SolveMode, SolveRequest, SolveStatus};

const SPEC: &str = include_str!("../bench/golden/fixtures/rust_native.case");

#[test]
fn emits_native_golden_record() {
    let mut model = Model::new();
    let mut variables: Vec<(String, IntVarRef)> = Vec::new();
    let mut by_name = HashMap::new();
    let mut objective = None;

    for (line_number, raw) in SPEC.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split_whitespace().collect::<Vec<_>>();
        match fields.as_slice() {
            ["var", name, lo, hi] => {
                let lo = lo.parse::<i32>().unwrap_or_else(|_| panic!("line {}: bad lower bound", line_number + 1));
                let hi = hi.parse::<i32>().unwrap_or_else(|_| panic!("line {}: bad upper bound", line_number + 1));
                let variable = model.int_range(lo, hi);
                assert!(by_name.insert((*name).to_string(), variable).is_none(), "duplicate variable {name}");
                variables.push(((*name).to_string(), variable));
            }
            ["linear", relation, rhs, terms @ ..] if !terms.is_empty() => {
                let relation = match *relation {
                    "eq" => Relation::Eq,
                    "le" => Relation::Le,
                    "ge" => Relation::Ge,
                    "lt" => Relation::Lt,
                    "gt" => Relation::Gt,
                    other => panic!("line {}: bad relation {other}", line_number + 1),
                };
                let rhs = rhs.parse::<i64>().unwrap_or_else(|_| panic!("line {}: bad right-hand side", line_number + 1));
                let mut linear_terms = Vec::with_capacity(terms.len());
                for term in terms {
                    let (coefficient, name) = term.split_once('*').unwrap_or_else(|| panic!("line {}: bad linear term", line_number + 1));
                    let coefficient = coefficient.parse::<i64>().unwrap_or_else(|_| panic!("line {}: bad coefficient", line_number + 1));
                    let variable = *by_name.get(name).unwrap_or_else(|| panic!("line {}: unknown variable {name}", line_number + 1));
                    linear_terms.push((coefficient, variable));
                }
                model.add_constraint(Constraint::Linear { terms: linear_terms, relation, rhs });
            }
            ["minimize", name] => {
                objective = Some(*by_name.get(*name).unwrap_or_else(|| panic!("line {}: unknown objective {name}", line_number + 1)));
            }
            _ => panic!("line {}: unsupported fixture statement", line_number + 1),
        }
    }

    let objective = objective.expect("fixture must declare an objective");
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(objective) });
    let mut events = IgnoreEvents;
    let request = SolveRequest { mode: SolveMode::Exact, seed: 0, threads: 1, ..SolveRequest::default() };
    let result = qayd::solve(&ModelPackage::new(model), &request, &mut events).expect("fixture solve must succeed");
    assert_eq!(result.status(), SolveStatus::Optimal);
    let candidate = result.primal().expect("fixture must be feasible");
    let value = candidate.objectives()[0];
    let canonical = variables
        .iter()
        .map(|(name, variable)| {
            let value = candidate.assignment().integers[variable.0].expect("fixture variable must be assigned");
            format!("\"{name}\":{value}")
        })
        .collect::<Vec<_>>()
        .join(",");

    assert_eq!(value, 1);
    assert_eq!(canonical, "\"x\":1,\"y\":2");
    println!(
        "QAYD_GOLDEN_RESULT={{\"status\":\"OPTIMAL\",\"senses\":[\"minimize\"],\"objectives\":[{value}],\"solution\":{{\"assignments\":{{{canonical}}}}},\"bound\":{{\"values\":[{value}],\"source\":\"complete-search\"}},\"proof\":{{\"claim\":\"optimality\",\"kind\":\"complete-search\",\"verified\":false}}}}"
    );
}
