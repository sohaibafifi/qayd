//! Fixed I4 root and canonical search pilot. Run the ignored test alone.
use std::time::Instant;

use qayd::constraints::scheduling::{cumulative, no_overlap};
use qayd::constraints::unary_edge_finding::UnaryEdgeFindingStats;
use qayd::model::{Constraint, IntGlobalConstraint, Model, ModelPackage};
use qayd::orchestrator::{
    solve_model_silent, CpControls, ProvenConclusion, SearchPhase, SearchPolicy, SolveMode, SolveRequest, SolveStatus, ValueSelector,
    VariableSelector, VerificationLevel,
};
use qayd::store::{prop_calls, reset_prop_calls};
use qayd::{Solver, VarId};
use std::collections::BTreeMap;

const ARMS: [&str; 3] = ["native", "edge", "cumulative"];
const FAMILIES: [&str; 4] = ["prefix", "mirror", "slack", "overload"];

#[derive(Clone)]
struct Case {
    family: &'static str,
    k: usize,
    bounds: Vec<(i32, i32)>,
    witness: Option<Vec<i32>>,
}

impl Case {
    fn new(family: &'static str, k: usize) -> Self {
        let tight = 2 * k as i32 - 2;
        let wide = 2 * k as i32 + 4;
        let mut bounds = vec![(0, if family == "slack" { wide } else { tight }); k];
        bounds.push((0, if family == "overload" { tight } else { wide }));
        let mut witness = (0..=k).map(|i| 2 * i as i32).collect::<Vec<_>>();
        if family == "mirror" {
            bounds = bounds.into_iter().map(|(lo, hi)| (-hi - 2, -lo - 2)).collect();
            witness.iter_mut().for_each(|value| *value = -*value - 2);
        }
        Self { family, k, bounds, witness: (family != "overload").then_some(witness) }
    }

    // Independent schedule enumeration with pairwise rejection, no Qayd code.
    fn solutions(&self) -> Vec<Vec<i32>> {
        fn visit(bounds: &[(i32, i32)], partial: &mut Vec<i32>, results: &mut Vec<Vec<i32>>) {
            if partial.len() == bounds.len() {
                results.push(partial.clone());
                return;
            }
            let (lo, hi) = bounds[partial.len()];
            for value in lo..=hi {
                if partial.iter().all(|&old| i64::from(old) + 2 <= i64::from(value) || i64::from(value) + 2 <= i64::from(old)) {
                    partial.push(value);
                    visit(bounds, partial, results);
                    partial.pop();
                }
            }
        }
        assert!(self.k <= 4);
        let mut out = Vec::new();
        visit(&self.bounds, &mut Vec::new(), &mut out);
        assert_eq!(out.is_empty(), self.witness.is_none());
        out
    }

    fn solver(&self, arm: &str) -> (Solver, Vec<VarId>) {
        let mut solver = Solver::new();
        solver.set_unary_edge_finding(arm == "edge");
        solver.set_unary_edge_finding_metrics(true);
        let vars = self.bounds.iter().map(|&(lo, hi)| solver.new_var_range(lo, hi)).collect::<Vec<_>>();
        no_overlap(&mut solver, &vars, &vec![2; vars.len()]);
        if arm == "cumulative" {
            cumulative(&mut solver, &vars, &vec![2; vars.len()], &vec![1; vars.len()], 1);
        }
        (solver, vars)
    }

    fn model(&self, arm: &str) -> ModelPackage {
        let mut model = Model::new();
        let starts = self.bounds.iter().map(|&(lo, hi)| model.int_range(lo, hi)).collect::<Vec<_>>();
        let durations = vec![2; starts.len()];
        model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap {
            starts: starts.clone(),
            durations: durations.clone(),
        }));
        if arm == "cumulative" {
            model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Cumulative {
                demands: vec![1; starts.len()],
                starts,
                durations,
                capacity: 1,
            }));
        }
        ModelPackage::new(model)
    }
}

fn stats_json(stats: UnaryEdgeFindingStats) -> String {
    format!(
        r#"{{"unary_calls":{},"directional_passes":{},"tree_operations":{},"overloads":{},"bound_updates":{},"eliminated_starts":{},"reason_premises":{},"unsupported_calls":{}}}"#,
        stats.propagation_calls,
        stats.directional_passes,
        stats.tree_operations,
        stats.overloads,
        stats.bound_updates,
        stats.eliminated_starts,
        stats.reason_premises,
        stats.unsupported_calls
    )
}

fn root_row(case: &Case, arm: &str, repetition: i32, expected: Option<&[Vec<i32>]>) -> String {
    let (mut solver, vars) = case.solver(arm);
    let before = vars.iter().map(|&var| solver.store.size(var) as u64).sum::<u64>();
    reset_prop_calls();
    let started = Instant::now();
    let success = solver.propagate().is_ok();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let calls = prop_calls();
    let stats = solver.unary_edge_finding_stats();
    // Store::values uses arbitrary sparse-set order. Canonicalize the snapshot
    // after timing so endpoint checks and repeated measurements compare bounds.
    let domains = vars
        .iter()
        .map(|&var| {
            let mut domain = solver.store.values(var).collect::<Vec<_>>();
            domain.sort_unstable();
            domain
        })
        .collect::<Vec<_>>();
    if let Some(witness) = &case.witness {
        assert!(success, "{} {} {arm}", case.family, case.k);
        for (domain, value) in domains.iter().zip(witness) {
            assert!(domain.contains(value));
        }
    } else {
        // k+1 length-2 tasks cannot fit the common length-2k window.
        assert_eq!(case.bounds.len(), case.k + 1);
        assert_eq!(case.bounds[0].1 + 2 - case.bounds[0].0, 2 * case.k as i32);
        if arm != "native" {
            assert!(!success, "overload not detected by {arm}");
        }
    }
    if let Some(expected) = expected {
        assert!(success || expected.is_empty());
        if success {
            for assignment in expected {
                for (domain, value) in domains.iter().zip(assignment) {
                    assert!(domain.contains(value));
                }
            }
        }
    }
    if arm == "edge" && case.family == "prefix" {
        assert_eq!(domains[case.k].first(), Some(&(2 * case.k as i32)));
    }
    if arm == "edge" && case.family == "mirror" {
        assert_eq!(domains[case.k].last(), Some(&(-2 * case.k as i32 - 2)));
    }
    let after = domains.iter().map(|domain| domain.len() as u64).sum::<u64>();
    format!(
        r#"{{"family":"{}","k":{},"arm":"{}","repetition":{},"status":"{}","domains":{:?},"removed_starts":{},"propagator_calls":{},"elapsed_ms":{},"unary":{}}}"#,
        case.family,
        case.k,
        arm,
        repetition,
        if success { "fixpoint" } else { "conflict" },
        domains,
        before - after,
        calls,
        elapsed_ms,
        stats_json(stats)
    )
}

#[test]
#[ignore]
fn unary_metrics_sweep() {
    for family in FAMILIES {
        for k in [2, 4, 8, 16, 32] {
            let case = Case::new(family, k);
            let expected = (k <= 4).then(|| case.solutions());
            for arm in ARMS {
                root_row(&case, arm, -1, expected.as_deref());
            }
            for repetition in 0..7 {
                for offset in 0..3 {
                    let arm = ARMS[(repetition + offset) % 3];
                    let row = root_row(&case, arm, repetition as i32, expected.as_deref());
                    eprintln!("I4_ROOT {row}");
                }
            }
        }
    }
    for family in FAMILIES {
        for k in [2, 3, 4] {
            let case = Case::new(family, k);
            let expected = case.solutions();
            for (seed_index, seed) in [0, 1, 7].into_iter().enumerate() {
                for offset in 0..3 {
                    let arm = ARMS[(seed_index + offset) % 3];
                    let package = case.model(arm);
                    let request = SolveRequest {
                        mode: SolveMode::Exact,
                        seed,
                        threads: 1,
                        cp: CpControls { unary_edge_finding: arm == "edge", unary_edge_finding_metrics: true, ..CpControls::default() },
                        search_policy: SearchPolicy::new(vec![SearchPhase::new(
                            (0..case.bounds.len()).collect(),
                            VariableSelector::Activity,
                            ValueSelector::Min,
                        )]),
                        ..SolveRequest::default()
                    };
                    let started = Instant::now();
                    let result = solve_model_silent(&package, &request).unwrap();
                    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
                    result.validate_contract().unwrap();
                    if expected.is_empty() {
                        assert_eq!(result.status(), SolveStatus::Unsatisfiable);
                        assert!(result.primal().is_none());
                        assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Unsatisfiable));
                    } else {
                        assert_eq!(result.status(), SolveStatus::Satisfiable);
                        let primal = result.primal().unwrap();
                        assert_eq!(primal.verification(), VerificationLevel::Final);
                        let values =
                            primal.assignment().integers.iter().map(|value| i32::try_from(value.unwrap()).unwrap()).collect::<Vec<_>>();
                        assert!(expected.contains(&values));
                    }
                    let stats = result.aggregate_search_stats();
                    let unary = result
                        .reports()
                        .iter()
                        .flat_map(|report| &report.metadata)
                        .filter(|(key, _)| key.starts_with("unary_") && key != "unary_edge_finding")
                        .fold(BTreeMap::<String, u64>::new(), |mut map, (key, value)| {
                            *map.entry(key.clone()).or_default() += value.parse::<u64>().unwrap();
                            map
                        });
                    eprintln!(
                        r#"I4_SEARCH {{"family":"{}","k":{},"seed":{},"arm":"{}","status":"{:?}","nodes":{},"failures":{},"propagator_calls":{},"elapsed_ms":{},"unary":{:?}}}"#,
                        family,
                        k,
                        seed,
                        arm,
                        result.status(),
                        stats.nodes,
                        stats.failures,
                        stats.propagator_calls,
                        elapsed_ms,
                        unary
                    );
                }
            }
        }
    }
}
