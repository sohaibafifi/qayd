//! Public soundness and lifecycle gates for mandatory unary edge finding.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use qayd::constraints::{interval, scheduling};
use qayd::inference::InferenceCheckConfig;
use qayd::model::{Constraint, IntExpr, IntGlobalConstraint, Model, ModelPackage, Objective, Relation};
use qayd::orchestrator::{
    solve_model_silent, CpControls, IgnoreEvents, ProofRequest, ProvenConclusion, SearchPhase, SearchPolicy, SemanticAssumption,
    SemanticAssumptionOp, SemanticSolveSession, SolveLimits, SolveMode, SolveRequest, SolveResult, SolveStatus, ValueSelector,
    VariableSelector, VerificationLevel,
};
use qayd::{solve_search, SearchControl, Solver, VarId};

#[derive(Clone, Debug)]
struct Schedule {
    domains: Vec<Vec<i32>>,
    tasks: Vec<usize>,
    durations: Vec<i64>,
}

impl Schedule {
    fn distinct(domains: Vec<Vec<i32>>, durations: Vec<i64>) -> Self {
        Self { tasks: (0..domains.len()).collect(), domains, durations }
    }

    // An occupied-unit-slot oracle, independent of precedence bounds, trees,
    // theta sets, and the I2 checker's pairwise endpoint predicate. Durations
    // in the exhaustive suite are small, while times may be near i32 limits.
    fn oracle(&self) -> BTreeSet<Vec<i32>> {
        let mut assignments = vec![Vec::new()];
        for domain in &self.domains {
            assignments = assignments
                .into_iter()
                .flat_map(|prefix| {
                    domain.iter().map(move |&value| {
                        let mut values = prefix.clone();
                        values.push(value);
                        values
                    })
                })
                .collect();
        }
        assignments
            .into_iter()
            .filter(|values| {
                let mut occupied = BTreeSet::new();
                self.tasks.iter().zip(&self.durations).all(|(&variable, &duration)| {
                    (0..duration).all(|offset| occupied.insert(i128::from(values[variable]) + i128::from(offset)))
                })
            })
            .collect()
    }

    fn solver(&self, enabled: bool, checked: bool) -> (Solver, Vec<VarId>) {
        let mut solver = Solver::new();
        solver.set_unary_edge_finding(enabled);
        solver.set_unary_edge_finding_metrics(true);
        let variables = self.domains.iter().map(|values| solver.new_var_set(values)).collect::<Vec<_>>();
        let starts = self.tasks.iter().map(|&index| variables[index]).collect::<Vec<_>>();
        scheduling::no_overlap(&mut solver, &starts, &self.durations);
        if checked {
            solver.set_inference_checking(Some(InferenceCheckConfig::default()));
        }
        (solver, variables)
    }

    fn verify(&self) {
        let expected = self.oracle();
        for enabled in [false, true] {
            let (mut solver, variables) = self.solver(enabled, true);
            if solver.propagate().is_err() {
                assert!(expected.is_empty(), "false root conflict: enabled={enabled}, {self:?}");
                continue;
            }
            assert_preserved(&solver, &variables, &expected);
            let mut actual = BTreeSet::new();
            solve_search(&mut solver, &variables, |state| {
                actual.insert(variables.iter().map(|&var| state.store.value(var)).collect());
                SearchControl::Continue
            });
            assert_eq!(actual, expected, "enabled={enabled}, {self:?}");
            assert_eq!(solver.inference_check_stats().resource_limited, 0, "{self:?}");
        }
    }
}

fn assert_preserved(solver: &Solver, variables: &[VarId], expected: &BTreeSet<Vec<i32>>) {
    for assignment in expected {
        for (&variable, &value) in variables.iter().zip(assignment) {
            assert!(solver.store.contains(variable, value), "feasible start removed: {assignment:?}, variable={variable:?}");
        }
    }
}

fn strength(mirror: bool) -> Schedule {
    let tight: Vec<i32> = if mirror { (4..=7).collect() } else { (0..=3).collect() };
    Schedule::distinct(vec![tight.clone(), tight, (0..=8).collect()], vec![3, 3, 2])
}

#[test]
fn exhaustive_signed_sparse_and_tied_domains_preserve_every_schedule() {
    let small = (1..8).map(|mask| (-1..=1).filter(|value| mask & (1 << (value + 1)) != 0).collect::<Vec<_>>()).collect::<Vec<_>>();
    for left in &small {
        for right in &small {
            for a in 1..=2 {
                for b in 1..=2 {
                    Schedule::distinct(vec![left.clone(), right.clone()], vec![a, b]).verify();
                }
            }
        }
    }
    let domains = [vec![-3, -2, -1, 0], vec![-2, 0, 2], vec![0, 1, 2, 3], vec![-1, 2]];
    for a in &domains {
        for b in &domains {
            for c in &domains {
                for durations in [vec![1, 1, 1], vec![2, 1, 2]] {
                    Schedule::distinct(vec![a.clone(), b.clone(), c.clone()], durations).verify();
                }
            }
        }
    }
}

#[test]
fn forward_and_mirrored_bounds_are_strictly_stronger_than_existing_filters() {
    for mirror in [false, true] {
        let schedule = strength(mirror);
        let expected = schedule.oracle();
        assert!(!expected.is_empty());
        for cumulative in [false, true] {
            let (mut baseline, variables) = schedule.solver(false, false);
            if cumulative {
                scheduling::cumulative(&mut baseline, &variables, &schedule.durations, &[1, 1, 1], 1);
            }
            baseline.propagate().unwrap();
            assert_eq!(
                (baseline.store.min(variables[2]), baseline.store.max(variables[2])),
                if cumulative && !mirror { (6, 8) } else { (0, 8) }
            );
            baseline.set_inference_checking(Some(InferenceCheckConfig::default()));
            baseline.set_unary_edge_finding(true);
            baseline.propagate().unwrap();
            assert_preserved(&baseline, &variables, &expected);
            assert_eq!((baseline.store.min(variables[2]), baseline.store.max(variables[2])), if mirror { (0, 2) } else { (6, 8) });
            let stats = baseline.unary_edge_finding_stats();
            assert!(stats.directional_passes > 0);
            // Existing cumulative energetic filtering already finds the
            // forward bound. The native resource and both mirror comparisons
            // must demonstrate strictly additional, independently checked work.
            if !cumulative || mirror {
                assert!(stats.bound_updates > 0 && stats.reason_premises > 0 && stats.eliminated_starts > 0);
                assert!(baseline.inference_check_stats().checked_inferences > 0);
            }
            assert_eq!(baseline.inference_check_stats().resource_limited, 0);
        }
    }
}

#[test]
fn signed_time_reversal_and_integer_endpoints_preserve_supports() {
    for offset in [i32::MIN, -11, i32::MAX - 10] {
        for mirror in [false, true] {
            let mut schedule = strength(mirror);
            for domain in &mut schedule.domains {
                for value in domain {
                    *value += offset;
                }
            }
            schedule.verify();
        }
    }
    let original = Schedule::distinct(vec![vec![-4, -1, 2], vec![-3, 0, 3], vec![-2, 1, 4]], vec![2, 3, 1]);
    let reversed = Schedule::distinct(
        original
            .domains
            .iter()
            .zip(&original.durations)
            .map(|(domain, &duration)| {
                let mut values = domain.iter().map(|&value| -value - duration as i32).collect::<Vec<_>>();
                values.sort_unstable();
                values
            })
            .collect(),
        original.durations.clone(),
    );
    let reflected = original
        .oracle()
        .into_iter()
        .map(|values| values.iter().zip(&original.durations).map(|(&value, &duration)| -value - duration as i32).collect())
        .collect::<BTreeSet<Vec<i32>>>();
    assert_eq!(reversed.oracle(), reflected);
    original.verify();
    reversed.verify();
}

#[test]
fn duplicate_starts_empty_and_singleton_resources_keep_their_semantics() {
    Schedule { domains: vec![vec![-1, 0, 1]], tasks: vec![0, 0], durations: vec![1, 2] }.verify();
    Schedule { domains: vec![vec![-1, 0], vec![1, 2]], tasks: vec![0, 1, 0], durations: vec![1, 1, 1] }.verify();
    Schedule::distinct(Vec::new(), Vec::new()).verify();
    Schedule::distinct(vec![vec![i32::MIN, 0, i32::MAX]], vec![1]).verify();
}

#[test]
fn backtracking_and_clones_rebuild_bounds_without_losing_root_supports() {
    let schedule = strength(false);
    let expected = schedule.oracle();
    let (mut solver, variables) = schedule.solver(true, true);
    solver.propagate().unwrap();
    let handle = solver.unary_edge_finding_handle().unwrap();
    for first in [0, 3, 0] {
        solver.store.push_level();
        solver.store.fix(variables[0], first).unwrap();
        solver.propagate().unwrap();
        let filtered = expected.iter().filter(|values| values[0] == first).cloned().collect();
        assert_preserved(&solver, &variables, &filtered);
        assert_eq!(solver.store.value(variables[1]), 3 - first);
        solver.store.pop_level();
        solver.enqueue_all();
        solver.propagate().unwrap();
        assert_preserved(&solver, &variables, &expected);
    }
    let calls_before = handle.stats().propagation_calls;
    let mut child = solver.clone();
    child.store.push_level();
    child.store.fix(variables[0], 3).unwrap();
    child.propagate().unwrap();
    assert!(handle.stats().propagation_calls > calls_before);
    assert_preserved(&solver, &variables, &expected);
    assert_eq!(solver.unary_edge_finding_stats(), handle.stats());
    solver.set_unary_edge_finding(false);
    solver.propagate().unwrap();
    assert_preserved(&solver, &variables, &expected);
}

#[test]
fn cancellation_during_propagation_can_be_requeued_without_false_conflicts() {
    for limit in [0, 1, 5, 15, 30, 60, 120] {
        let schedule = strength(false);
        let expected = schedule.oracle();
        let (mut solver, variables) = schedule.solver(true, true);
        let polls = AtomicUsize::new(0);
        solver.propagate_until(|| polls.fetch_add(1, Ordering::Relaxed) >= limit).unwrap();
        assert_preserved(&solver, &variables, &expected);
        solver.enqueue_all();
        solver.propagate().unwrap();
        assert_eq!(solver.store.min(variables[2]), 6);
        assert_preserved(&solver, &variables, &expected);
    }
}

#[test]
fn unsupported_zero_optional_order_and_wide_duration_paths_remain_unchanged() {
    let zero = Schedule::distinct(vec![vec![0, 1, 2], vec![0, 1, 2]], vec![0, 2]);
    let mut zero_domains = Vec::new();
    for enabled in [false, true] {
        let (mut solver, vars) = zero.solver(enabled, false);
        solver.propagate().unwrap();
        zero_domains.push(vars.iter().map(|&var| solver.store.values(var).collect::<Vec<_>>()).collect::<Vec<_>>());
        assert_eq!(solver.unary_edge_finding_stats().directional_passes, 0);
        if enabled {
            assert!(solver.unary_edge_finding_stats().unsupported_calls > 0);
        }
    }
    assert_eq!(zero_domains[0], zero_domains[1]);
    for optional in [false, true] {
        let mut all = Vec::new();
        for enabled in [false, true] {
            let mut solver = Solver::new();
            solver.set_unary_edge_finding(enabled);
            solver.set_unary_edge_finding_metrics(true);
            let a = solver.store.new_interval(0, 2, 2);
            let b = if optional { solver.store.new_optional_interval(0, 2, 2) } else { solver.store.new_interval(0, 2, 2) };
            interval::no_overlap(&mut solver, &[a, b]);
            let mut variables = vec![solver.store.interval_start_var(a), solver.store.interval_start_var(b)];
            if let Some(presence) = solver.store.interval_presence_var(b) {
                variables.push(presence);
            }
            let mut actual = BTreeSet::new();
            solve_search(&mut solver, &variables, |state| {
                actual.insert(variables.iter().map(|&var| state.store.value(var)).collect::<Vec<_>>());
                SearchControl::Continue
            });
            all.push(actual);
            assert_eq!(solver.unary_edge_finding_stats().directional_passes, 0);
            if enabled {
                assert!(solver.unary_edge_finding_stats().unsupported_calls > 0);
            }
        }
        assert_eq!(all[0], all[1]);
        assert!(!all[0].is_empty());
    }
    // Durations outside the native i32 interval representation keep the legacy
    // fixed-task implementation. This singleton cannot conflict or prune.
    for enabled in [false, true] {
        let (mut solver, vars) = Schedule::distinct(vec![vec![-1, 0, 1]], vec![i64::from(i32::MAX) + 1]).solver(enabled, false);
        solver.propagate().unwrap();
        assert_eq!(solver.store.values(vars[0]).collect::<Vec<_>>(), vec![-1, 0, 1]);
        assert_eq!(solver.unary_edge_finding_stats().directional_passes, 0);
    }
}

fn package(schedule: &Schedule, objective: bool) -> ModelPackage {
    let mut model = Model::new();
    let starts = schedule.domains.iter().map(|values| model.int_set(values.clone())).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap {
        starts: schedule.tasks.iter().map(|&task| starts[task]).collect(),
        durations: schedule.durations.clone(),
    }));
    // A Boolean row exercises I3 production alongside unary explanations.
    // The nontrivial bridge below keeps both in the same CP component and
    // search engine. Every oracle schedule still extends with all Booleans 0.
    let booleans = (0..5).map(|_| model.int_range(0, 1)).collect::<Vec<_>>();
    model.add_constraint(Constraint::Linear { terms: booleans.iter().map(|&var| (1, var)).collect(), relation: Relation::Le, rhs: 1 });
    model.add_constraint(Constraint::Linear {
        terms: vec![(1, starts[0]), (1, booleans[0])],
        relation: Relation::Le,
        rhs: i64::from(*schedule.domains[0].iter().max().unwrap()),
    });
    if objective {
        model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(starts[2]) });
    }
    ModelPackage::new(model)
}

fn request(enabled: bool, threads: usize) -> SolveRequest {
    SolveRequest {
        mode: SolveMode::Exact,
        seed: 7,
        threads,
        cp: CpControls {
            unary_edge_finding: enabled,
            unary_edge_finding_metrics: true,
            semantic_minimization: true,
            deferred_explanations: true,
            explanation_metrics: true,
            check_inferences: true,
            ..CpControls::default()
        },
        search_policy: SearchPolicy::new(vec![SearchPhase::new(
            (3..8).chain(0..3).collect(),
            VariableSelector::Activity,
            ValueSelector::Max,
        )]),
        ..SolveRequest::default()
    }
}

fn count(result: &SolveResult, key: &str) -> u64 {
    result
        .reports()
        .iter()
        .flat_map(|report| &report.metadata)
        .filter(|(name, _)| name == key)
        .map(|(_, value)| value.parse::<u64>().unwrap())
        .sum()
}

fn verify_result(result: &SolveResult, expected: &BTreeSet<Vec<i32>>, objective: bool, bridge_max: i32) {
    result.validate_contract().unwrap();
    if expected.is_empty() {
        assert_eq!(result.status(), SolveStatus::Unsatisfiable);
        assert!(result.primal().is_none());
        assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Unsatisfiable));
    } else {
        assert_eq!(result.status(), if objective { SolveStatus::Optimal } else { SolveStatus::Satisfiable });
        let candidate = result.primal().unwrap();
        assert_eq!(candidate.verification(), VerificationLevel::Final);
        let values = candidate.assignment().integers.iter().map(|value| i32::try_from(value.unwrap()).unwrap()).collect::<Vec<_>>();
        assert!(expected.contains(&values[..3]));
        assert!(values[3..].iter().all(|&value| value == 0 || value == 1));
        assert!(values[3..].iter().sum::<i32>() <= 1);
        assert!(i64::from(values[0]) + i64::from(values[3]) <= i64::from(bridge_max));
        if objective {
            assert_eq!(candidate.objectives(), &[i64::from(expected.iter().map(|values| values[2]).min().unwrap())]);
            assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Optimal));
        }
    }
}

#[test]
fn canonical_csp_and_optimization_workers_preserve_oracle_results_with_i1_i2_i3() {
    assert!(!CpControls::default().unary_edge_finding);
    assert!(!CpControls::default().unary_edge_finding_metrics);
    let mut deferred_handles = 0;
    let mut checked = 0;
    for impossible in [false, true] {
        let mut schedule = strength(false);
        if impossible {
            schedule.domains[2] = (0..=5).collect();
        }
        let expected = schedule.oracle();
        for objective in [false, true] {
            for threads in [1, 2] {
                for enabled in [false, true] {
                    let mut options = request(enabled, threads);
                    if impossible || objective {
                        options.proof = ProofRequest::Require;
                    }
                    let result = solve_model_silent(&package(&schedule, objective), &options).unwrap_or_else(|error| {
                        panic!("impossible={impossible}, objective={objective}, threads={threads}, enabled={enabled}: {error:?}")
                    });
                    verify_result(&result, &expected, objective, *schedule.domains[0].iter().max().unwrap());
                    assert_eq!(count(&result, "inference_resource_limited"), 0);
                    checked += count(&result, "inference_checked_inferences");
                    deferred_handles += count(&result, "explanation_created_handles");
                    if enabled {
                        assert!(count(&result, "unary_directional_passes") > 0);
                    } else {
                        assert_eq!(count(&result, "unary_directional_passes"), 0);
                    }
                }
            }
        }
    }
    assert!(checked > 0 && deferred_handles > 0);
}

#[test]
fn persistent_sessions_toggle_unary_filtering_and_temporary_assumptions() {
    let schedule = strength(false);
    let expected = schedule.oracle();
    let mut session = SemanticSolveSession::new(package(&schedule, false)).unwrap();
    let stop = AtomicBool::new(false);
    for (enabled, start) in [(true, 6), (false, 0), (true, 0), (false, 8), (true, 7)] {
        let mut options = request(enabled, 1);
        options.assumptions.push(SemanticAssumption { variable: 2, operation: SemanticAssumptionOp::Eq, value: start });
        let result = session.solve_with_external_stop(&options, &stop, &mut IgnoreEvents).unwrap();
        let filtered = expected.iter().filter(|values| values[2] == start).cloned().collect();
        verify_result(&result, &filtered, false, *schedule.domains[0].iter().max().unwrap());
    }
    let result = session.solve_with_external_stop(&request(true, 1), &stop, &mut IgnoreEvents).unwrap();
    verify_result(&result, &expected, false, *schedule.domains[0].iter().max().unwrap());
}

#[test]
fn zero_deadline_cannot_publish_a_completion_claim() {
    for threads in [1, 2] {
        let mut options = request(true, threads);
        options.limits = SolveLimits { time: Some(Duration::ZERO), ..SolveLimits::default() };
        let result = solve_model_silent(&package(&strength(false), true), &options).unwrap();
        result.validate_contract().unwrap();
        assert_eq!(result.status(), SolveStatus::Unknown);
        assert!(result.primal().is_none() && result.proof().is_none());
    }
}

struct TemporaryInstance(PathBuf);

impl Drop for TemporaryInstance {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn cli_accepts_unary_options_and_reports_checked_propagation() {
    let instance = TemporaryInstance(std::env::temp_dir().join(format!("qayd-unary-edge-finding-{}.xml", std::process::id())));
    std::fs::write(
        &instance.0,
        r#"<instance format="XCSP3" type="CSP">
      <variables><var id="a">0..3</var><var id="b">0..3</var><var id="c">0..8</var></variables>
      <constraints><noOverlap><origins>a b c</origins><lengths>3 3 2</lengths></noOverlap></constraints>
    </instance>"#,
    )
    .unwrap();
    for enabled in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_qayd"));
        command.args(["--verbose", "--unary-edge-finding-metrics", "--check-inferences"]);
        if enabled {
            command.arg("--unary-edge-finding");
        }
        let output = command.arg(&instance.0).output().unwrap();
        assert!(output.status.success(), "{:?}: {}", output.status, String::from_utf8_lossy(&output.stderr));
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("s SATISFIABLE"), "{stdout}");
        assert!(stdout.contains(&format!("c unary_edge_finding {enabled}")), "{stdout}");
    }
}
