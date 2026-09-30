//! Independent occupancy and lifecycle gates for native cumulative modes.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use qayd::constraints::{interval, scheduling};
use qayd::inference::InferenceCheckConfig;
use qayd::model::{Constraint, IntExpr, IntGlobalConstraint, IntVarRef, Model, ModelPackage, Objective, Relation};
use qayd::orchestrator::{
    solve_model_silent, CpControls, IgnoreEvents, ProofRequest, ProvenConclusion, SearchPhase, SearchPolicy, SemanticAssumption,
    SemanticAssumptionOp, SemanticSolveSession, SolveLimits, SolveMode, SolveRequest, SolveResult, SolveStatus, ValueSelector,
    VariableSelector, VerificationLevel,
};
use qayd::{optimize_var_assuming, solve_search, Assumption, IntervalId, SearchControl, Solver, VarId};

mod common;
#[path = "common/cumulative_models.rs"]
pub mod cumulative_models;
use cumulative_models::Case;

const ARMS: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];
static NEVER_STOP: AtomicBool = AtomicBool::new(false);

fn configure(solver: &mut Solver, incremental: bool, lifted: bool, metrics: bool) {
    solver.set_incremental_cumulative(incremental);
    solver.set_lifted_cumulative_explanations(lifted);
    solver.set_cumulative_metrics(metrics);
    assert_eq!(solver.incremental_cumulative_enabled(), incremental);
    assert_eq!(solver.lifted_cumulative_explanations_enabled(), lifted);
}

fn enumerate(solver: &mut Solver, variables: &[VarId]) -> BTreeSet<Vec<i32>> {
    let mut actual = BTreeSet::new();
    solve_search(solver, variables, |state| {
        let values = variables.iter().map(|&variable| state.store.value(variable)).collect();
        assert!(actual.insert(values), "enumeration must not repeat an assignment");
        SearchControl::Continue
    });
    actual
}

fn preserved(solver: &Solver, variables: &[VarId], expected: &BTreeSet<Vec<i32>>) {
    for values in expected {
        for (&variable, &value) in variables.iter().zip(values) {
            assert!(solver.store.contains(variable, value), "removed feasible value {value} from {variable:?}: {values:?}");
        }
    }
}

fn snapshot(solver: &Solver, variables: &[VarId]) -> Vec<BTreeSet<i32>> {
    variables.iter().map(|&variable| solver.store.values(variable).collect()).collect()
}

#[test]
fn incremental_frozen_grid_preserves_complete_solution_sets() {
    let expected_counts = [[72, 400, 30, 936, 6, 1650], [16, 66, 8, 156, 4, 408]];
    for (family_index, family) in ["dense", "anchored"].into_iter().enumerate() {
        for n in 4..=6 {
            for capacity in [2, 3] {
                let case = Case::new(family, n, capacity);
                let expected = case.solutions();
                assert_eq!(expected.len(), expected_counts[family_index][(n - 4) * 2 + (capacity - 2) as usize]);
                for incremental in [false, true] {
                    let (mut solver, variables) = case.solver(incremental, false, true);
                    assert_eq!(
                        enumerate(&mut solver, &variables),
                        expected,
                        "{family}, n={n}, capacity={capacity}, incremental={incremental}"
                    );
                    let stats = solver.cumulative_stats();
                    assert!(stats.propagation_calls > 0);
                    if incremental {
                        assert!(stats.incremental_updates > 0);
                    } else {
                        assert!(stats.profile_rebuilds > 0);
                        assert_eq!(stats.incremental_updates, 0);
                    }
                    assert_eq!(stats.lifted_bounds, 0);
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct NativeCase {
    domains: Vec<Vec<i32>>,
    durations: Vec<i32>,
    optional: Vec<bool>,
    // Repeated indices deliberately repeat the same IntervalId, with their
    // own resource demand on each occurrence.
    occurrences: Vec<(usize, i32)>,
    capacity: i32,
}

impl NativeCase {
    fn assignment_domains(&self) -> Vec<Vec<i32>> {
        let mut domains = self.domains.clone();
        domains.extend(self.optional.iter().filter(|&&optional| optional).map(|_| vec![0, 1]));
        domains
    }

    fn accepts(&self, values: &[i32]) -> bool {
        let domains = self.assignment_domains();
        if values.len() != domains.len() || values.iter().zip(&domains).any(|(value, domain)| !domain.contains(value)) {
            return false;
        }
        let mut occupied = BTreeMap::<i128, i128>::new();
        for &(index, demand) in &self.occurrences {
            if self.optional[index] {
                let presence = self.domains.len() + self.optional[..index].iter().filter(|&&optional| optional).count();
                if values[presence] == 0 {
                    continue;
                }
            }
            for offset in 0..self.durations[index] {
                let usage = occupied.entry(i128::from(values[index]) + i128::from(offset)).or_default();
                *usage += i128::from(demand);
                if *usage > i128::from(self.capacity) {
                    return false;
                }
            }
        }
        true
    }

    fn solutions(&self) -> BTreeSet<Vec<i32>> {
        common::oracle::solutions(&self.assignment_domains(), |values| self.accepts(values)).into_iter().collect()
    }

    fn solver(&self, incremental: bool, lifted: bool, checked: bool) -> (Solver, Vec<VarId>, Vec<IntervalId>) {
        let mut solver = Solver::new();
        configure(&mut solver, incremental, lifted, true);
        let mut intervals = Vec::new();
        for (index, domain) in self.domains.iter().enumerate() {
            let lower = *domain.iter().min().unwrap();
            let upper = *domain.iter().max().unwrap();
            assert!(i64::from(upper) - i64::from(lower) <= 32, "test domains must be cheap to trim");
            let interval = if self.optional[index] {
                solver.store.new_optional_interval(lower, upper, self.durations[index])
            } else {
                solver.store.new_interval(lower, upper, self.durations[index])
            };
            let start = solver.store.interval_start_var(interval);
            for value in lower..=upper {
                if !domain.contains(&value) {
                    solver.store.remove(start, value).unwrap();
                }
            }
            intervals.push(interval);
        }
        let mut variables = intervals.iter().map(|&interval| solver.store.interval_start_var(interval)).collect::<Vec<_>>();
        variables.extend(intervals.iter().filter_map(|&interval| solver.store.interval_presence_var(interval)));
        let tasks = self.occurrences.iter().map(|&(index, _)| intervals[index]).collect::<Vec<_>>();
        let demands = self.occurrences.iter().map(|&(_, demand)| demand).collect::<Vec<_>>();
        interval::cumulative(&mut solver, &tasks, &demands, self.capacity);
        if checked {
            solver.set_inference_checking(Some(InferenceCheckConfig::default()));
        }
        (solver, variables, intervals)
    }

    fn verify(&self) {
        let expected = self.solutions();
        for (incremental, lifted) in ARMS {
            let (mut solver, variables, _) = self.solver(incremental, lifted, true);
            assert_eq!(enumerate(&mut solver, &variables), expected, "incremental={incremental}, lifted={lifted}, case={self:?}");
            assert_eq!(solver.inference_check_stats().resource_limited, 0, "{self:?}");
            if variables.is_empty() {
                continue;
            }
            for minimizing in [false, true] {
                let (mut solver, variables, _) = self.solver(incremental, lifted, true);
                let (best, _, complete) =
                    optimize_var_assuming(&mut solver, &variables, &[], variables[0], minimizing, &NEVER_STOP, |_, values| {
                        assert!(expected.contains(values));
                    });
                let optimum =
                    if minimizing { expected.iter().map(|values| values[0]).min() } else { expected.iter().map(|values| values[0]).max() };
                assert!(complete);
                assert_eq!(best.as_ref().map(|(_, value)| *value), optimum.map(i64::from));
                if let Some((values, _)) = best {
                    assert!(expected.contains(&values));
                }
            }
        }
    }
}

#[test]
fn deterministic_generated_sparse_resources_match_the_checked_occupancy_oracle() {
    fn next(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    const SEED: u64 = 0x15C5_9F04_A7D2_631B;
    let mut state = SEED;
    let mut feasible = 0;
    let mut infeasible = 0;
    let mut root_pruned = 0;
    let mut checked = 0;
    for round in 0..64 {
        let n = 3 + round % 2;
        let domains = (0..n)
            .map(|_| {
                let size = 2 + (next(&mut state) % 2) as usize;
                let mut values = BTreeSet::new();
                while values.len() < size {
                    values.insert((next(&mut state) % 6) as i32 - 2);
                }
                values.into_iter().collect()
            })
            .collect();
        let durations = (0..n).map(|_| (next(&mut state) % 5) as i32).collect();
        let mut optional = vec![false; n];
        if round % 3 != 0 {
            optional[round % n] = true;
        }
        if round % 5 == 0 {
            optional[(round + 1) % n] = true;
        }
        let mut occurrences = (0..n).map(|index| (index, (next(&mut state) % 4) as i32)).collect::<Vec<_>>();
        if round % 4 == 1 {
            occurrences.push(((next(&mut state) % n as u64) as usize, (next(&mut state) % 4) as i32));
        }
        let mut case = NativeCase { domains, durations, optional, occurrences, capacity: 1 + (next(&mut state) % 4) as i32 };
        if round % 8 == 0 {
            // Anchor compulsory parts in some cases so the corpus includes
            // useful root filtering, beyond fully fixed capacity conflicts.
            case.domains = vec![vec![-1, 0], vec![0, 1], vec![0, 1, 3]];
            case.durations = vec![3 + (round / 8 % 2) as i32, 3, 2];
            case.optional.fill(false);
            case.occurrences = vec![(0, 1), (1, 1), (2, 1)];
            case.capacity = 2;
        }
        let expected = case.solutions();
        if expected.is_empty() {
            infeasible += 1;
        } else {
            feasible += 1;
        }
        for (incremental, lifted) in ARMS {
            // Include the generator position even if I2 itself rejects a
            // reason by panicking before the solution comparison is reached.
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let (mut solver, variables, _) = case.solver(incremental, lifted, true);
                if solver.propagate().is_err() {
                    assert!(expected.is_empty(), "false root conflict");
                    return (BTreeSet::new(), false, solver.inference_check_stats());
                }
                preserved(&solver, &variables, &expected);
                let pruned = case
                    .assignment_domains()
                    .iter()
                    .zip(&variables)
                    .any(|(domain, &variable)| domain.iter().any(|&value| !solver.store.contains(variable, value)));
                let actual = enumerate(&mut solver, &variables);
                (actual, pruned, solver.inference_check_stats())
            }));
            let (actual, pruned, checks) = outcome.unwrap_or_else(|payload| {
                let reason = payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| payload.downcast_ref::<&str>().copied())
                    .unwrap_or("non-string panic");
                panic!("seed={SEED:#x}, round={round}, incremental={incremental}, lifted={lifted}, case={case:?}: {reason}");
            });
            assert_eq!(actual, expected, "seed={SEED:#x}, round={round}, incremental={incremental}, lifted={lifted}, case={case:?}");
            assert_eq!(checks.resource_limited, 0, "seed={SEED:#x}, round={round}, case={case:?}");
            root_pruned += usize::from(pruned && !expected.is_empty());
            checked += checks.checked_inferences;
        }
    }
    assert!(feasible > 0 && infeasible > 0, "the generated corpus must cover both outcomes");
    assert!(root_pruned > 0 && checked > 0, "the generated corpus must exercise checked resource filtering");
}

#[test]
fn four_modes_preserve_optional_duplicate_zero_and_signed_resource_semantics() {
    let cases = [
        NativeCase {
            domains: vec![vec![-2, 0, 1], vec![-1, 0, 2], vec![-2, 1, 3]],
            durations: vec![3, 2, 1],
            optional: vec![false; 3],
            occurrences: vec![(0, 1), (1, 2), (2, 1)],
            capacity: 2,
        },
        NativeCase {
            domains: vec![vec![0, 1, 3], vec![0, 2, 4]],
            durations: vec![2, 2],
            optional: vec![false; 2],
            occurrences: vec![(0, 1), (0, 1), (1, 1)],
            capacity: 2,
        },
        NativeCase {
            domains: vec![vec![0, 1, 3], vec![0, 2, 4]],
            durations: vec![2, 2],
            optional: vec![false; 2],
            occurrences: vec![(0, 1), (0, 1), (1, 1)],
            capacity: 3,
        },
        NativeCase {
            domains: vec![vec![0, 1], vec![0, 1, 2]],
            durations: vec![2, 2],
            optional: vec![true, false],
            occurrences: vec![(0, 1), (0, 2), (1, 1)],
            capacity: 2,
        },
        NativeCase {
            domains: vec![vec![0, 2]; 3],
            durations: vec![2; 3],
            optional: vec![false, true, true],
            occurrences: vec![(0, 2), (1, 1), (2, 1)],
            capacity: 2,
        },
        NativeCase {
            domains: vec![vec![-1, 0, 2]; 3],
            durations: vec![0, 2, 1],
            optional: vec![false; 3],
            occurrences: vec![(0, 100), (1, 0), (2, 1)],
            capacity: 1,
        },
        NativeCase { domains: Vec::new(), durations: Vec::new(), optional: Vec::new(), occurrences: Vec::new(), capacity: 0 },
    ];
    for case in cases {
        case.verify();
    }
}

#[test]
fn widened_occupancy_oracle_covers_both_integer_endpoints_and_distant_tasks() {
    for origin in [i32::MIN, -7, i32::MAX - 2] {
        NativeCase {
            domains: vec![vec![origin, origin + 1, origin + 2]; 2],
            durations: vec![2, 2],
            optional: vec![false; 2],
            occurrences: vec![(0, 1), (1, 1)],
            capacity: 1,
        }
        .verify();
    }
    NativeCase {
        domains: vec![vec![i32::MIN, i32::MIN + 1], vec![i32::MAX - 1, i32::MAX]],
        durations: vec![3, 3],
        optional: vec![false, true],
        occurrences: vec![(0, 2), (1, 2)],
        capacity: 2,
    }
    .verify();
}

#[test]
fn incremental_nested_pop_same_depth_siblings_and_clones_match_fresh_filtering() {
    let case = Case::new("dense", 4, 2);
    let expected = case.solutions();
    let (mut reference, variables) = case.solver(false, false, true);
    let (mut cached, cached_variables) = case.solver(true, false, true);
    assert_eq!(variables, cached_variables);
    reference.propagate().unwrap();
    cached.propagate().unwrap();
    let root = snapshot(&reference, &variables);
    assert_eq!(root, snapshot(&cached, &variables));
    for first in [0, 4, 0] {
        let parent_solutions = expected.iter().filter(|values| values[0] == first).cloned().collect::<BTreeSet<_>>();
        for solver in [&mut reference, &mut cached] {
            solver.store.push_level();
            solver.store.fix(variables[0], first).unwrap();
            solver.propagate().unwrap();
            preserved(solver, &variables, &parent_solutions);
        }
        assert_eq!(snapshot(&reference, &variables), snapshot(&cached, &variables));
        let parent = snapshot(&reference, &variables);
        let second_values = parent_solutions.iter().map(|values| values[1]).collect::<BTreeSet<_>>();
        for second in second_values {
            let nested = parent_solutions.iter().filter(|values| values[1] == second).cloned().collect::<BTreeSet<_>>();
            for solver in [&mut reference, &mut cached] {
                solver.store.push_level();
                solver.store.fix(variables[1], second).unwrap();
                solver.propagate().unwrap();
                preserved(solver, &variables, &nested);
            }
            assert_eq!(snapshot(&reference, &variables), snapshot(&cached, &variables));
            let mut child = cached.clone();
            assert!(child.incremental_cumulative_enabled());
            child.store.push_level();
            let witness = nested.first().unwrap();
            child.store.fix(variables[2], witness[2]).unwrap();
            child.propagate().unwrap();
            preserved(&child, &variables, &BTreeSet::from([witness.clone()]));
            assert_eq!(snapshot(&reference, &variables), snapshot(&cached, &variables));
            for solver in [&mut reference, &mut cached] {
                solver.store.pop_level();
                solver.enqueue_all();
                solver.propagate().unwrap();
                assert_eq!(snapshot(solver, &variables), parent);
            }
        }
        for solver in [&mut reference, &mut cached] {
            solver.store.pop_level();
            solver.enqueue_all();
            solver.propagate().unwrap();
            assert_eq!(snapshot(solver, &variables), root);
        }
    }
    assert!(cached.cumulative_stats().restored_tasks > 0);
    assert_eq!(enumerate(&mut cached.clone(), &variables), expected);
}

#[test]
fn root_toggles_and_disabled_metrics_preserve_the_native_and_legacy_contracts() {
    let controls = CpControls::default();
    assert!(!controls.incremental_cumulative && !controls.lifted_cumulative_explanations && !controls.cumulative_metrics);
    let mut default_solver = Solver::new();
    assert!(!default_solver.incremental_cumulative_enabled() && !default_solver.lifted_cumulative_explanations_enabled());
    assert_eq!(default_solver.cumulative_stats(), Default::default());
    let case = Case::new("anchored", 4, 2);
    let expected = case.solutions();
    let (mut solver, variables) = case.solver(false, false, true);
    solver.propagate().unwrap();
    let root = snapshot(&solver, &variables);
    for (incremental, lifted) in ARMS.into_iter().chain(ARMS.into_iter().rev()) {
        solver.set_incremental_cumulative(incremental);
        solver.set_lifted_cumulative_explanations(lifted);
        solver.propagate().unwrap();
        assert_eq!(snapshot(&solver, &variables), root);
        let mut clone = solver.clone();
        assert_eq!(clone.incremental_cumulative_enabled(), incremental);
        assert_eq!(clone.lifted_cumulative_explanations_enabled(), lifted);
        assert_eq!(enumerate(&mut clone, &variables), expected);
    }
    solver.set_cumulative_metrics(false);
    assert_eq!(enumerate(&mut solver, &variables), expected);
    assert_eq!(solver.cumulative_stats(), Default::default());

    configure(&mut default_solver, true, true, true);
    let starts = (0..2).map(|_| default_solver.new_var_range(0, 2)).collect::<Vec<_>>();
    scheduling::cumulative(&mut default_solver, &starts, &[2, 2], &[1, 1], 1);
    assert_eq!(enumerate(&mut default_solver, &starts), BTreeSet::from([vec![0, 2], vec![2, 0]]));
    assert_eq!(default_solver.cumulative_stats(), Default::default(), "the legacy scheduling adapter is outside I5");
}

#[test]
fn optional_presence_and_start_changes_restore_the_profile_across_siblings() {
    let case = NativeCase {
        domains: vec![vec![0], vec![0, 2], vec![0, 2]],
        durations: vec![2; 3],
        optional: vec![false, true, true],
        occurrences: vec![(0, 1), (1, 1), (2, 1)],
        capacity: 2,
    };
    let expected = case.solutions();
    for lifted in [false, true] {
        let (mut reference, variables, _) = case.solver(false, lifted, true);
        let (mut cached, cached_variables, intervals) = case.solver(true, lifted, true);
        assert_eq!(variables, cached_variables);
        let presence = cached.store.interval_presence_var(intervals[1]).unwrap();
        reference.propagate().unwrap();
        cached.propagate().unwrap();
        let root = snapshot(&reference, &variables);
        for present in [1, 0, 1] {
            let supported = expected.iter().filter(|values| values[3] == present).cloned().collect::<BTreeSet<_>>();
            for solver in [&mut reference, &mut cached] {
                solver.store.push_level();
                solver.store.fix(presence, present).unwrap();
                solver.propagate().unwrap();
                preserved(solver, &variables, &supported);
                solver.store.push_level();
                solver.store.fix(variables[1], 0).unwrap();
                solver.propagate().unwrap();
                let fixed = supported.iter().filter(|values| values[1] == 0).cloned().collect();
                preserved(solver, &variables, &fixed);
            }
            assert_eq!(snapshot(&reference, &variables), snapshot(&cached, &variables));
            for solver in [&mut reference, &mut cached] {
                solver.store.pop_level();
                solver.store.pop_level();
                solver.enqueue_all();
                solver.propagate().unwrap();
                assert_eq!(snapshot(solver, &variables), root);
            }
        }
        assert_eq!(enumerate(&mut cached, &variables), expected);
    }
}

#[test]
fn nonroot_registration_uses_a_sound_fallback_after_root_rollback() {
    for (incremental, lifted) in ARMS {
        let mut solver = Solver::new();
        configure(&mut solver, incremental, lifted, true);
        let intervals = [solver.store.new_interval(0, 2, 2), solver.store.new_interval(0, 2, 2)];
        let variables = intervals.map(|interval| solver.store.interval_start_var(interval));
        solver.set_inference_checking(Some(InferenceCheckConfig::default()));
        solver.store.push_level();
        solver.store.fix(variables[0], 0).unwrap();
        interval::cumulative(&mut solver, &intervals, &[1, 1], 1);
        solver.propagate().unwrap();
        assert_eq!(solver.store.value(variables[1]), 2);
        solver.store.pop_level();
        solver.enqueue_all();
        solver.propagate().unwrap();
        assert_eq!(enumerate(&mut solver, &variables), BTreeSet::from([vec![0, 2], vec![2, 0]]));
        if incremental {
            assert!(solver.cumulative_stats().fallback_calls > 0);
        }
    }
}

fn request(count: usize, incremental: bool, lifted: bool, threads: usize) -> SolveRequest {
    SolveRequest {
        mode: SolveMode::Exact,
        seed: 7,
        threads,
        cp: CpControls {
            incremental_cumulative: incremental,
            lifted_cumulative_explanations: lifted,
            cumulative_metrics: true,
            check_inferences: true,
            ..CpControls::default()
        },
        search_policy: SearchPolicy::new(vec![SearchPhase::new(
            (0..count).collect(),
            VariableSelector::Activity,
            ValueSelector::RandomSeeded,
        )]),
        ..SolveRequest::default()
    }
}

fn assignment(result: &SolveResult) -> Vec<i32> {
    let candidate = result.primal().unwrap();
    assert_eq!(candidate.verification(), VerificationLevel::Final);
    candidate.assignment().integers.iter().map(|value| i32::try_from(value.unwrap()).unwrap()).collect()
}

fn check_result(result: &SolveResult, expected: &BTreeSet<Vec<i32>>, objective: Option<bool>) {
    result.validate_contract().unwrap();
    if expected.is_empty() {
        assert_eq!(result.status(), SolveStatus::Unsatisfiable);
        assert!(result.primal().is_none());
        assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Unsatisfiable));
    } else {
        assert_eq!(result.status(), if objective.is_some() { SolveStatus::Optimal } else { SolveStatus::Satisfiable });
        assert!(expected.contains(&assignment(result)));
        if let Some(minimizing) = objective {
            let optimum =
                if minimizing { expected.iter().map(|values| values[0]).min() } else { expected.iter().map(|values| values[0]).max() };
            assert_eq!(result.primal().unwrap().objectives(), &[i64::from(optimum.unwrap())]);
            assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Optimal));
        }
    }
}

#[test]
fn canonical_native_lowering_keeps_verified_statuses_and_both_optima_with_two_workers() {
    for impossible in [false, true] {
        let mut case = Case::new("dense", 4, 2);
        if impossible {
            case.domains = vec![vec![0, 1]; 4];
        }
        let expected = case.solutions();
        assert_eq!(expected.is_empty(), impossible);
        for objective in [None, Some(true), Some(false)] {
            let mut package = case.model();
            if let Some(minimize) = objective {
                package.model.add_objective(Objective::IntExpr { minimize, expr: IntExpr::Variable(IntVarRef(0)) });
            }
            for (incremental, lifted) in ARMS {
                for threads in [1, 2] {
                    let mut options = request(case.n, incremental, lifted, threads);
                    if objective.is_some() || impossible {
                        options.proof = ProofRequest::Require;
                    }
                    let result = solve_model_silent(&package, &options).unwrap_or_else(|error| {
                        panic!("impossible={impossible}, objective={objective:?}, incremental={incremental}, lifted={lifted}, threads={threads}: {error:?}")
                    });
                    check_result(&result, &expected, objective);
                }
            }
        }
    }
}

#[test]
fn repeated_semantic_start_ids_keep_every_resource_occurrence() {
    let domains = vec![vec![-1, 0, 2], vec![-1, 1, 3]];
    let expected = common::oracle::solutions(&domains, |values| {
        let mut loads = BTreeMap::<i64, i64>::new();
        for (start, duration, demand) in [(values[0], 2, 1), (values[0], 1, 1), (values[1], 2, 1)] {
            for offset in 0..duration {
                *loads.entry(i64::from(start) + offset).or_default() += demand;
            }
        }
        loads.values().all(|&load| load <= 2)
    })
    .into_iter()
    .collect::<BTreeSet<_>>();
    let mut model = Model::new();
    let variables = domains.iter().map(|domain| model.int_set(domain.clone())).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Cumulative {
        starts: vec![variables[0], variables[0], variables[1]],
        durations: vec![2, 1, 2],
        demands: vec![1, 1, 1],
        capacity: 2,
    }));
    let package = ModelPackage::new(model);
    for (incremental, lifted) in ARMS {
        for values in common::oracle::solutions(&domains, |_| true) {
            let mut options = request(2, incremental, lifted, 1);
            options.assumptions = values
                .iter()
                .enumerate()
                .map(|(variable, &value)| SemanticAssumption { variable, operation: SemanticAssumptionOp::Eq, value })
                .collect();
            let result = solve_model_silent(&package, &options).unwrap();
            let supported = if expected.contains(&values) { BTreeSet::from([values]) } else { BTreeSet::new() };
            check_result(&result, &supported, None);
        }
    }
}

#[test]
fn temporary_assumptions_and_session_toggles_do_not_retain_profile_state() {
    let case = Case::new("dense", 4, 2);
    let expected = case.solutions();
    let mut session = SemanticSolveSession::new(case.model()).unwrap();
    for (incremental, lifted) in ARMS.into_iter().chain(ARMS.into_iter().rev()) {
        for first in [0, 4] {
            let filtered = expected.iter().filter(|values| values[0] == first).cloned().collect::<BTreeSet<_>>();
            let (mut solver, variables) = case.solver(incremental, lifted, true);
            let (best, _, complete) = optimize_var_assuming(
                &mut solver,
                &variables,
                &[Assumption::eq(variables[0], first)],
                variables[1],
                true,
                &NEVER_STOP,
                |_, _| {},
            );
            assert!(complete);
            assert_eq!(best.as_ref().map(|(_, value)| *value), filtered.iter().map(|values| i64::from(values[1])).min());
            assert!(filtered.contains(&best.unwrap().0));

            let mut options = request(case.n, incremental, lifted, 1);
            options.assumptions = vec![SemanticAssumption { variable: 0, operation: SemanticAssumptionOp::Eq, value: first }];
            let result = session.solve_with_external_stop(&options, &NEVER_STOP, &mut IgnoreEvents).unwrap();
            check_result(&result, &filtered, None);
            options.assumptions =
                (0..case.n).map(|variable| SemanticAssumption { variable, operation: SemanticAssumptionOp::Eq, value: 0 }).collect();
            options.proof = ProofRequest::Require;
            let result = session.solve_with_external_stop(&options, &NEVER_STOP, &mut IgnoreEvents).unwrap();
            check_result(&result, &BTreeSet::new(), None);
        }
    }
    let result = session.solve_with_external_stop(&request(case.n, true, true, 1), &NEVER_STOP, &mut IgnoreEvents).unwrap();
    check_result(&result, &expected, None);
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

#[test]
fn cumulative_modes_coexist_with_i1_i2_i3_i4_and_i6_on_a_connected_model() {
    let domains = vec![(0..=3).collect::<Vec<_>>(), (0..=3).collect(), (0..=8).collect()];
    let durations = [3, 3, 2];
    let expected = common::oracle::solutions(&domains, |values| {
        let mut occupied = BTreeSet::new();
        values.iter().zip(durations).all(|(&start, duration)| (start..start + duration).all(|time| occupied.insert(time)))
    })
    .into_iter()
    .collect::<BTreeSet<_>>();
    let mut model = Model::new();
    let starts = domains.iter().map(|domain| model.int_set(domain.clone())).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap {
        starts: starts.clone(),
        durations: durations.map(i64::from).to_vec(),
    }));
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Cumulative {
        starts: starts.clone(),
        durations: durations.map(i64::from).to_vec(),
        demands: vec![1; 3],
        capacity: 2,
    }));
    let booleans = (0..5).map(|_| model.int_range(0, 1)).collect::<Vec<_>>();
    model.add_constraint(Constraint::Linear {
        terms: booleans.iter().map(|&variable| (1, variable)).collect(),
        relation: Relation::Le,
        rhs: 1,
    });
    model.add_constraint(Constraint::Linear { terms: vec![(1, starts[0]), (1, booleans[0])], relation: Relation::Le, rhs: 3 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(starts[2]) });
    let package = ModelPackage::new(model);
    let mut deferred = 0;
    let mut unary = 0;
    let mut cumulative = 0;
    for (incremental, lifted) in ARMS {
        for threads in [1, 2] {
            let mut options = request(8, incremental, lifted, threads);
            options.cp.semantic_minimization = true;
            options.cp.deferred_explanations = true;
            options.cp.explanation_metrics = true;
            options.cp.unary_edge_finding = true;
            options.cp.unary_edge_finding_metrics = true;
            options.cp.extended_nogoods = true;
            options.cp.cpip = true;
            options.proof = ProofRequest::Require;
            options.search_policy =
                SearchPolicy::new(vec![SearchPhase::new((3..8).chain(0..3).collect(), VariableSelector::Activity, ValueSelector::Max)]);
            let result = solve_model_silent(&package, &options).unwrap();
            result.validate_contract().unwrap();
            assert_eq!(result.status(), SolveStatus::Optimal);
            let values = assignment(&result);
            assert!(expected.contains(&values[..3]));
            assert!(values[3..].iter().all(|&value| value == 0 || value == 1));
            assert!(values[3..].iter().sum::<i32>() <= 1);
            assert!(values[0] + values[3] <= 3);
            assert_eq!(result.primal().unwrap().objectives(), &[i64::from(expected.iter().map(|values| values[2]).min().unwrap())]);
            assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Optimal));
            deferred += count(&result, "explanation_created_handles");
            unary += count(&result, "unary_directional_passes");
            cumulative += count(&result, "cumulative_propagation_calls");
        }
    }
    assert!(deferred > 0 && unary > 0 && cumulative > 0, "coexistence must exercise the native scheduling and deferred-reason paths");
}

#[test]
fn interruption_during_updates_can_resume_and_never_publishes_completion() {
    let case = Case::new("anchored", 4, 2);
    let expected = case.solutions();
    for (incremental, lifted) in ARMS {
        for stop_after in [0, 2, 8, 32] {
            let (mut solver, variables) = case.solver(incremental, lifted, true);
            solver.propagate().unwrap();
            let root = snapshot(&solver, &variables);
            let first = expected.first().unwrap()[0];
            let restricted = expected.iter().filter(|values| values[0] == first).cloned().collect();
            solver.store.push_level();
            solver.store.fix(variables[0], first).unwrap();
            let polls = AtomicUsize::new(0);
            solver.propagate_until(|| polls.fetch_add(1, Ordering::Relaxed) >= stop_after).unwrap();
            preserved(&solver, &variables, &restricted);
            solver.enqueue_all();
            solver.propagate().unwrap();
            preserved(&solver, &variables, &restricted);
            solver.store.pop_level();
            solver.enqueue_all();
            solver.propagate().unwrap();
            assert_eq!(snapshot(&solver, &variables), root);
            assert_eq!(enumerate(&mut solver, &variables), expected);
        }
        let (mut solver, variables) = case.solver(incremental, lifted, true);
        let stop = AtomicBool::new(false);
        let (best, _, complete) = optimize_var_assuming(&mut solver, &variables, &[], variables[0], true, &stop, |_, values| {
            assert!(expected.contains(values));
            stop.store(true, Ordering::Relaxed);
        });
        assert!(stop.load(Ordering::Relaxed));
        assert!(!complete);
        assert!(expected.contains(&best.unwrap().0));
        let mut options = request(case.n, incremental, lifted, 2);
        options.limits = SolveLimits { time: Some(Duration::ZERO), ..SolveLimits::default() };
        let result = solve_model_silent(&case.model(), &options).unwrap();
        result.validate_contract().unwrap();
        assert_eq!(result.status(), SolveStatus::Unknown);
        assert!(result.primal().is_none() && result.proof().is_none());
    }
}
