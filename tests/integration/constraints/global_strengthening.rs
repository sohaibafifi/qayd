use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;

use qayd::constraints::interval;
use qayd::constraints::lex::lex;
use qayd::constraints::primitives::all_different_except;
use qayd::constraints::scheduling::{bin_loads, bin_packing, cumulative_var};
use qayd::{optimize_with, solve_search, SearchControl, Solver, VarId};

static NEVER_STOP: AtomicBool = AtomicBool::new(false);

fn enumerate(solver: &mut Solver, variables: &[VarId]) -> BTreeSet<Vec<i32>> {
    let mut solutions = BTreeSet::new();
    solve_search(solver, variables, |solver| {
        solutions.insert(variables.iter().map(|&variable| solver.store.value(variable)).collect());
        SearchControl::Continue
    });
    solutions
}

fn assignments(domains: &[Vec<i32>], accept: &mut impl FnMut(&[i32]) -> bool) -> BTreeSet<Vec<i32>> {
    fn visit(
        domains: &[Vec<i32>],
        index: usize,
        assignment: &mut Vec<i32>,
        accept: &mut impl FnMut(&[i32]) -> bool,
        solutions: &mut BTreeSet<Vec<i32>>,
    ) {
        if index == domains.len() {
            if accept(assignment) {
                solutions.insert(assignment.clone());
            }
            return;
        }
        for &value in &domains[index] {
            assignment.push(value);
            visit(domains, index + 1, assignment, accept, solutions);
            assignment.pop();
        }
    }

    let mut solutions = BTreeSet::new();
    visit(domains, 0, &mut Vec::new(), accept, &mut solutions);
    solutions
}

fn next(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed >> 32
}

fn random_domain(seed: &mut u64, values: &[i32]) -> Vec<i32> {
    loop {
        let domain = values.iter().copied().filter(|_| next(seed) & 1 == 1).collect::<Vec<_>>();
        if !domain.is_empty() {
            return domain;
        }
    }
}

#[test]
fn all_different_except_matches_holey_oracle() {
    let mut seed = 0x51a9_3c72_88f0_21d4;
    for _ in 0..96 {
        let domains = (0..4).map(|_| random_domain(&mut seed, &[-1, 0, 1, 2])).collect::<Vec<_>>();
        let except = if next(&mut seed) & 1 == 0 { vec![0] } else { vec![-1, 1] };
        let mut solver = Solver::new();
        let variables = domains.iter().map(|domain| solver.new_var_set(domain)).collect::<Vec<_>>();
        all_different_except(&mut solver, &variables, &except);
        let got = enumerate(&mut solver, &variables);
        let want = assignments(&domains, &mut |assignment| {
            (0..assignment.len()).all(|left| {
                ((left + 1)..assignment.len()).all(|right| assignment[left] != assignment[right] || except.contains(&assignment[left]))
            })
        });
        assert_eq!(got, want, "domains={domains:?} except={except:?}");

        let mut learning = Solver::new();
        let variables = domains.iter().map(|domain| learning.new_var_set(domain)).collect::<Vec<_>>();
        all_different_except(&mut learning, &variables, &except);
        let (best, _) = optimize_with(&mut learning, &variables, variables[0], true, &NEVER_STOP, |_| {});
        assert_eq!(best.map(|(_, value)| value), want.iter().map(|assignment| assignment[0]).min());
    }
}

#[test]
fn all_different_except_filters_shared_values_but_keeps_exempt_repetitions() {
    let mut solver = Solver::new();
    let fixed = solver.new_var_set(&[1]);
    let optional = solver.new_var_set(&[0, 1]);
    let tail = solver.new_var_set(&[0, 1, 2]);
    all_different_except(&mut solver, &[fixed, optional, tail], &[0]);
    solver.enqueue_all();
    solver.propagate().unwrap();
    assert_eq!(solver.store.values(optional).collect::<Vec<_>>(), vec![0]);
    assert_eq!(solver.store.values(tail).collect::<Vec<_>>(), vec![0, 2]);
}

#[test]
fn lex_automaton_matches_holey_oracle() {
    let mut seed = 0x24e1_09b7_c3a8_5d6f;
    for strict in [false, true] {
        for _ in 0..80 {
            let x_domains = (0..3).map(|_| random_domain(&mut seed, &[-1, 0, 1, 2])).collect::<Vec<_>>();
            let y_domains = (0..3).map(|_| random_domain(&mut seed, &[-1, 0, 1, 2])).collect::<Vec<_>>();
            let mut solver = Solver::new();
            let x = x_domains.iter().map(|domain| solver.new_var_set(domain)).collect::<Vec<_>>();
            let y = y_domains.iter().map(|domain| solver.new_var_set(domain)).collect::<Vec<_>>();
            lex(&mut solver, &x, &y, strict);
            let mut variables = x.clone();
            variables.extend(&y);
            let got = enumerate(&mut solver, &variables);
            let mut domains = x_domains.clone();
            domains.extend(y_domains.clone());
            let want = assignments(&domains, &mut |assignment| {
                let (left, right) = assignment.split_at(3);
                if strict {
                    left < right
                } else {
                    left <= right
                }
            });
            assert_eq!(got, want, "strict={strict} x={x_domains:?} y={y_domains:?}");
        }
    }
}

#[test]
fn bin_packing_domain_subsets_match_oracle() {
    let mut seed = 0x19d2_70ab_61ce_f403;
    for _ in 0..72 {
        let domains = (0..4).map(|_| random_domain(&mut seed, &[0, 1, 2])).collect::<Vec<_>>();
        let sizes = (0..4).map(|_| 1 + (next(&mut seed) % 3) as i64).collect::<Vec<_>>();
        let capacities = (0..3).map(|_| 2 + (next(&mut seed) % 4) as i64).collect::<Vec<_>>();
        let mut solver = Solver::new();
        let items = domains.iter().map(|domain| solver.new_var_set(domain)).collect::<Vec<_>>();
        bin_packing(&mut solver, &items, &sizes, &capacities);
        let got = enumerate(&mut solver, &items);
        let want = assignments(&domains, &mut |assignment| {
            let mut loads = vec![0i64; capacities.len()];
            for (index, &bin) in assignment.iter().enumerate() {
                loads[bin as usize] += sizes[index];
            }
            loads.iter().zip(&capacities).all(|(load, capacity)| load <= capacity)
        });
        assert_eq!(got, want, "domains={domains:?} sizes={sizes:?} capacities={capacities:?}");
    }
}

#[test]
fn bin_packing_saturated_subset_prunes_other_items() {
    let mut solver = Solver::new();
    let mut items = (0..4).map(|_| solver.new_var_set(&[0, 1])).collect::<Vec<_>>();
    items.push(solver.new_var_set(&[0, 1, 2]));
    bin_packing(&mut solver, &items, &[2, 2, 2, 2, 3], &[4, 4, 3]);
    solver.enqueue_all();
    solver.propagate().unwrap();
    assert_eq!(solver.store.values(items[4]).collect::<Vec<_>>(), vec![2]);
}

#[test]
fn bin_loads_matches_holey_oracle_without_indicators() {
    let mut seed = 0x7b10_5ed4_92ca_31f8;
    for _ in 0..48 {
        let domains = (0..4).map(|_| random_domain(&mut seed, &[0, 1, 2])).collect::<Vec<_>>();
        let sizes = (0..4).map(|_| 1 + (next(&mut seed) % 3) as i64).collect::<Vec<_>>();
        let total = sizes.iter().sum::<i64>() as i32;
        let mut solver = Solver::new();
        let items = domains.iter().map(|domain| solver.new_var_set(domain)).collect::<Vec<_>>();
        let loads = (0..3).map(|_| solver.new_var_range(0, total)).collect::<Vec<_>>();
        bin_loads(&mut solver, &items, &sizes, &loads);
        let mut variables = items.clone();
        variables.extend(&loads);
        let got = enumerate(&mut solver, &variables);

        let item_assignments = assignments(&domains, &mut |_| true);
        let want = item_assignments
            .into_iter()
            .map(|mut assignment| {
                let mut exact = vec![0i32; 3];
                for (index, &bin) in assignment.iter().enumerate() {
                    exact[bin as usize] += sizes[index] as i32;
                }
                assignment.extend(exact);
                assignment
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(got, want, "domains={domains:?} sizes={sizes:?}");
    }
}

#[test]
fn bin_loads_fixed_total_forces_required_items() {
    let mut solver = Solver::new();
    let first = solver.new_var_set(&[0, 1]);
    let second = solver.new_var_set(&[0, 1]);
    let loads = [solver.new_var_set(&[5]), solver.new_var_set(&[0])];
    bin_loads(&mut solver, &[first, second], &[2, 3], &loads);
    solver.enqueue_all();
    solver.propagate().unwrap();
    assert_eq!(solver.store.value(first), 0);
    assert_eq!(solver.store.value(second), 0);
}

#[test]
fn bin_loads_rejects_a_committed_load_above_i32() {
    let mut solver = Solver::new();
    let items = [solver.new_var_set(&[0]), solver.new_var_set(&[0])];
    let loads = [solver.new_var_range(0, i32::MAX), solver.new_var_range(0, i32::MAX)];
    bin_loads(&mut solver, &items, &[i64::from(i32::MAX), i64::from(i32::MAX)], &loads);
    solver.enqueue_all();
    assert!(solver.propagate().is_err());
}

#[test]
fn cumulative_profile_memory_is_independent_of_numeric_horizon() {
    let mut solver = Solver::new();
    let first = solver.store.new_interval(0, 1_000_000_000, 1);
    let second = solver.store.new_interval(0, 1_000_000_000, 1);
    interval::cumulative(&mut solver, &[first, second], &[1, 1], 2);
    solver.enqueue_all();
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_start_min(first), 0);
    assert_eq!(solver.store.interval_start_max(second), 1_000_000_000);

    let mut variable = Solver::new();
    let starts = [variable.new_var_range(0, 1_000_000_000), variable.new_var_range(0, 1_000_000_000)];
    let durations = [variable.new_var_set(&[1]), variable.new_var_set(&[1])];
    let heights = [variable.new_var_set(&[1]), variable.new_var_set(&[1])];
    let capacity = variable.new_var_set(&[2]);
    cumulative_var(&mut variable, &starts, &durations, &heights, capacity);
    variable.enqueue_all();
    variable.propagate().unwrap();
    assert_eq!(variable.store.max(starts[1]), 1_000_000_000);
}
