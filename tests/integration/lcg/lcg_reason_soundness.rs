use qayd::constraints::linear::{linear, Relation};
use qayd::{maximize, Solver, VarId};

/// Tight reasons must preserve a proven optimum. 18-item knapsack, optimum 122.
#[test]
fn knapsack_18_tight_reasons_stay_optimal() {
    let n = 18usize;
    let weights: Vec<i64> = (0..n).map(|i| 1 + (i as i64 * 7) % 13).collect();
    let values: Vec<i64> = (0..n).map(|i| 1 + (i as i64 * 11) % 17).collect();
    let capacity = weights.iter().sum::<i64>() / 2;

    let mut solver = Solver::new();
    let x: Vec<VarId> = (0..n).map(|_| solver.new_var_range(0, 1)).collect();
    linear(&mut solver, &weights, &x, Relation::Le, capacity);

    let mut coeffs = values.clone();
    coeffs.push(-1);
    let mut vars = x;
    let objective = solver.new_var_range(0, values.iter().sum::<i64>() as i32);
    vars.push(objective);
    linear(&mut solver, &coeffs, &vars, Relation::Eq, 0);

    let (_, best) = maximize(&mut solver, &vars, objective).expect("knapsack is satisfiable");
    assert_eq!(best, 122);
}
