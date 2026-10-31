use qayd::constraints::intension::intension;
use qayd::constraints::linear::{linear, Relation};
use qayd::expr::{eq, mul, var};
use qayd::{minimize, Solver, VarId};

fn build_low_autocorrelation(sequence: Option<&[i32]>, n: usize) -> (Solver, Vec<VarId>, VarId) {
    let mut solver = Solver::new();
    let x: Vec<VarId> = match sequence {
        Some(values) => {
            assert_eq!(values.len(), n);
            values.iter().map(|&value| solver.new_var_set(&[value])).collect()
        }
        None => (0..n).map(|_| solver.new_var_set(&[-1, 1])).collect(),
    };
    let mut all = x.clone();
    let mut squares = Vec::new();

    for lag in 1..n {
        let mut products = Vec::new();
        for i in 0..(n - lag) {
            let product = solver.new_var_set(&[-1, 1]);
            intension(&mut solver, eq(var(product), mul(vec![var(x[i]), var(x[i + lag])])));
            products.push(product);
            all.push(product);
        }

        let width = (n - lag) as i32;
        let correlation = solver.new_var_range(-width, width);
        let mut coeffs = vec![1; products.len()];
        let mut vars = products;
        coeffs.push(-1);
        vars.push(correlation);
        linear(&mut solver, &coeffs, &vars, Relation::Eq, 0);
        all.push(correlation);

        let square = solver.new_var_range(-(width * width), width * width);
        intension(&mut solver, eq(var(square), mul(vec![var(correlation), var(correlation)])));
        squares.push(square);
        all.push(square);
    }

    let limit: i32 = (1..n).map(|lag| (n - lag).pow(2) as i32).sum();
    let objective = solver.new_var_range(-limit, limit);
    let mut coeffs = vec![1; squares.len()];
    coeffs.push(-1);
    squares.push(objective);
    linear(&mut solver, &coeffs, &squares, Relation::Eq, 0);
    all.push(objective);

    (solver, all, objective)
}

fn low_autocorrelation_energy(values: &[i32]) -> i32 {
    (1..values.len())
        .map(|lag| {
            let correlation: i32 = (0..(values.len() - lag)).map(|i| values[i] * values[i + lag]).sum();
            correlation * correlation
        })
        .sum()
}

#[test]
fn low_autocorrelation_minimization_matches_brute_force() {
    let n = 5usize;
    let (mut solver, all, objective) = build_low_autocorrelation(None, n);
    let (_, got) = minimize(&mut solver, &all, objective).expect("feasible");
    let want = (0..(1 << n))
        .map(|mask| {
            let values: Vec<i32> = (0..n).map(|i| if mask & (1 << i) == 0 { -1 } else { 1 }).collect();
            low_autocorrelation_energy(&values)
        })
        .min()
        .unwrap();
    assert_eq!(got, want);
}

#[test]
fn low_autocorrelation_20_known_witness_has_cost_26() {
    let sequence = [1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1];
    assert_eq!(low_autocorrelation_energy(&sequence), 26);

    let (mut solver, all, objective) = build_low_autocorrelation(Some(&sequence), sequence.len());
    let (_, got) = minimize(&mut solver, &all, objective).expect("feasible");
    assert_eq!(got, 26);
}
