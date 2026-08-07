use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use qayd::constraints::linear::Relation;
use qayd::engines::ls::cop::{solve_ls, LocalSearchSpec, LsConfig, MAX_DOMAIN_VALUES};
use qayd::problem::{Objective, Problem};
use qayd::store::Solver;

/// A search variable whose domain is too large to materialise (> 4096 values)
/// and whose only feasible region sits far from its lower bound must still be
/// moved off that bound: LS finds a feasible incumbent by sampling the range.
#[test]
fn large_range_domain_finds_feasible_incumbent() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 5000);
    assert!(solver.store.size(x) > MAX_DOMAIN_VALUES, "test needs an unmaterialised domain");
    let problem = Problem { solver, search: vec![x], objective: Some(Objective::Var(true, x)) };

    let mut spec = LocalSearchSpec::default();
    spec.add_var(x);
    spec.add_linear(vec![1], vec![x], Relation::Ge, 4500);

    let stop = AtomicBool::new(false);
    let mut incumbent: Option<(i64, Vec<i32>)> = None;
    std::thread::scope(|scope| {
        // Bound the run: cap wall time, but exit promptly once search stops.
        scope.spawn(|| {
            for _ in 0..500 {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            stop.store(true, Ordering::Relaxed);
        });
        let _ = solve_ls(problem, spec, &stop, 1, LsConfig::default(), |value, solution, _| {
            if incumbent.is_none() {
                incumbent = Some((value, solution.to_vec()));
            }
            stop.store(true, Ordering::Relaxed);
        });
        stop.store(true, Ordering::Relaxed);
    });

    let (value, solution) = incumbent.expect("LS never found a feasible incumbent for a >4096 domain");
    assert!(solution[0] >= 4500, "incumbent {solution:?} violates x >= 4500");
    assert_eq!(value, i64::from(solution[0]));
}
