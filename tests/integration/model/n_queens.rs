//! Phase 0 milestone: solve N-Queens via a hand-built binary model and check
//! the solution counts against the known sequence (OEIS A000170).

use qayd::constraints::primitives::not_equal_offset;
use qayd::{count_solutions, first_solution, Solver, VarId};

fn build_queens(solver: &mut Solver, n: i32) -> Vec<VarId> {
    let q: Vec<VarId> = (0..n).map(|_| solver.new_var_range(0, n - 1)).collect();
    for i in 0..n as usize {
        for j in (i + 1)..n as usize {
            let (di, dj) = (i as i32, j as i32);
            not_equal_offset(solver, q[i], q[j], 0);
            not_equal_offset(solver, q[i], q[j], di - dj);
            not_equal_offset(solver, q[i], q[j], dj - di);
        }
    }
    q
}

fn count(n: i32) -> u64 {
    let mut solver = Solver::new();
    let q = build_queens(&mut solver, n);
    count_solutions(&mut solver, &q)
}

#[test]
fn solution_counts_match_known_sequence() {
    // A000170: 1, 0, 0, 2, 10, 4, 40, 92, 352, ...
    let expected = [(1, 1), (2, 0), (3, 0), (4, 2), (5, 10), (6, 4), (7, 40), (8, 92), (9, 352)];
    for (n, want) in expected {
        assert_eq!(count(n), want, "{n}-queens solution count");
    }
}

#[test]
fn first_solution_is_valid() {
    let mut solver = Solver::new();
    let q = build_queens(&mut solver, 8);
    let sol = first_solution(&mut solver, &q).expect("8-queens is satisfiable");
    assert_eq!(sol.len(), 8);
    // All columns distinct and no two queens share a diagonal.
    for i in 0..8usize {
        assert!((0..8).contains(&sol[i]));
        for j in (i + 1)..8usize {
            assert_ne!(sol[i], sol[j], "same column");
            assert_ne!((sol[i] - sol[j]).abs(), (i as i32 - j as i32).abs(), "same diagonal");
        }
    }
}

#[test]
fn unsatisfiable_has_no_first_solution() {
    let mut solver = Solver::new();
    let q = build_queens(&mut solver, 3);
    assert_eq!(first_solution(&mut solver, &q), None);
}
