//! `qayd` — yet another constraint-programming solver.
//!
//! # Phase 0 — the kernel
//!
//! This phase provides the foundation everything else builds on:
//!
//! - [`ids`] — `VarId`, `PropId`, `ValId` newtypes.
//! - [`trail`] — the trailed state manager and `ReversibleInt`.
//! - [`domain`] — the trailed sparse-set integer domain.
//! - [`store`] — the [`Store`] (domains + trail + scheduling) and [`Solver`].
//! - [`propagator`] — the [`Propagator`] trait and the [`Inconsistency`] type.
//! - [`search`] — DFS with first-fail branching.
//! - [`oracle`] — a brute-force satisfaction oracle for testing.
//! - [`constraints`] — the constraint catalogue (binary disequality so far).
//!
//! ## Example: count the solutions of 4-Queens
//!
//! ```
//! use qayd::constraints::primitives::not_equal_offset;
//! use qayd::{count_solutions, Solver, VarId};
//!
//! let n = 4;
//! let mut solver = Solver::new();
//! let q: Vec<VarId> = (0..n).map(|_| solver.new_var_range(0, n - 1)).collect();
//! for i in 0..n as usize {
//!     for j in (i + 1)..n as usize {
//!         let (di, dj) = (i as i32, j as i32);
//!         not_equal_offset(&mut solver, q[i], q[j], 0);
//!         not_equal_offset(&mut solver, q[i], q[j], di - dj);
//!         not_equal_offset(&mut solver, q[i], q[j], dj - di);
//!     }
//! }
//! assert_eq!(count_solutions(&mut solver, &q), 2);
//! ```

pub mod constraints;
pub mod domain;
pub mod expr;
pub mod ids;
pub mod lcg;
pub mod oracle;
pub mod propagator;
pub mod search;
pub mod store;
pub mod trail;
pub mod xcsp;

pub use expr::Expr;
pub use ids::{PropId, ValId, VarId};
pub use propagator::{Event, Inconsistency, Propagator};
pub use search::{
    count_solutions, first_solution, maximize, minimize, optimize_with, solve, solve_interruptible,
    SearchControl, SolveStats,
};
pub use store::{Solver, Store};
pub use trail::{ReversibleInt, Trail};
