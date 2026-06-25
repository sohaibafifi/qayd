//! `qayd`: a constraint-programming solver.
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

pub mod collection;
pub mod constraints;
pub mod domain;
pub mod expr;
pub mod ids;
pub mod lcg;
mod lns;
mod ls;
mod parallel;
mod problem;
pub mod propagator;
#[cfg(feature = "python")]
mod python;
pub mod search;
pub mod store;
pub mod trail;
pub mod xcsp;

pub use expr::Expr;
pub use ids::{PropId, VarId};
// The collection (list/lambda) IR lives under `qayd::collection::*`; it is not
// re-exported here because its `Expr` would clash with the intension `Expr`.
pub use propagator::{Event, Inconsistency, Propagator};
pub use search::{
    count_solutions, first_solution, maximize, minimize, optimize_with, solve, solve_interruptible, SearchControl, SolveStats,
};
pub use store::{Solver, Store};
pub use trail::{ReversibleInt, Trail};

/// SplitMix64 finalizer: scrambles a counter/seed into a well-distributed
/// `u64`. Shared so every seed-derived stream stays bit-identical across
/// modules - determinism depends on it.
pub(crate) fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}
