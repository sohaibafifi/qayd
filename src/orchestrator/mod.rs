//! Frontend-neutral solve control plane.
//!
//! Engines own algorithms. This module owns requests, budgets, executable plans,
//! cross-engine events, verification boundaries, and final result semantics.

mod budget;
mod collection;
mod cp;
#[cfg(feature = "python")]
mod diagnostics;
mod executor;
mod finalize;
mod integer_search;
mod plan;
mod sat;
mod session;
mod solve;
mod types;
mod verify;

pub use budget::*;
pub(crate) use collection::*;
pub(crate) use cp::*;
#[cfg(feature = "python")]
pub use diagnostics::*;
pub use executor::*;
pub(crate) use finalize::*;
pub use plan::*;
pub(crate) use sat::*;
pub use session::*;
pub use solve::*;
pub use types::*;
pub use verify::*;
