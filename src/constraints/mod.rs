//! The constraint catalogue: each constraint is a
//! [`Propagator`](crate::Propagator) plus a posting helper.

pub mod count;
pub mod flatten;
pub mod graph;
pub mod intension;
pub mod interval;
pub mod lex;
pub mod linear;
pub mod list;
pub mod primitives;
pub mod scheduling;
pub mod table;
