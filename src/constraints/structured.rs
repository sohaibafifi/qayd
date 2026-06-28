//! Transitional re-export. Structured propagators split by domain:
//! list propagators to [`crate::constraints::list`], interval/scheduling
//! propagators to [`crate::constraints::interval`]. Kept so existing
//! `constraints::structured::` imports keep working until they migrate.

pub use super::interval::*;
pub use super::list::*;
