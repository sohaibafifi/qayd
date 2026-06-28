//! Transitional re-export. The structured domains moved to
//! [`crate::domains`]: list membership to [`crate::domains::list`] and the
//! interval handle to [`crate::domains::interval`]. Kept so existing
//! `crate::structured::` imports keep working until they migrate.

pub use crate::domains::interval::{IntervalDomain, IntervalEvent, IntervalPresence};
pub use crate::domains::list::{ListDomain, ListEvent};
