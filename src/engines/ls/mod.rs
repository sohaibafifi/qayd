//! Local-search engines: incumbent-oriented heuristics over a defined model.
//!
//! `cop` is the min-conflicts search over integer variables (the `--ls` COP path); `lists` is the move-based search over list models.

#[doc(hidden)]
#[allow(dead_code)]
pub(crate) mod cop;
pub(crate) mod disjunctive_schedule;
pub(crate) mod integer;
#[allow(dead_code)]
pub(crate) mod lists;
pub(crate) mod scenario_schedule;
