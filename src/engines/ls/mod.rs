//! Local-search engines: incumbent-oriented heuristics over a defined model.
//!
//! `cop` is the min-conflicts search over integer variables (the `--ls` COP path); `lists` is the move-based search over list models.

pub(crate) mod cop;
pub mod lists;
