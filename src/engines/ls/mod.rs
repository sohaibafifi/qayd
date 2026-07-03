//! Local-search engines: incumbent-oriented heuristics over a defined model.
//!
//! `cop` is the min-conflicts search over integer variables (the `--ls` COP path); `lists` is the move-based search over list models.

#[doc(hidden)]
pub mod cop;
pub mod lists;
