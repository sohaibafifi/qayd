//! Local-search engines: incumbent-oriented heuristics over a defined model.
//!
//! `cop` is the min-conflicts search over integer variables (the `--ls` COP path); `lists` is the move-based search over list models.

#[doc(hidden)]
#[allow(dead_code)]
pub(crate) mod cop;
#[allow(dead_code)]
pub(crate) mod lists;
