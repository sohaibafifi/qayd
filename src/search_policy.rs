//! Frontend-neutral exact-search policy declarations.
//!
//! A policy contains only semantic integer-variable identifiers. Compilation
//! maps them to the physical CP variables used by an executable solve plan.
//! The exact engine always appends its ordinary `Auto` completion phase, so a
//! policy can guide a prefix of the search without weakening completeness.

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

use crate::ids::VarId;

/// Variable selection used inside one ordered semantic search phase.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VariableSelector {
    /// Preserve the exact engine's current objective-aware dom/wdeg and
    /// activity policy.
    #[default]
    Auto,
    /// Select the first unfixed variable in the declared scope.
    InputOrder,
    /// Select a variable with the smallest current domain.
    FirstFail,
    /// Minimize current domain size divided by weighted degree.
    DomWdeg,
    /// Select the variable with the greatest learned-clause activity.
    Activity,
}

impl VariableSelector {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::InputOrder => "input-order",
            Self::FirstFail => "first-fail",
            Self::DomWdeg => "dom-wdeg",
            Self::Activity => "activity",
        }
    }
}

impl fmt::Display for VariableSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for VariableSelector {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "input-order" => Ok(Self::InputOrder),
            "first-fail" => Ok(Self::FirstFail),
            "dom-wdeg" => Ok(Self::DomWdeg),
            "activity" => Ok(Self::Activity),
            _ => Err(format!("unknown variable selector '{value}'; expected auto, input-order, first-fail, dom-wdeg, or activity")),
        }
    }
}

/// Value selection used for the equality decision `x = v` in one phase.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ValueSelector {
    /// Preserve saved phases, objective directions, seeded diversification and
    /// rephasing from the current exact engine.
    #[default]
    Auto,
    /// Select the smallest supported value.
    Min,
    /// Select the greatest supported value.
    Max,
    /// Select the supported value at or above the numeric domain midpoint.
    Median,
    /// Select a reproducible supported value from the request/worker seed.
    RandomSeeded,
    /// Prefer the current saved phase, including engine-produced incumbent or
    /// relaxation guidance, then fall back to `Auto` value selection.
    Hint,
}

impl ValueSelector {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Min => "min",
            Self::Max => "max",
            Self::Median => "median",
            Self::RandomSeeded => "random-seeded",
            Self::Hint => "hint",
        }
    }
}

impl fmt::Display for ValueSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ValueSelector {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "median" => Ok(Self::Median),
            "random-seeded" => Ok(Self::RandomSeeded),
            "hint" => Ok(Self::Hint),
            _ => Err(format!("unknown value selector '{value}'; expected auto, min, max, median, random-seeded, or hint")),
        }
    }
}

/// One ordered search phase over semantic integer variables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchPhase {
    scope: Vec<usize>,
    semantic_keys: Vec<usize>,
    variable: VariableSelector,
    value: ValueSelector,
}

impl SearchPhase {
    pub fn new(scope: Vec<usize>, variable: VariableSelector, value: ValueSelector) -> Self {
        let semantic_keys = scope.clone();
        Self { scope, semantic_keys, variable, value }
    }

    pub fn scope(&self) -> &[usize] {
        &self.scope
    }

    pub fn variable_selector(&self) -> VariableSelector {
        self.variable
    }

    pub fn value_selector(&self) -> ValueSelector {
        self.value
    }

    pub(crate) fn projected(scope: Vec<usize>, semantic_keys: Vec<usize>, variable: VariableSelector, value: ValueSelector) -> Self {
        debug_assert_eq!(scope.len(), semantic_keys.len());
        Self { scope, semantic_keys, variable, value }
    }

    pub(crate) fn semantic_keys(&self) -> &[usize] {
        &self.semantic_keys
    }
}

/// Ordered semantic phases followed by an implicit physical `Auto` fallback.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SearchPolicy {
    phases: Vec<SearchPhase>,
}

impl SearchPolicy {
    pub fn new(phases: Vec<SearchPhase>) -> Self {
        Self { phases }
    }

    pub fn phases(&self) -> &[SearchPhase] {
        &self.phases
    }

    /// Whether this policy is exactly the implicit engine `Auto` fallback.
    pub fn is_auto(&self) -> bool {
        self.phases.is_empty()
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        let mut seen = BTreeSet::new();
        for (phase_index, phase) in self.phases.iter().enumerate() {
            if phase.scope.is_empty() {
                return Err(format!("search policy phase {phase_index} has an empty variable scope"));
            }
            for &variable in &phase.scope {
                if !seen.insert(variable) {
                    return Err(format!("search policy variable {variable} appears in more than one phase or more than once in one phase"));
                }
            }
        }
        Ok(())
    }
}

impl From<Vec<SearchPhase>> for SearchPolicy {
    fn from(phases: Vec<SearchPhase>) -> Self {
        Self::new(phases)
    }
}

/// Physical phase embedded in a compiled exact CP solve plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CompiledSearchPhase {
    pub(crate) scope: Vec<VarId>,
    pub(crate) semantic_salts: Vec<u64>,
    pub(crate) variable: VariableSelector,
    pub(crate) value: ValueSelector,
}
