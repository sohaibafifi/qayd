use super::{EngineKind, SolveError};

/// Public, representation-free description of one compiled engine step.
/// Physical plans stay sealed inside the orchestrator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnginePlan {
    engine: EngineKind,
}

impl EnginePlan {
    pub fn new(engine: EngineKind) -> Self {
        Self { engine }
    }

    pub fn engine(&self) -> EngineKind {
        self.engine
    }
}

/// How component assignments are combined. The first implementation supports
/// disjoint variable arenas only; coupled master/subproblem plans require an
/// explicit future merge strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecompositionMerge {
    Disjoint,
}

/// Executable composition chosen by the orchestrator.
#[derive(Clone)]
pub enum ExecutablePlan {
    Single(EnginePlan),
    Sequential(Vec<ExecutablePlan>),
    Portfolio(Vec<ExecutablePlan>),
    Decomposed { components: Vec<ExecutablePlan>, merge: DecompositionMerge },
}

impl ExecutablePlan {
    pub fn validate(&self) -> Result<(), SolveError> {
        match self {
            Self::Single(_) => Ok(()),
            Self::Sequential(plans) | Self::Portfolio(plans) if plans.is_empty() => {
                Err(SolveError::Compile("composite solve plan is empty".to_string()))
            }
            Self::Sequential(plans) | Self::Portfolio(plans) => plans.iter().try_for_each(Self::validate),
            Self::Decomposed { components, .. } if components.is_empty() => {
                Err(SolveError::Compile("decomposition has no components".to_string()))
            }
            Self::Decomposed { components, .. } => components.iter().try_for_each(Self::validate),
        }
    }
}
