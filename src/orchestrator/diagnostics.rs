//! Canonical diagnostics over semantic integer models.
//!
//! Frontends identify variables by semantic arena index. Compilation, physical
//! selector management, cancellation, and explanation decoding stay behind the
//! orchestrator boundary.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::model::{CompiledCp, ModelPackage};
use crate::mus::{self, MusRel, MusResult};
use crate::search::{self, SearchControl};

use super::{SolveBudget, SolveError, SolveRequest, TerminationReason};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelMusResult {
    /// The selected semantic constraints are jointly satisfiable.
    ///
    /// A diagnostic never exposes the compiled solver's physical assignment.
    Satisfiable,
    Mus(Vec<usize>),
    Interrupted,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelMusEnumeration {
    pub muses: Vec<Vec<usize>>,
    pub msses: Vec<Vec<usize>>,
    pub complete: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MusAtomRelation {
    Eq,
    Ne,
    Ge,
    Le,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelMusAtom {
    pub variable: usize,
    pub relation: MusAtomRelation,
    pub value: i32,
}

pub type ModelMusExplanation = Vec<(usize, Vec<ModelMusAtom>)>;

fn with_budget<T>(
    request: &SolveRequest,
    external_stop: &AtomicBool,
    operation: impl FnOnce(&SolveBudget) -> Result<T, SolveError>,
) -> Result<T, SolveError> {
    request.validate()?;
    let budget = SolveBudget::new(request.limits.time);
    super::budget::apply_memory_limit(request.limits.memory_bytes, &budget);
    if external_stop.load(Ordering::Acquire) {
        budget.cancel_with(TerminationReason::ExternalCancellation);
    }
    let monitor_done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !monitor_done.load(Ordering::Acquire) && !budget.expired() {
                if external_stop.load(Ordering::Acquire) {
                    budget.cancel_with(TerminationReason::ExternalCancellation);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        });
        let result = operation(&budget);
        monitor_done.store(true, Ordering::Release);
        result
    })
}

fn compile(package: &ModelPackage, budget: &SolveBudget) -> Result<Option<CompiledCp>, SolveError> {
    if !package.validate_interruptible(budget.stop()).map_err(|errors| SolveError::Compile(errors.join("; ")))? {
        return Ok(None);
    }
    CompiledCp::compile_interruptible(&package.model, budget.stop()).map_err(|error| SolveError::Compile(error.reason))
}

fn map_variables(compiled: &CompiledCp, variables: &[usize], label: &str, stop: &AtomicBool) -> Result<Vec<crate::ids::VarId>, SolveError> {
    variables
        .iter()
        .map(|&index| {
            if stop.load(Ordering::Acquire) {
                return Err(SolveError::Interrupted(format!("{label} mapping was interrupted")));
            }
            compiled
                .int_variables()
                .get(index)
                .copied()
                .ok_or_else(|| SolveError::InvalidRequest(format!("{label} references unknown integer variable {index}")))
        })
        .collect()
}

fn semantic_index(compiled: &CompiledCp, variable: crate::ids::VarId, stop: &AtomicBool) -> Result<usize, SolveError> {
    for (index, candidate) in compiled.int_variables().iter().enumerate() {
        if stop.load(Ordering::Acquire) {
            return Err(SolveError::Interrupted("diagnostic explanation decoding was interrupted".to_string()));
        }
        if *candidate == variable {
            return Ok(index);
        }
    }
    Err(SolveError::InvalidResult(format!("diagnostic explanation references non-semantic variable {}", variable.index())))
}

pub fn count_model_solutions_with_external_stop(
    package: &ModelPackage,
    variables: &[usize],
    request: &SolveRequest,
    external_stop: &AtomicBool,
) -> Result<u64, SolveError> {
    with_budget(request, external_stop, |budget| {
        let Some(compiled) = compile(package, budget)? else {
            return Err(SolveError::Interrupted("solution-count compilation was interrupted".to_string()));
        };
        let variables = map_variables(&compiled, variables, "solution count", budget.stop())?;
        let mut solver = compiled.problem().solver.clone();
        let stats = search::solve_interruptible(&mut solver, &variables, |_| SearchControl::Continue, budget.stop());
        if budget.expired() {
            return Err(SolveError::Interrupted("solution counting stopped before exhaustively visiting the search space".to_string()));
        }
        Ok(stats.solutions)
    })
}

pub fn extract_model_mus_with_external_stop(
    package: &ModelPackage,
    variables: &[usize],
    selectors: &[usize],
    request: &SolveRequest,
    external_stop: &AtomicBool,
) -> Result<ModelMusResult, SolveError> {
    with_budget(request, external_stop, |budget| {
        let Some(compiled) = compile(package, budget)? else {
            return Ok(ModelMusResult::Interrupted);
        };
        let variables = map_variables(&compiled, variables, "MUS decision set", budget.stop())?;
        let selector_variables = map_variables(&compiled, selectors, "MUS selector set", budget.stop())?;
        let mut solver = compiled.problem().solver.clone();
        Ok(match mus::extract_mus(&mut solver, &variables, &selector_variables, budget.stop()) {
            MusResult::Sat(_) => ModelMusResult::Satisfiable,
            MusResult::Interrupted => ModelMusResult::Interrupted,
            MusResult::Mus(core) => ModelMusResult::Mus(
                core.into_iter()
                    .map(|variable| {
                        selector_variables
                            .iter()
                            .position(|candidate| *candidate == variable)
                            .map(|position| selectors[position])
                            .ok_or_else(|| SolveError::InvalidResult("MUS returned an unknown selector".to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        })
    })
}

pub fn enumerate_model_mus_with_external_stop(
    package: &ModelPackage,
    variables: &[usize],
    selectors: &[usize],
    limit: Option<usize>,
    request: &SolveRequest,
    external_stop: &AtomicBool,
) -> Result<ModelMusEnumeration, SolveError> {
    with_budget(request, external_stop, |budget| {
        let Some(compiled) = compile(package, budget)? else {
            return Ok(ModelMusEnumeration::default());
        };
        let variables = map_variables(&compiled, variables, "MUS decision set", budget.stop())?;
        let selector_variables = map_variables(&compiled, selectors, "MUS selector set", budget.stop())?;
        let mut solver = compiled.problem().solver.clone();
        let result = mus::enumerate_mus(&mut solver, &variables, &selector_variables, budget.stop(), limit);
        let decode = |sets: Vec<Vec<crate::ids::VarId>>| {
            sets.into_iter()
                .map(|set| {
                    set.into_iter()
                        .map(|variable| {
                            if budget.stop().load(Ordering::Acquire) {
                                return Err(SolveError::Interrupted("MUS enumeration decoding was interrupted".to_string()));
                            }
                            selector_variables
                                .iter()
                                .position(|candidate| *candidate == variable)
                                .map(|position| selectors[position])
                                .ok_or_else(|| SolveError::InvalidResult("MUS enumeration returned an unknown selector".to_string()))
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(ModelMusEnumeration { muses: decode(result.muses)?, msses: decode(result.msses)?, complete: result.complete })
    })
}

pub fn explain_model_mus_with_external_stop(
    package: &ModelPackage,
    variables: &[usize],
    mus_selectors: &[usize],
    request: &SolveRequest,
    external_stop: &AtomicBool,
) -> Result<Option<ModelMusExplanation>, SolveError> {
    with_budget(request, external_stop, |budget| {
        let Some(compiled) = compile(package, budget)? else {
            return Err(SolveError::Interrupted("MUS explanation compilation was interrupted".to_string()));
        };
        let variables = map_variables(&compiled, variables, "MUS decision set", budget.stop())?;
        let selectors = map_variables(&compiled, mus_selectors, "MUS selector set", budget.stop())?;
        let mut solver = compiled.problem().solver.clone();
        mus::explain_mus(&mut solver, &variables, &selectors, budget.stop())
            .map(|explanation| {
                explanation
                    .constraints
                    .into_iter()
                    .map(|(selector, atoms)| {
                        let selector = selectors
                            .iter()
                            .position(|candidate| *candidate == selector)
                            .map(|position| mus_selectors[position])
                            .ok_or_else(|| SolveError::InvalidResult("MUS explanation returned an unknown selector".to_string()))?;
                        let atoms = atoms
                            .into_iter()
                            .map(|atom| {
                                Ok(ModelMusAtom {
                                    variable: semantic_index(&compiled, atom.var, budget.stop())?,
                                    relation: match atom.rel {
                                        MusRel::Eq => MusAtomRelation::Eq,
                                        MusRel::Ne => MusAtomRelation::Ne,
                                        MusRel::Ge => MusAtomRelation::Ge,
                                        MusRel::Le => MusAtomRelation::Le,
                                    },
                                    value: atom.value,
                                })
                            })
                            .collect::<Result<Vec<_>, SolveError>>()?;
                        Ok((selector, atoms))
                    })
                    .collect::<Result<Vec<_>, SolveError>>()
            })
            .transpose()
    })
}
