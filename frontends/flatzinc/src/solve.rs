//! Search adapter and MiniZinc FlatZinc protocol renderer.
//!
//! Parsing builds a canonical `ModelPackage`; this module submits it to the
//! frontend-neutral orchestrator, projects the verified result back onto
//! annotated FlatZinc outputs, and renders the protocol markers.

use std::io::{self, Write};
use std::time::Duration;

use qayd::model::IntVarRef;
use qayd::orchestrator::{EventControl, EventSink, SolveError, SolveEvent, SolveLimits, SolveMode, SolveRequest, SolveResult, SolveStatus};

use crate::model::{Model, Output};

/// Run-time options from the command line.
pub(crate) struct Options {
    /// Stream improving objectives and print end-of-search statistics (as `%` comments).
    pub(crate) verbose: bool,
    /// Wall-clock limit shared by every orchestrated solve phase.
    pub(crate) time_limit: Option<Duration>,
}

/// FlatZinc-specific output structure retained beside the canonical package.
struct OutputProjection {
    outputs: Vec<Output>,
}

impl OutputProjection {
    /// Print annotated output items from a verified assignment.
    fn print_solution(&self, result: &SolveResult) -> Result<(), SolveError> {
        let candidate =
            result.primal().ok_or_else(|| SolveError::InvalidResult("FlatZinc solution status has no primal assignment".to_string()))?;
        let value = |variable: IntVarRef| {
            let value = candidate
                .assignment()
                .integers
                .get(variable.0)
                .copied()
                .flatten()
                .ok_or_else(|| SolveError::InvalidResult(format!("FlatZinc output variable {} is undefined", variable.0)))?;
            i32::try_from(value).map_err(|_| SolveError::InvalidResult("FlatZinc output value does not fit in an i32".to_string()))
        };
        let format_value = |variable: IntVarRef, is_bool: bool| -> Result<String, SolveError> {
            let value = value(variable)?;
            if is_bool {
                if value != 0 {
                    Ok("true".to_string())
                } else {
                    Ok("false".to_string())
                }
            } else {
                Ok(value.to_string())
            }
        };

        for output in &self.outputs {
            match output {
                Output::Var { name, var, is_bool } => println!("{name} = {};", format_value(*var, *is_bool)?),
                Output::Array { name, dims, vars, is_bool } => {
                    let ranges = dims.iter().map(|(lo, hi)| format!("{lo}..{hi}")).collect::<Vec<_>>().join(", ");
                    let values = vars.iter().map(|&var| format_value(var, *is_bool)).collect::<Result<Vec<_>, _>>()?.join(", ");
                    println!("{name} = array{}d({ranges}, [{values}]);", dims.len());
                }
            }
        }
        Ok(())
    }
}

/// FlatZinc event rendering. The compatibility adapter currently publishes its
/// final candidate, while future progress events can be rendered without
/// putting an optimization loop back into this frontend.
struct FlatZincEvents {
    verbose: bool,
    last_objective: Option<i64>,
}

impl FlatZincEvents {
    fn publish_objective(&mut self, objectives: &[i64]) {
        let Some(&objective) = objectives.first() else {
            return;
        };
        if self.verbose && self.last_objective != Some(objective) {
            println!("% o {objective}");
            let _ = io::stdout().flush();
        }
        self.last_objective = Some(objective);
    }
}

impl EventSink for FlatZincEvents {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        match event {
            SolveEvent::Candidate(candidate) => self.publish_objective(candidate.objectives()),
            SolveEvent::Progress { objectives, .. } => self.publish_objective(&objectives),
            SolveEvent::Bound(_) | SolveEvent::Proof(_) | SolveEvent::Finished(_) => {}
        }
        Ok(EventControl::Continue)
    }
}

/// Solve `model` through the shared orchestrator and print MiniZinc protocol output.
pub(crate) fn solve(model: Model, opts: &Options) -> Result<(), SolveError> {
    let request = SolveRequest {
        mode: SolveMode::Exact,
        limits: SolveLimits { time: opts.time_limit, ..SolveLimits::default() },
        ..SolveRequest::default()
    };
    let (package, outputs) = model.into_package();
    let projection = OutputProjection { outputs };
    let mut events = FlatZincEvents { verbose: opts.verbose, last_objective: None };
    let result = qayd::solve(&package, &request, &mut events)?;

    render_result(&projection, &result)?;
    if opts.verbose {
        print_stats(&result);
    }
    Ok(())
}

/// Render an orchestrator-owned final status using MiniZinc protocol markers.
fn render_result(projection: &OutputProjection, result: &SolveResult) -> Result<(), SolveError> {
    if result.primal().is_some() {
        projection.print_solution(result)?;
        println!("----------");
        if result.status() == SolveStatus::Optimal {
            println!("==========");
        }
    } else {
        let marker = if result.status() == SolveStatus::Unsatisfiable { "=====UNSATISFIABLE=====" } else { "=====UNKNOWN=====" };
        println!("{marker}");
    }
    Ok(())
}

/// Print a `%` statistics comment line aggregated from all engine reports.
fn print_stats(result: &SolveResult) {
    let elapsed = result.elapsed();
    let stats = result.aggregate_search_stats();
    let nodes = stats.nodes;
    let failures = stats.failures;
    let solutions = stats.solutions;
    let learned_lits = stats.learned_lits;
    println!("% time={:.3}s nodes={nodes} failures={failures} solutions={solutions} learned_lits={learned_lits}", elapsed.as_secs_f64(),);
}
