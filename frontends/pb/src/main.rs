//! `qayd-pb`: an OPB (pseudo-Boolean) front-end for `qayd`.
//!
//! Reads a linear OPB instance (PB Competition format), builds a 0/1 model on
//! the `qayd` core, and prints results in the competition output format:
//!
//! ```text
//! s SATISFIABLE | UNSATISFIABLE | OPTIMUM FOUND | UNKNOWN | UNSUPPORTED
//! v x1 -x2 x3 ...          (model, satisfiable/optimum only)
//! o <objective value>      (each incumbent, flushed immediately)
//! ```
//!
//! Current scope: linear PBS/PBO without certificates. Non-linear product
//! terms are reported `s UNSUPPORTED`.

mod opb;

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use opb::{Objective, Op, Opb};
use qayd::model::{Constraint, IntExpr, IntVarRef, Model, ModelObject, ModelPackage, Objective as SemanticObjective, Relation};
use qayd::orchestrator::{
    solve_model_with_stop, EventControl, EventSink, SolveError, SolveEvent, SolveLimits, SolveMode, SolveRequest, SolveResult, SolveStatus,
};

struct Options {
    verbose: bool,
    time_limit: Option<Duration>,
    memory_limit_mb: Option<u64>,
}

fn main() {
    let (opts, path) = parse_args();
    let input = read_input(path.as_deref());

    let instance = opb::parse(&input).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });

    let stop = Arc::new(AtomicBool::new(false));
    install_signal_handler(&stop, opts.verbose);

    let status = match run(&instance, Arc::clone(&stop), &opts) {
        Ok(status) => status,
        Err(FrontendError::Unsupported) => {
            println!("s UNSUPPORTED");
            std::process::exit(0);
        }
        Err(FrontendError::Solve(error)) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    };
    std::process::exit(competition_exit_code(status));
}

enum FrontendError {
    Unsupported,
    Solve(SolveError),
}

impl From<SolveError> for FrontendError {
    fn from(error: SolveError) -> Self {
        Self::Solve(error)
    }
}

struct LoweredPackage {
    package: ModelPackage,
}

/// Build the semantic model, delegate execution, and render the competition
/// protocol. Status and proof semantics come exclusively from the orchestrator.
fn run(instance: &Opb, stop: Arc<AtomicBool>, options: &Options) -> Result<SolveStatus, FrontendError> {
    if instance.nonlinear {
        return Err(FrontendError::Unsupported);
    }

    let Some(lowered) = lower_problem(instance) else {
        return Err(FrontendError::Unsupported);
    };
    let request = SolveRequest {
        mode: SolveMode::Exact,
        limits: SolveLimits {
            time: options.time_limit,
            memory_bytes: options.memory_limit_mb.map(|mb| mb.saturating_mul(1024).saturating_mul(1024)),
            ..SolveLimits::default()
        },
        ..SolveRequest::default()
    };
    let mut sink = CompetitionSink::new();
    let result = solve_model_with_stop(&lowered.package, &request, stop, &mut sink)?;
    if options.verbose {
        let elapsed = result.elapsed();
        eprintln!("c time={:.3}s vars={} constraints={}", elapsed.as_secs_f64(), instance.n_vars, instance.constraints.len());
    }
    render_result(&result, instance.n_vars, &mut sink)?;
    Ok(result.status())
}

fn lower_problem(instance: &Opb) -> Option<LoweredPackage> {
    let mut model = Model::new();
    // 0/1 variable for each OPB index; `vars[i]` is OPB variable `x(i+1)`.
    let vars: Vec<IntVarRef> = (0..instance.n_vars).map(|_| model.bool_var()).collect();

    for c in &instance.constraints {
        let (terms, rhs) = normalize(&c.terms, c.rhs, &vars);
        let rel = match c.op {
            Op::Ge => Relation::Ge,
            Op::Le => Relation::Le,
            Op::Eq => Relation::Eq,
            Op::Gt => Relation::Gt,
            Op::Lt => Relation::Lt,
        };
        model.add_constraint(Constraint::Linear { terms, relation: rel, rhs });
    }

    let objective = match &instance.objective {
        None => None,
        Some(objective) => Some(lower_objective(&vars, objective)?),
    };
    if let Some(objective) = objective {
        model.add_objective(objective);
    }
    let mut package = ModelPackage::new(model);
    for (index, variable) in vars.into_iter().enumerate() {
        let object = ModelObject::IntVar(variable);
        let name = format!("x{}", index + 1);
        package.metadata.names.insert(object, name.clone());
        package.metadata.frontend_ids.insert(("opb".to_string(), name), object);
        package.metadata.outputs.push(object);
    }
    Some(LoweredPackage { package })
}

/// Collapse terms to one coefficient per variable and fold `~x` literals and the
/// resulting constant into the right-hand side.
///
/// `c·~x = c·(1 - x) = c - c·x`, so a negated term contributes `-c` to `x` and
/// `c` to the LHS constant, which moves to the rhs as `-c`.
fn normalize(terms: &[opb::Term], rhs: i64, vars: &[IntVarRef]) -> (Vec<(i64, IntVarRef)>, i64) {
    let mut coeff_by_var = vec![0i64; vars.len()];
    let mut constant = 0i64; // LHS constant to move to the rhs
    for t in terms {
        let idx = t.var - 1;
        if t.negated {
            coeff_by_var[idx] -= t.coeff;
            constant += t.coeff;
        } else {
            coeff_by_var[idx] += t.coeff;
        }
    }
    let mut normalized = Vec::new();
    for (idx, &c) in coeff_by_var.iter().enumerate() {
        if c != 0 {
            normalized.push((c, vars[idx]));
        }
    }
    (normalized, rhs - constant)
}

fn lower_objective(vars: &[IntVarRef], obj: &Objective) -> Option<SemanticObjective> {
    // Objective in x-form (coefficients on `x`). `normalize` folds the `~x`
    // constant into its third return as `0 - folded`, so the additive constant
    // of the true objective is its negation.
    let (terms, neg_constant) = normalize(&obj.terms, 0, vars);
    let constant = neg_constant.checked_neg()?;

    let mut expressions = terms
        .into_iter()
        .map(|(coefficient, variable)| IntExpr::Mul(vec![IntExpr::Constant(coefficient), IntExpr::Variable(variable)]))
        .collect::<Vec<_>>();
    if constant != 0 {
        expressions.push(IntExpr::Constant(constant));
    }
    let expr = if expressions.is_empty() { IntExpr::Constant(0) } else { IntExpr::Add(expressions) };
    Some(SemanticObjective::IntExpr { minimize: !obj.maximize, expr })
}

struct CompetitionSink {
    last_objective: Option<i64>,
}

impl CompetitionSink {
    fn new() -> Self {
        Self { last_objective: None }
    }

    fn publish_objective(&mut self, value: i64) -> Result<(), SolveError> {
        if self.last_objective != Some(value) {
            println!("o {value}");
            io::stdout().flush().map_err(|error| SolveError::Engine(error.to_string()))?;
            self.last_objective = Some(value);
        }
        Ok(())
    }
}

impl EventSink for CompetitionSink {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        match event {
            SolveEvent::Candidate(candidate) => {
                if let Some(&objective) = candidate.objectives().first() {
                    self.publish_objective(objective)?;
                }
            }
            SolveEvent::Progress { objectives, .. } => {
                if let Some(&objective) = objectives.first() {
                    self.publish_objective(objective)?;
                }
            }
            SolveEvent::Bound(_) | SolveEvent::Proof(_) | SolveEvent::Finished(_) => {}
        }
        Ok(EventControl::Continue)
    }
}

fn render_result(result: &SolveResult, n_vars: usize, sink: &mut CompetitionSink) -> Result<(), SolveError> {
    if let Some(raw) = result.primal().and_then(|candidate| candidate.objectives().first()).copied() {
        sink.publish_objective(raw)?;
    }
    match result.status() {
        SolveStatus::Optimal => {
            println!("s OPTIMUM FOUND");
            print_result_model(result, n_vars)?;
        }
        SolveStatus::Satisfiable => {
            println!("s SATISFIABLE");
            print_result_model(result, n_vars)?;
        }
        SolveStatus::Unsatisfiable => println!("s UNSATISFIABLE"),
        SolveStatus::Unknown => println!("s UNKNOWN"),
        SolveStatus::Unsupported => println!("s UNSUPPORTED"),
    }
    Ok(())
}

fn print_result_model(result: &SolveResult, n_vars: usize) -> Result<(), SolveError> {
    let candidate = result.primal().ok_or_else(|| SolveError::InvalidResult("SAT/OPTIMUM result has no verified PB model".to_string()))?;
    if candidate.assignment().integers.len() != n_vars {
        return Err(SolveError::InvalidResult(format!(
            "verified PB model has {} variables, expected {n_vars}",
            candidate.assignment().integers.len()
        )));
    }
    let model = candidate
        .assignment()
        .integers
        .iter()
        .map(|value| match value {
            Some(0) => Ok(0),
            Some(1) => Ok(1),
            Some(value) => Err(SolveError::InvalidResult(format!("verified PB variable has non-Boolean value {value}"))),
            None => Err(SolveError::InvalidResult("verified PB model contains an unassigned variable".to_string())),
        })
        .collect::<Result<Vec<_>, _>>()?;
    print_model(&model);
    Ok(())
}

/// Print the `v` line: `x<i>` when true, `-x<i>` when false (1-based).
fn print_model(model: &[i32]) {
    print!("v");
    for (idx, &value) in model.iter().enumerate() {
        let n = idx + 1;
        if value != 0 {
            print!(" x{n}");
        } else {
            print!(" -x{n}");
        }
    }
    println!();
    let _ = io::stdout().flush();
}

fn read_input(path: Option<&str>) -> String {
    match path {
        Some(path) => std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("error: cannot read {path}: {e}");
            std::process::exit(1);
        }),
        None => {
            let mut input = String::new();
            io::Read::read_to_string(&mut io::stdin(), &mut input).unwrap_or_else(|e| {
                eprintln!("error: cannot read stdin: {e}");
                std::process::exit(1);
            });
            input
        }
    }
}

fn parse_args() -> (Options, Option<String>) {
    let mut opts = Options { verbose: false, time_limit: None, memory_limit_mb: None };
    let mut path = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-v" | "--verbose" => opts.verbose = true,
            "-t" | "--time" => {
                let secs = args.next().and_then(|s| s.parse::<u64>().ok()).unwrap_or_else(|| {
                    eprintln!("error: -t/--time expects seconds");
                    std::process::exit(1);
                });
                opts.time_limit = Some(Duration::from_secs(secs));
            }
            "--memory-mb" => {
                let mb = args.next().and_then(|s| s.parse::<u64>().ok()).filter(|&mb| mb > 0).unwrap_or_else(|| {
                    eprintln!("error: --memory-mb expects a positive integer");
                    std::process::exit(1);
                });
                opts.memory_limit_mb = Some(mb);
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other if other.starts_with('-') => {
                eprintln!("error: unknown option `{other}`");
                print_usage();
                std::process::exit(1);
            }
            other => {
                if path.is_some() {
                    eprintln!("error: more than one input file given");
                    std::process::exit(1);
                }
                path = Some(other.to_string());
            }
        }
    }
    (opts, path)
}

fn print_usage() {
    eprintln!(
        "usage: qayd-pb [options] [instance.opb]\n\
         \n\
         options:\n\
         \x20 -v, --verbose       print statistics as comment lines on stderr\n\
         \x20 -t, --time <sec>    stop after <sec> seconds\n\
         \x20 --memory-mb <mb>    set an address-space memory limit on Linux\n\
         \x20 -h, --help          show this help\n\
         \n\
         Reads a linear OPB instance from a file, or stdin when no file is given.\n\
         Prints PB Competition output (s/v/o lines)."
    );
}

fn install_signal_handler(stop: &Arc<AtomicBool>, verbose: bool) {
    let flag = Arc::clone(stop);
    if let Err(err) = ctrlc::set_handler(move || {
        flag.store(true, Ordering::Release);
    }) {
        if verbose {
            eprintln!("c signal_handler=unavailable reason={err}");
        }
    }
}

fn competition_exit_code(status: SolveStatus) -> i32 {
    // Mirrors the PB Competition convention: 10 SAT, 20 UNSAT, 30 OPTIMUM,
    // 0 otherwise.
    match status {
        SolveStatus::Satisfiable => 10,
        SolveStatus::Unsatisfiable => 20,
        SolveStatus::Optimal => 30,
        SolveStatus::Unknown | SolveStatus::Unsupported => 0,
    }
}
