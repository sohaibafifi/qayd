//! `qayd-sat`: a minimal DIMACS CNF front-end for `qayd`.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use qayd::model::{BoolLiteral, Constraint, Model, ModelObject, ModelPackage};
use qayd::orchestrator::{
    IgnoreEvents, SatBackendMode, SatControls, SatPreprocess, SolveError, SolveLimits, SolveMode, SolveRequest, SolveResult, SolveStatus,
};
use qayd_sat::{assignment_satisfies, parse_dimacs};

struct Options {
    verbose: bool,
    time_limit: Option<Duration>,
    memory_limit_mb: Option<u64>,
    competition: bool,
    backend: SatBackendMode,
    seed: u64,
    preprocess: SatPreprocess,
    proof: Option<String>,
}

fn main() {
    let (opts, path) = parse_args();
    let input = match path {
        Some(path) => std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("error: cannot read {path}: {e}");
            std::process::exit(1);
        }),
        None => {
            let mut input = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut input).unwrap_or_else(|e| {
                eprintln!("error: cannot read stdin: {e}");
                std::process::exit(1);
            });
            input
        }
    };

    let cnf = parse_dimacs(&input).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });

    let stop = Arc::new(AtomicBool::new(false));
    install_signal_handler(&stop, opts.verbose);
    let request = SolveRequest {
        mode: SolveMode::Exact,
        seed: opts.seed,
        limits: SolveLimits {
            time: opts.time_limit,
            memory_bytes: opts.memory_limit_mb.map(|mb| mb.saturating_mul(1024).saturating_mul(1024)),
            ..SolveLimits::default()
        },
        sat: SatControls { backend: Some(opts.backend), preprocess: opts.preprocess, proof_path: opts.proof.clone() },
        ..SolveRequest::default()
    };
    let package = semantic_package(&cnf);
    let mut events = IgnoreEvents;
    let result = qayd::solve_with_stop(&package, &request, Arc::clone(&stop), &mut events).unwrap_or_else(|error| {
        eprintln!("error: {error}");
        std::process::exit(1);
    });
    print_result(&cnf, &result).unwrap_or_else(|error| {
        eprintln!("error: {error}");
        std::process::exit(1);
    });
    if opts.verbose {
        let report = result.reports().first();
        let stats = result.aggregate_search_stats();
        eprintln!(
            "c time={:.3}s nodes={} failures={} solutions={} learned_lits={} binary_visits={} watched_visits={} watched_scans={} binary_props={} pre_rounds={} pre_in={} pre_out={} pre_dup_lits={} pre_taut={} pre_units={} pre_pure={} pre_subsumed={} pre_ssr_lits={} pre_bve_vars={} pre_bve_resolvents={} pre_blocked={}",
            result.elapsed().as_secs_f64(),
            stats.nodes,
            stats.failures,
            stats.solutions,
            stats.learned_lits,
            stats.binary_clause_visits,
            stats.watched_clause_visits,
            stats.watched_literal_scans,
            stats.binary_implications,
            metadata_counter(report, "pre_rounds"),
            metadata_counter(report, "pre_in"),
            metadata_counter(report, "pre_out"),
            metadata_counter(report, "pre_dup_lits"),
            metadata_counter(report, "pre_taut"),
            metadata_counter(report, "pre_units"),
            metadata_counter(report, "pre_pure"),
            metadata_counter(report, "pre_subsumed"),
            metadata_counter(report, "pre_ssr_lits"),
            metadata_counter(report, "pre_bve_vars"),
            metadata_counter(report, "pre_bve_resolvents"),
            metadata_counter(report, "pre_blocked"),
        );
    }
    if opts.competition {
        std::process::exit(competition_exit_code(result.status()));
    }
}

fn parse_args() -> (Options, Option<String>) {
    let mut opts = Options {
        verbose: false,
        time_limit: None,
        memory_limit_mb: None,
        competition: false,
        backend: SatBackendMode::Native,
        seed: 0,
        preprocess: SatPreprocess::Basic,
        proof: None,
    };
    let mut path = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-v" | "--verbose" => opts.verbose = true,
            "--competition" => opts.competition = true,
            "--native" => opts.backend = SatBackendMode::Native,
            "--linear" => opts.backend = SatBackendMode::Linear,
            "--no-preprocess" => opts.preprocess = SatPreprocess::Off,
            "--preprocess" => {
                let level = args.next().unwrap_or_else(|| {
                    eprintln!("error: --preprocess expects off, basic, or full");
                    std::process::exit(1);
                });
                opts.preprocess = parse_preprocess_level(&level).unwrap_or_else(|| {
                    eprintln!("error: invalid --preprocess `{level}`; expected off, basic, or full");
                    std::process::exit(1);
                });
            }
            "--proof" => {
                opts.proof = Some(args.next().unwrap_or_else(|| {
                    eprintln!("error: --proof expects a path");
                    std::process::exit(1);
                }));
            }
            "--seed" => {
                opts.seed = args.next().and_then(|s| s.parse::<u64>().ok()).unwrap_or_else(|| {
                    eprintln!("error: --seed expects an unsigned integer");
                    std::process::exit(1);
                });
            }
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
        "usage: qayd-sat [options] [instance.cnf]\n\
         \n\
         options:\n\
         \x20 -v, --verbose       print statistics as comment lines on stderr\n\
         \x20 --competition       use SAT competition exit codes: 10 SAT, 20 UNSAT, 0 UNKNOWN\n\
         \x20 -t, --time <sec>    stop after <sec> seconds\n\
         \x20 --memory-mb <mb>    set an address-space memory limit on Linux\n\
         \x20 --native            use direct native clauses (default)\n\
         \x20 --linear            use the old linear clause lowering\n\
         \x20 --seed <n>          set a reproducible native CDCL seed\n\
         \x20 --preprocess <lvl>  native preprocessing: off, basic, full\n\
         \x20 --no-preprocess     disable native CNF preprocessing\n\
         \x20 --proof <path>      write a DRAT proof (native backend only)\n\
         \x20 -h, --help          show this help\n\
         \n\
         Reads DIMACS CNF from a file, or stdin when no file is given."
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
    match status {
        SolveStatus::Satisfiable => 10,
        SolveStatus::Unsatisfiable => 20,
        SolveStatus::Optimal | SolveStatus::Unknown | SolveStatus::Unsupported => 0,
    }
}

fn parse_preprocess_level(level: &str) -> Option<SatPreprocess> {
    match level {
        "off" => Some(SatPreprocess::Off),
        "basic" => Some(SatPreprocess::Basic),
        "full" => Some(SatPreprocess::Full),
        _ => None,
    }
}

fn semantic_package(cnf: &qayd_sat::Cnf) -> ModelPackage {
    let mut model = Model::new();
    let variables = (0..cnf.vars).map(|_| model.bool_var()).collect::<Vec<_>>();
    for clause in &cnf.clauses {
        model.add_constraint(Constraint::Clause(
            clause
                .iter()
                .map(|literal| BoolLiteral { variable: variables[literal.unsigned_abs() as usize - 1], positive: *literal > 0 })
                .collect(),
        ));
    }
    let mut package = ModelPackage::new(model);
    for (index, variable) in variables.into_iter().enumerate() {
        let object = ModelObject::IntVar(variable);
        let name = (index + 1).to_string();
        package.metadata.names.insert(object, name.clone());
        package.metadata.frontend_ids.insert(("dimacs".to_string(), name), object);
        package.metadata.outputs.push(object);
    }
    package
}

fn print_result(cnf: &qayd_sat::Cnf, result: &SolveResult) -> Result<(), SolveError> {
    match result.status() {
        SolveStatus::Satisfiable => {
            let candidate =
                result.primal().ok_or_else(|| SolveError::InvalidResult("SAT result has no verified assignment".to_string()))?;
            if candidate.assignment().integers.len() != cnf.vars {
                return Err(SolveError::InvalidResult(format!(
                    "verified SAT assignment has {} variables, expected {}",
                    candidate.assignment().integers.len(),
                    cnf.vars
                )));
            }
            let assignment = candidate
                .assignment()
                .integers
                .iter()
                .map(|value| match value {
                    Some(0) => Ok(false),
                    Some(1) => Ok(true),
                    Some(value) => Err(SolveError::InvalidResult(format!("verified SAT variable has non-Boolean value {value}"))),
                    None => Err(SolveError::InvalidResult("verified SAT assignment contains an unassigned variable".to_string())),
                })
                .collect::<Result<Vec<_>, _>>()?;
            debug_assert!(assignment_satisfies(cnf, &assignment));
            println!("s SATISFIABLE");
            print_assignment(&assignment);
        }
        SolveStatus::Unsatisfiable => println!("s UNSATISFIABLE"),
        SolveStatus::Unknown => println!("s UNKNOWN"),
        SolveStatus::Unsupported => println!("s UNSUPPORTED"),
        SolveStatus::Optimal => {
            return Err(SolveError::InvalidResult("SAT decision frontend received an optimization result".to_string()));
        }
    }
    Ok(())
}

fn metadata_counter(report: Option<&qayd::orchestrator::EngineReport>, key: &str) -> u64 {
    report
        .and_then(|report| report.metadata.iter().find_map(|(name, value)| if name == key { value.parse().ok() } else { None }))
        .unwrap_or(0)
}

fn print_assignment(assignment: &[bool]) {
    print!("v");
    for (idx, &value) in assignment.iter().enumerate() {
        let lit = if value { idx as i32 + 1 } else { -(idx as i32 + 1) };
        print!(" {lit}");
    }
    println!(" 0");
    let _ = io::stdout().flush();
}
