//! `qayd-sat`: a minimal DIMACS CNF front-end for `qayd`.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use qayd_sat::proof::{ProofFormat, ProofWriter};
use qayd_sat::{
    assignment_satisfies, parse_dimacs, solve_cnf_native_seeded_with_proof_options, solve_cnf_with_backend_seeded_options,
    PreprocessOptions, SatBackend, SatResult, Status,
};

struct Options {
    verbose: bool,
    time_limit: Option<Duration>,
    memory_limit_mb: Option<u64>,
    competition: bool,
    backend: SatBackend,
    seed: u64,
    preprocess: PreprocessOptions,
    proof: Option<String>,
}

fn main() {
    let (opts, path) = parse_args();
    if let Some(limit) = opts.memory_limit_mb {
        set_memory_limit(limit).unwrap_or_else(|e| {
            eprintln!("error: cannot apply memory limit: {e}");
            std::process::exit(1);
        });
    }

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
    if let Some(limit) = opts.time_limit {
        let flag = Arc::clone(&stop);
        thread::spawn(move || {
            thread::sleep(limit);
            flag.store(true, Ordering::Relaxed);
        });
    }

    let start = Instant::now();
    let result = solve_with_options(&cnf, &stop, &opts);
    print_result(&cnf, &result);
    if opts.verbose {
        eprintln!(
            "c time={:.3}s nodes={} failures={} solutions={} learned_lits={} binary_visits={} watched_visits={} watched_scans={} binary_props={} pre_rounds={} pre_in={} pre_out={} pre_dup_lits={} pre_taut={} pre_units={} pre_pure={} pre_subsumed={} pre_ssr_lits={} pre_bve_vars={} pre_bve_resolvents={} pre_blocked={}",
            start.elapsed().as_secs_f64(),
            result.stats.nodes,
            result.stats.failures,
            result.stats.solutions,
            result.stats.learned_lits,
            result.stats.binary_clause_visits,
            result.stats.watched_clause_visits,
            result.stats.watched_literal_scans,
            result.stats.binary_implications,
            result.preprocess.rounds,
            result.preprocess.input_clauses,
            result.preprocess.output_clauses,
            result.preprocess.duplicate_literals,
            result.preprocess.tautological_clauses,
            result.preprocess.unit_assignments,
            result.preprocess.pure_assignments,
            result.preprocess.subsumed_clauses,
            result.preprocess.self_subsumed_literals,
            result.preprocess.bve_variables,
            result.preprocess.bve_resolvents,
            result.preprocess.blocked_clauses,
        );
    }
    if opts.competition {
        std::process::exit(competition_exit_code(result.status));
    }
}

fn parse_args() -> (Options, Option<String>) {
    let mut opts = Options {
        verbose: false,
        time_limit: None,
        memory_limit_mb: None,
        competition: false,
        backend: SatBackend::Native,
        seed: 0,
        preprocess: PreprocessOptions::default(),
        proof: None,
    };
    let mut path = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-v" | "--verbose" => opts.verbose = true,
            "--competition" => opts.competition = true,
            "--native" => opts.backend = SatBackend::Native,
            "--linear" => opts.backend = SatBackend::Linear,
            "--no-preprocess" => opts.preprocess = PreprocessOptions::off(),
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
        flag.store(true, Ordering::Relaxed);
    }) {
        if verbose {
            eprintln!("c signal_handler=unavailable reason={err}");
        }
    }
}

fn competition_exit_code(status: Status) -> i32 {
    match status {
        Status::Satisfiable => 10,
        Status::Unsatisfiable => 20,
        Status::Unknown => 0,
    }
}

#[cfg(target_os = "linux")]
fn set_memory_limit(memory_limit_mb: u64) -> io::Result<()> {
    let bytes = memory_limit_mb.saturating_mul(1024).saturating_mul(1024);
    let mut current = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    let get_rc = unsafe { libc::getrlimit(libc::RLIMIT_AS, &mut current) };
    if get_rc != 0 {
        return Err(io::Error::last_os_error());
    }
    let requested = bytes as libc::rlim_t;
    if current.rlim_max != libc::RLIM_INFINITY && requested > current.rlim_max {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("requested limit exceeds hard limit {}", current.rlim_max)));
    }
    let limit = libc::rlimit { rlim_cur: requested, rlim_max: current.rlim_max };
    let rc = unsafe { libc::setrlimit(libc::RLIMIT_AS, &limit) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(not(target_os = "linux"))]
fn set_memory_limit(_memory_limit_mb: u64) -> io::Result<()> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "--memory-mb is only supported on Linux"))
}

fn parse_preprocess_level(level: &str) -> Option<PreprocessOptions> {
    match level {
        "off" => Some(PreprocessOptions::off()),
        "basic" => Some(PreprocessOptions::basic()),
        "full" => Some(PreprocessOptions::full()),
        _ => None,
    }
}

fn solve_with_options(cnf: &qayd_sat::Cnf, stop: &AtomicBool, opts: &Options) -> SatResult {
    let Some(path) = opts.proof.as_deref() else {
        return solve_cnf_with_backend_seeded_options(cnf, stop, opts.backend, opts.seed, opts.preprocess);
    };
    if opts.backend != SatBackend::Native {
        eprintln!("error: --proof requires --native");
        std::process::exit(1);
    }

    let file = File::create(path).unwrap_or_else(|e| {
        eprintln!("error: cannot create proof file {path}: {e}");
        std::process::exit(1);
    });
    let mut proof = ProofWriter::new(BufWriter::new(file), ProofFormat::Drat);
    solve_cnf_native_seeded_with_proof_options(cnf, stop, opts.seed, opts.preprocess, &mut proof).unwrap_or_else(|e| {
        eprintln!("error: cannot write proof file {path}: {e}");
        std::process::exit(1);
    })
}

fn print_result(cnf: &qayd_sat::Cnf, result: &SatResult) {
    match result.status {
        Status::Satisfiable => {
            let assignment = result.assignment.as_deref().expect("SAT result has an assignment");
            debug_assert!(assignment_satisfies(cnf, assignment));
            println!("s SATISFIABLE");
            print_assignment(assignment);
        }
        Status::Unsatisfiable => println!("s UNSATISFIABLE"),
        Status::Unknown => println!("s UNKNOWN"),
    }
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
