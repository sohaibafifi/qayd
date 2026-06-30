//! `qayd-sat`: a minimal DIMACS CNF front-end for `qayd`.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use qayd_sat::{assignment_satisfies, parse_dimacs, solve_cnf, SatResult, Status};

struct Options {
    verbose: bool,
    time_limit: Option<Duration>,
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
    if let Some(limit) = opts.time_limit {
        let flag = Arc::clone(&stop);
        thread::spawn(move || {
            thread::sleep(limit);
            flag.store(true, Ordering::Relaxed);
        });
    }

    let start = Instant::now();
    let result = solve_cnf(&cnf, &stop);
    print_result(&cnf, &result);
    if opts.verbose {
        eprintln!(
            "c time={:.3}s nodes={} failures={} solutions={} learned_lits={}",
            start.elapsed().as_secs_f64(),
            result.stats.nodes,
            result.stats.failures,
            result.stats.solutions,
            result.stats.learned_lits,
        );
    }
}

fn parse_args() -> (Options, Option<String>) {
    let mut opts = Options { verbose: false, time_limit: None };
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
         \x20 -t, --time <sec>    stop after <sec> seconds\n\
         \x20 -h, --help          show this help\n\
         \n\
         Reads DIMACS CNF from a file, or stdin when no file is given."
    );
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
