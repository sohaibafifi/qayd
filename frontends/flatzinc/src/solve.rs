//! Search driver: dispatch CSP / COP, stream progress, honour a time limit.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use qayd::{optimize_with, solve_interruptible, SearchControl, SolveStats, VarId};

use crate::model::Model;

/// Run-time options from the command line.
pub(crate) struct Options {
    /// Stream improving bounds (`o` lines) and print end-of-search statistics.
    pub(crate) verbose: bool,
    /// Wall-clock limit; search is interrupted once it elapses.
    pub(crate) time_limit: Option<Duration>,
}

/// Solve `model` and print FlatZinc-style status, solution, and (in verbose mode)
/// progress and statistics.
pub(crate) fn solve(mut model: Model, opts: &Options) {
    let start = Instant::now();
    let stop = Arc::new(AtomicBool::new(false));
    if let Some(limit) = opts.time_limit {
        let flag = Arc::clone(&stop);
        thread::spawn(move || {
            thread::sleep(limit);
            flag.store(true, Ordering::Relaxed);
        });
    }
    let stop_ref: &AtomicBool = &stop;
    match model.objective {
        Some((minimizing, obj)) => solve_opt(&mut model, minimizing, obj, stop_ref, opts, start),
        None => solve_sat(&mut model, stop_ref, opts, start),
    }
}

/// Branch-and-bound optimisation, streaming each improving bound when verbose.
fn solve_opt(model: &mut Model, minimizing: bool, obj: VarId, stop: &AtomicBool, opts: &Options, start: Instant) {
    let verbose = opts.verbose;
    let (best, stats) = optimize_with(&mut model.solver, &model.search, obj, minimizing, stop, |value| {
        if verbose {
            println!("o {value}");
            let _ = io::stdout().flush();
        }
    });
    let interrupted = stop.load(Ordering::Relaxed);
    match best {
        Some((solution, value)) => {
            if !verbose {
                println!("o {value}");
            }
            // An incumbent under an interrupted search is feasible but not proven optimal.
            println!("s {}", if interrupted { "SATISFIABLE" } else { "OPTIMUM FOUND" });
            print_solution(model, &solution);
        }
        None => println!("s {}", if interrupted { "UNKNOWN" } else { "UNSATISFIABLE" }),
    }
    if verbose {
        print_stats(&stats, start);
    }
}

/// Satisfaction search: report the first solution, or UNSAT / UNKNOWN.
fn solve_sat(model: &mut Model, stop: &AtomicBool, opts: &Options, start: Instant) {
    let mut solution = None;
    let stats = solve_interruptible(
        &mut model.solver,
        &model.search,
        |s| {
            solution = Some(model.search.iter().map(|&v| s.store.value(v)).collect::<Vec<i32>>());
            SearchControl::Stop
        },
        stop,
    );
    match solution {
        Some(values) => {
            println!("s SATISFIABLE");
            print_solution(model, &values);
        }
        None => println!("s {}", if stop.load(Ordering::Relaxed) { "UNKNOWN" } else { "UNSATISFIABLE" }),
    }
    if opts.verbose {
        print_stats(&stats, start);
    }
}

/// Print `v name = value` for each named variable, in declaration order.
fn print_solution(model: &Model, assignment: &[i32]) {
    for (name, var) in &model.names {
        if let Some(pos) = model.search.iter().position(|&v| v == *var) {
            println!("v {name} = {}", assignment[pos]);
        }
    }
}

/// Print a `%` statistics comment line.
fn print_stats(stats: &SolveStats, start: Instant) {
    println!(
        "% time={:.3}s nodes={} failures={} solutions={} learned_lits={}",
        start.elapsed().as_secs_f64(),
        stats.nodes,
        stats.failures,
        stats.solutions,
        stats.learned_lits,
    );
}
