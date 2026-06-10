//! Search driver: dispatch CSP / COP, stream progress, honour a time limit.
//! Output follows the MiniZinc FlatZinc solver protocol: `name = value;` items
//! for annotated output variables, `----------` after a solution, `==========`
//! after a completeness proof, `=====UNSATISFIABLE=====` / `=====UNKNOWN=====`
//! otherwise. Verbose extras are emitted as `%` comment lines.

use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use qayd::{optimize_with, solve_interruptible, SearchControl, SolveStats, VarId};

use crate::model::{Model, Output};

/// Run-time options from the command line.
pub(crate) struct Options {
    /// Stream improving bounds and print end-of-search statistics (as `%` comments).
    pub(crate) verbose: bool,
    /// Wall-clock limit; search is interrupted once it elapses.
    pub(crate) time_limit: Option<Duration>,
}

/// Solve `model` and print the result in the MiniZinc solver protocol.
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
            println!("% o {value}");
            let _ = io::stdout().flush();
        }
    });
    let interrupted = stop.load(Ordering::Relaxed);
    match best {
        Some((solution, _)) => {
            print_solution(model, &solution);
            println!("----------");
            // `==========` is the completeness proof; an interrupted incumbent
            // is feasible but not proven optimal.
            if !interrupted {
                println!("==========");
            }
        }
        None => println!("{}", if interrupted { "=====UNKNOWN=====" } else { "=====UNSATISFIABLE=====" }),
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
            print_solution(model, &values);
            println!("----------");
        }
        None => println!("{}", if stop.load(Ordering::Relaxed) { "=====UNKNOWN=====" } else { "=====UNSATISFIABLE=====" }),
    }
    if opts.verbose {
        print_stats(&stats, start);
    }
}

/// Print the annotated output items in MiniZinc's FlatZinc solution format:
/// `name = value;` and `name = arrayNd(1..a, ..., [values]);`.
fn print_solution(model: &Model, assignment: &[i32]) {
    let by_search: HashMap<VarId, i32> = model.search.iter().copied().zip(assignment.iter().copied()).collect();
    // Constants and root-fixed variables keep their value in the store.
    let value = |v: VarId| by_search.get(&v).copied().unwrap_or_else(|| model.solver.store.value(v));
    let fmt = |v: VarId, is_bool: bool| {
        let x = value(v);
        if is_bool {
            if x != 0 {
                "true".to_string()
            } else {
                "false".to_string()
            }
        } else {
            x.to_string()
        }
    };
    for out in &model.outputs {
        match out {
            Output::Var { name, var, is_bool } => println!("{name} = {};", fmt(*var, *is_bool)),
            Output::Array { name, dims, vars, is_bool } => {
                let ranges = dims.iter().map(|(lo, hi)| format!("{lo}..{hi}")).collect::<Vec<_>>().join(", ");
                let values = vars.iter().map(|&v| fmt(v, *is_bool)).collect::<Vec<_>>().join(", ");
                println!("{name} = array{}d({ranges}, [{values}]);", dims.len());
            }
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
