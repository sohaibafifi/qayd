//! XCSP3-core front-end: parse an instance, build the model, solve, and report.
//!
//! A useful subset of XCSP3-core is supported. Unhandled features produce a
//! clear error rather than a wrong answer.

pub mod build;
pub mod dom;
pub mod parse;

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::search::{optimize_with, solve_interruptible, SearchControl};

fn write_values<W: Write>(w: &mut W, sol: &[i32]) -> std::io::Result<()> {
    write!(w, "v")?;
    for v in sol {
        write!(w, " {v}")?;
    }
    writeln!(w)
}

/// Parse, build, and solve an XCSP3 instance, returning competition-style
/// output lines (`s ...`, `o ...`, `v ...`) as a string. No `c` comments.
pub fn run(xml: &str) -> Result<String, String> {
    let mut buf = Vec::new();
    let never = AtomicBool::new(false);
    run_to(xml, false, &never, &mut buf)?;
    String::from_utf8(buf).map_err(|e| e.to_string())
}

/// Solve an XCSP3 instance, **streaming** output to `w` as it is produced. When
/// `verbose`, emits `c` comment lines (model size, search stats, time) and an
/// `o` line per improving bound, flushing after each so progress is live.
///
/// Search halts as soon as `stop` is set (time limit / Ctrl+C), reporting the
/// best solution found so far (`s SATISFIABLE` / `s UNKNOWN` instead of
/// `s OPTIMUM FOUND` / `s UNSATISFIABLE`).
pub fn run_to<W: Write>(
    xml: &str,
    verbose: bool,
    stop: &AtomicBool,
    w: &mut W,
) -> Result<(), String> {
    let start = Instant::now();
    let root = dom::parse(xml)?;
    let built = build::build(&root)?;
    let mut solver = built.solver;
    let vars = built.vars;

    let to_err = |e: std::io::Error| e.to_string();

    if verbose {
        let n_cons = root.child("constraints").map_or(0, |c| c.children.len());
        let kind = if built.objective.is_some() {
            "COP"
        } else {
            "CSP"
        };
        writeln!(w, "c qayd XCSP3").map_err(to_err)?;
        writeln!(w, "c type {kind}").map_err(to_err)?;
        writeln!(w, "c variables {}", solver.store.num_vars()).map_err(to_err)?;
        writeln!(w, "c constraint elements {n_cons}").map_err(to_err)?;
        writeln!(w, "c propagators {}", solver.num_propagators()).map_err(to_err)?;
        w.flush().map_err(to_err)?;
    }

    match built.objective {
        None => {
            let mut sol = None;
            let stats = solve_interruptible(
                &mut solver,
                &vars,
                |s| {
                    sol = Some(vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
                    SearchControl::Stop
                },
                stop,
            );
            if verbose {
                writeln!(w, "c nodes {} failures {}", stats.nodes, stats.failures)
                    .map_err(to_err)?;
            }
            match sol {
                Some(s) => {
                    writeln!(w, "s SATISFIABLE").map_err(to_err)?;
                    write_values(w, &s).map_err(to_err)?;
                }
                None if stop.load(Ordering::Relaxed) => writeln!(w, "s UNKNOWN").map_err(to_err)?,
                None => writeln!(w, "s UNSATISFIABLE").map_err(to_err)?,
            }
        }
        Some((minimizing, obj)) => {
            // Stream each improving bound live.
            let mut io_err: Option<std::io::Error> = None;
            let (best, stats) = optimize_with(&mut solver, &vars, obj, minimizing, stop, |v| {
                if verbose && io_err.is_none() {
                    if let Err(e) = writeln!(w, "o {v}").and_then(|_| w.flush()) {
                        io_err = Some(e);
                    }
                }
            });
            if let Some(e) = io_err {
                return Err(e.to_string());
            }
            if verbose {
                writeln!(w, "c nodes {} failures {}", stats.nodes, stats.failures)
                    .map_err(to_err)?;
            }
            let interrupted = stop.load(Ordering::Relaxed);
            match best {
                Some((sol, value)) => {
                    // Proven optimal only if search finished without interruption.
                    let status = if interrupted {
                        "s SATISFIABLE"
                    } else {
                        "s OPTIMUM FOUND"
                    };
                    writeln!(w, "{status}").map_err(to_err)?;
                    writeln!(w, "o {value}").map_err(to_err)?;
                    write_values(w, &sol).map_err(to_err)?;
                }
                None if interrupted => writeln!(w, "s UNKNOWN").map_err(to_err)?,
                None => writeln!(w, "s UNSATISFIABLE").map_err(to_err)?,
            }
        }
    }

    if verbose {
        if stop.load(Ordering::Relaxed) {
            writeln!(w, "c interrupted").map_err(to_err)?;
        }
        writeln!(w, "c time {:.3}s", start.elapsed().as_secs_f64()).map_err(to_err)?;
    }
    w.flush().map_err(to_err)?;
    Ok(())
}
