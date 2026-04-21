//! XCSP3-core front-end.
//!
//! Parsing is delegated to the `xcsp3-rust-parser` crate (callback interface);
//! [`callback::Model`] maps each callback onto the solver. This module wires
//! parsing to search and reports competition-style output.

mod callback;

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use xcsp3_rust_parser::xcsp_runner::XcspRunner;

use crate::search::{optimize_with, solve_interruptible, SearchControl};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn write_values<W: Write>(w: &mut W, sol: &[i32]) -> std::io::Result<()> {
    write!(w, "v")?;
    for v in sol {
        write!(w, " {v}")?;
    }
    writeln!(w)
}

/// Parse, build, and solve an XCSP3 instance, returning competition-style
/// output lines as a string. No `c` comments.
pub fn run(xml: &str) -> Result<String, String> {
    let mut buf = Vec::new();
    let never = AtomicBool::new(false);
    run_to(xml, false, &never, &mut buf)?;
    String::from_utf8(buf).map_err(|e| e.to_string())
}

/// Solve an XCSP3 instance, streaming output to `w`. When `verbose`, emits `c`
/// comment lines. Search halts as soon as `stop` is set (time limit / Ctrl+C).
pub fn run_to<W: Write>(
    xml: &str,
    verbose: bool,
    stop: &AtomicBool,
    w: &mut W,
) -> Result<(), String> {
    let start = Instant::now();

    // The parser reads from a path; stage the (already decompressed) XML.
    let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("qayd-{}-{}.xml", std::process::id(), n));
    std::fs::write(&path, xml).map_err(|e| e.to_string())?;

    let mut model = callback::Model::new();
    let path_str = path.to_str().ok_or("bad temp path")?;
    let parsed = XcspRunner::run(path_str, &mut model);
    let _ = std::fs::remove_file(&path);
    parsed.map_err(|e| e.to_string())?;
    if let Some(e) = model.error.take() {
        return Err(e);
    }

    let mut solver = model.solver;
    let vars = model.order;
    let to_err = |e: std::io::Error| e.to_string();

    if verbose {
        let kind = if model.objective.is_some() {
            "COP"
        } else {
            "CSP"
        };
        writeln!(w, "c qayd XCSP3").map_err(to_err)?;
        writeln!(w, "c type {kind}").map_err(to_err)?;
        writeln!(w, "c variables {}", solver.store.num_vars()).map_err(to_err)?;
        writeln!(w, "c propagators {}", solver.num_propagators()).map_err(to_err)?;
        w.flush().map_err(to_err)?;
    }

    match model.objective {
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
