//! XCSP3-core front-end: parse via `xcsp3-rust-parser`, build through
//! [`callback::Model`], solve, and report competition-style output.

mod callback;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use xcsp3_rust_parser::xcsp_runner::XcspRunner;

use crate::ids::VarId;
use crate::search::{optimize_seeded, solve_interruptible_seeded, SearchControl, SolveStats};
use crate::store::Solver;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

struct StagedXml(PathBuf);

impl Drop for StagedXml {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// XCSP execution settings. Parallel search is opt-in with `workers > 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunOptions {
    /// Reproducible base seed. Worker `w` uses `seed + w`.
    pub seed: u64,
    /// Number of independent portfolio workers.
    pub workers: usize,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            workers: 1,
        }
    }
}

fn write_values<W: Write>(w: &mut W, sol: &[i32]) -> std::io::Result<()> {
    write!(w, "v")?;
    for v in sol {
        write!(w, " {v}")?;
    }
    writeln!(w)
}

fn stage_xml(xml: &str) -> Result<StagedXml, String> {
    let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("qayd-{}-{}.xml", std::process::id(), n));
    std::fs::write(&path, xml).map_err(|e| e.to_string())?;
    Ok(StagedXml(path))
}

/// Immutable root model. Portfolio workers clone it instead of reparsing XML.
#[derive(Clone)]
struct Problem {
    solver: Solver,
    order: Vec<VarId>,
    objective: Option<(bool, VarId)>,
}

fn parse_problem(path: &Path) -> Result<Problem, String> {
    let mut model = callback::Model::new();
    let path_str = path.to_str().ok_or("bad temp path")?;
    let parsed = XcspRunner::run(path_str, &mut model);
    parsed.map_err(|e| e.to_string())?;
    if let Some(e) = model.error.take() {
        return Err(e);
    }
    Ok(Problem {
        solver: model.solver,
        order: model.order,
        objective: model.objective,
    })
}

struct CopShared {
    cancel: AtomicBool,
    proved: AtomicBool,
    best: AtomicI64,
    solution: Mutex<Option<(Vec<i32>, i32)>>,
    minimizing: bool,
}

impl CopShared {
    fn improve(&self, value: i32, solution: &[i32]) -> bool {
        let value64 = value as i64;
        let mut current = self.best.load(Ordering::Relaxed);
        while if self.minimizing {
            value64 < current
        } else {
            value64 > current
        } {
            match self.best.compare_exchange_weak(
                current,
                value64,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let mut slot = self.solution.lock().unwrap();
                    if self.best.load(Ordering::Acquire) == value64 {
                        *slot = Some((solution.to_vec(), value));
                    }
                    return true;
                }
                Err(actual) => current = actual,
            }
        }
        false
    }
}

enum WorkerMsg {
    Improved(i32),
    Done(SolveStats),
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
    run_to_with_options(xml, verbose, stop, w, RunOptions::default())
}

/// Like [`run_to`], with explicit seed and portfolio-worker settings.
pub fn run_to_with_options<W: Write>(
    xml: &str,
    verbose: bool,
    stop: &AtomicBool,
    w: &mut W,
    options: RunOptions,
) -> Result<(), String> {
    let start = Instant::now();
    let options = RunOptions {
        workers: options.workers.max(1),
        ..options
    };
    let path = stage_xml(xml)?;
    let problem = parse_problem(&path.0)?;
    let options = RunOptions {
        workers: if problem.objective.is_some() {
            options.workers
        } else {
            1
        },
        ..options
    };
    let to_err = |e: std::io::Error| e.to_string();

    if verbose {
        let kind = if problem.objective.is_some() {
            "COP"
        } else {
            "CSP"
        };
        writeln!(w, "c qayd XCSP3").map_err(to_err)?;
        writeln!(w, "c type {kind}").map_err(to_err)?;
        writeln!(w, "c variables {}", problem.solver.store.num_vars()).map_err(to_err)?;
        writeln!(w, "c propagators {}", problem.solver.num_propagators()).map_err(to_err)?;
        writeln!(w, "c seed {}", options.seed).map_err(to_err)?;
        writeln!(w, "c workers {}", options.workers).map_err(to_err)?;
        w.flush().map_err(to_err)?;
    }

    if options.workers == 1 || problem.objective.is_none() {
        solve_single(problem, verbose, stop, w, options.seed)?;
    } else {
        let minimizing = problem.objective.unwrap().0;
        solve_parallel_cop(problem, minimizing, verbose, stop, w, options)?;
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

fn solve_single<W: Write>(
    problem: Problem,
    verbose: bool,
    stop: &AtomicBool,
    w: &mut W,
    seed: u64,
) -> Result<(), String> {
    let mut solver = problem.solver;
    let vars = problem.order;
    let to_err = |e: std::io::Error| e.to_string();

    match problem.objective {
        None => {
            let mut sol = None;
            let stats = solve_interruptible_seeded(
                &mut solver,
                &vars,
                |s| {
                    sol = Some(vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
                    SearchControl::Stop
                },
                stop,
                seed,
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
            let (best, stats, _) = optimize_seeded(
                &mut solver,
                &vars,
                obj,
                minimizing,
                stop,
                seed,
                None,
                |v, _| {
                    if verbose && io_err.is_none() {
                        if let Err(e) = writeln!(w, "o {v}").and_then(|_| w.flush()) {
                            io_err = Some(e);
                        }
                    }
                },
            );
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

    Ok(())
}

fn solve_parallel_cop<W: Write>(
    problem: Problem,
    minimizing: bool,
    verbose: bool,
    stop: &AtomicBool,
    w: &mut W,
    options: RunOptions,
) -> Result<(), String> {
    let shared = Arc::new(CopShared {
        cancel: AtomicBool::new(false),
        proved: AtomicBool::new(false),
        best: AtomicI64::new(if minimizing { i64::MAX } else { i64::MIN }),
        solution: Mutex::new(None),
        minimizing,
    });
    let (tx, rx) = mpsc::channel();
    let mut stats = SolveStats::default();
    let mut printed = None;
    let mut error = None;
    let mut models = Vec::with_capacity(options.workers);
    models.push(problem);
    for _ in 1..options.workers {
        models.push(models[0].clone());
    }

    std::thread::scope(|scope| {
        let monitor = Arc::clone(&shared);
        scope.spawn(move || {
            while !monitor.cancel.load(Ordering::Relaxed) {
                if stop.load(Ordering::Relaxed) {
                    monitor.cancel.store(true, Ordering::Relaxed);
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        });
        for (worker, model) in models.into_iter().enumerate() {
            let shared = Arc::clone(&shared);
            let tx = tx.clone();
            scope.spawn(move || {
                let mut solver = model.solver;
                let vars = model.order;
                let (worker_minimizing, obj) =
                    model.objective.expect("COP worker lost its objective");
                assert_eq!(worker_minimizing, minimizing, "COP direction mismatch");
                let (_, worker_stats, complete) = optimize_seeded(
                    &mut solver,
                    &vars,
                    obj,
                    minimizing,
                    &shared.cancel,
                    options.seed.wrapping_add(worker as u64),
                    Some(&shared.best),
                    |value, solution| {
                        if shared.improve(value, solution) {
                            let _ = tx.send(WorkerMsg::Improved(value));
                        }
                    },
                );
                if complete {
                    shared.proved.store(true, Ordering::Relaxed);
                    shared.cancel.store(true, Ordering::Relaxed);
                }
                let _ = tx.send(WorkerMsg::Done(worker_stats));
            });
        }
        drop(tx);
        for msg in rx {
            match msg {
                WorkerMsg::Improved(value)
                    if printed.is_none_or(
                        |old| {
                            if minimizing {
                                value < old
                            } else {
                                value > old
                            }
                        },
                    ) =>
                {
                    printed = Some(value);
                    if verbose && error.is_none() {
                        if let Err(e) = writeln!(w, "o {value}").and_then(|_| w.flush()) {
                            error = Some(e.to_string());
                            shared.cancel.store(true, Ordering::Relaxed);
                        }
                    }
                }
                WorkerMsg::Improved(_) => {}
                WorkerMsg::Done(worker_stats) => {
                    stats.solutions += worker_stats.solutions;
                    stats.nodes += worker_stats.nodes;
                    stats.failures += worker_stats.failures;
                    stats.learned_lits += worker_stats.learned_lits;
                }
            }
        }
        shared.cancel.store(true, Ordering::Relaxed);
    });

    if let Some(e) = error {
        return Err(e);
    }
    let to_err = |e: std::io::Error| e.to_string();
    if verbose {
        writeln!(w, "c nodes {} failures {}", stats.nodes, stats.failures).map_err(to_err)?;
    }
    let interrupted = stop.load(Ordering::Relaxed);
    let proved = shared.proved.load(Ordering::Relaxed);
    match shared.solution.lock().unwrap().clone() {
        Some((sol, value)) => {
            let status = if proved {
                "s OPTIMUM FOUND"
            } else {
                "s SATISFIABLE"
            };
            writeln!(w, "{status}").map_err(to_err)?;
            writeln!(w, "o {value}").map_err(to_err)?;
            write_values(w, &sol).map_err(to_err)?;
        }
        None if interrupted => writeln!(w, "s UNKNOWN").map_err(to_err)?,
        None if proved => writeln!(w, "s UNSATISFIABLE").map_err(to_err)?,
        None => writeln!(w, "s UNKNOWN").map_err(to_err)?,
    }
    Ok(())
}
