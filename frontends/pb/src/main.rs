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
//! Current scope: linear PBS/PBO without certificates. Non-linear (product)
//! terms and objectives whose value range exceeds `i32` are reported
//! `s UNSUPPORTED`.

mod opb;

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use opb::{Objective, Op, Opb};
use qayd::constraints::linear::{linear, Relation};
use qayd::{first_solution, optimize_with, Solver, VarId};

struct Options {
    verbose: bool,
    time_limit: Option<Duration>,
    memory_limit_mb: Option<u64>,
}

/// Final answer status, used for the exit code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Satisfiable,
    Optimum,
    Unsatisfiable,
    Unknown,
    Unsupported,
}

fn main() {
    let (opts, path) = parse_args();
    if let Some(limit) = opts.memory_limit_mb {
        set_memory_limit(limit).unwrap_or_else(|e| {
            eprintln!("error: cannot apply memory limit: {e}");
            std::process::exit(1);
        });
    }

    let input = read_input(path.as_deref());

    let instance = opb::parse(&input).unwrap_or_else(|e| {
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
    let status = run(&instance, &stop);
    if opts.verbose {
        eprintln!("c time={:.3}s vars={} constraints={}", start.elapsed().as_secs_f64(), instance.n_vars, instance.constraints.len());
    }
    std::process::exit(competition_exit_code(status));
}

/// Build the model and solve. Prints the competition output; returns the status.
fn run(instance: &Opb, stop: &AtomicBool) -> Status {
    if instance.nonlinear {
        println!("s UNSUPPORTED");
        return Status::Unsupported;
    }

    let mut solver = Solver::new();
    // 0/1 variable for each OPB index; `vars[i]` is OPB variable `x(i+1)`.
    let vars: Vec<VarId> = (0..instance.n_vars).map(|_| solver.new_var_range(0, 1)).collect();

    for c in &instance.constraints {
        let (coeffs, cvars, rhs) = normalize(&c.terms, c.rhs, &vars);
        let rel = match c.op {
            Op::Ge => Relation::Ge,
            Op::Le => Relation::Le,
            Op::Eq => Relation::Eq,
            Op::Gt => Relation::Gt,
            Op::Lt => Relation::Lt,
        };
        linear(&mut solver, &coeffs, &cvars, rel, rhs);
    }

    match &instance.objective {
        None => solve_satisfaction(&mut solver, &vars),
        Some(obj) if obj.terms.is_empty() => solve_satisfaction(&mut solver, &vars),
        Some(obj) => solve_optimization(&mut solver, &vars, obj, stop),
    }
}

/// Collapse terms to one coefficient per variable and fold `~x` literals and the
/// resulting constant into the right-hand side.
///
/// `c·~x = c·(1 - x) = c - c·x`, so a negated term contributes `-c` to `x` and
/// `c` to the LHS constant, which moves to the rhs as `-c`.
fn normalize(terms: &[opb::Term], rhs: i64, vars: &[VarId]) -> (Vec<i64>, Vec<VarId>, i64) {
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
    let mut coeffs = Vec::new();
    let mut cvars = Vec::new();
    for (idx, &c) in coeff_by_var.iter().enumerate() {
        if c != 0 {
            coeffs.push(c);
            cvars.push(vars[idx]);
        }
    }
    (coeffs, cvars, rhs - constant)
}

fn solve_satisfaction(solver: &mut Solver, vars: &[VarId]) -> Status {
    match first_solution(solver, vars) {
        Some(model) => {
            println!("s SATISFIABLE");
            print_model(&model);
            Status::Satisfiable
        }
        None => {
            println!("s UNSATISFIABLE");
            Status::Unsatisfiable
        }
    }
}

fn solve_optimization(solver: &mut Solver, vars: &[VarId], obj: &Objective, stop: &AtomicBool) -> Status {
    // Objective in x-form (coefficients on `x`). `normalize` folds the `~x`
    // constant into its third return as `0 - folded`, so the additive constant
    // of the true objective is its negation.
    let (ocoeffs, ovars, neg_constant) = normalize(&obj.terms, 0, vars);
    let constant = -neg_constant;

    // Bounds of the x-form sum; the materialised objective variable needs an
    // i32 domain, so bail out if the range does not fit.
    let (mut lo, mut hi) = (0i64, 0i64);
    for &c in &ocoeffs {
        if c < 0 {
            lo += c;
        } else {
            hi += c;
        }
    }
    if lo < i32::MIN as i64 || hi > i32::MAX as i64 {
        println!("s UNSUPPORTED");
        return Status::Unsupported;
    }

    let obj_var = solver.new_var_range(lo as i32, hi as i32);
    // obj_var = sum(ocoeffs · ovars):  sum - obj_var = 0.
    let mut eq_coeffs = ocoeffs.clone();
    let mut eq_vars = ovars.clone();
    eq_coeffs.push(-1);
    eq_vars.push(obj_var);
    linear(solver, &eq_coeffs, &eq_vars, Relation::Eq, 0);

    // Reported objective value = x-form value + the folded `~x` constant, sign
    // handled by the maximise flag.
    let minimizing = !obj.maximize;
    let report = |x_value: i32| -> i64 { x_value as i64 + constant };

    let (best, _stats) = optimize_with(solver, vars, obj_var, minimizing, stop, |x_value| {
        println!("o {}", report(x_value));
        let _ = io::stdout().flush();
    });

    let interrupted = stop.load(Ordering::Relaxed);
    match best {
        Some((model, x_value)) => {
            if interrupted {
                println!("o {}", report(x_value));
                println!("s SATISFIABLE");
                print_model(&model);
                Status::Satisfiable
            } else {
                println!("o {}", report(x_value));
                println!("s OPTIMUM FOUND");
                print_model(&model);
                Status::Optimum
            }
        }
        None if interrupted => {
            println!("s UNKNOWN");
            Status::Unknown
        }
        None => {
            println!("s UNSATISFIABLE");
            Status::Unsatisfiable
        }
    }
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
        flag.store(true, Ordering::Relaxed);
    }) {
        if verbose {
            eprintln!("c signal_handler=unavailable reason={err}");
        }
    }
}

fn competition_exit_code(status: Status) -> i32 {
    // Mirrors the PB Competition convention: 10 SAT, 20 UNSAT, 30 OPTIMUM,
    // 0 otherwise.
    match status {
        Status::Satisfiable => 10,
        Status::Unsatisfiable => 20,
        Status::Optimum => 30,
        Status::Unknown | Status::Unsupported => 0,
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
