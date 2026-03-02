//! `qayd` CLI.
//!
//! - `qayd [-v] <instance.xml[.lzma|.xz]>` — parse and solve an XCSP3 instance.
//! - `qayd [-v] [n]` — demo: solve N-Queens for board size `n` (default 8).
//!
//! `-v` / `--verbose` adds `c` comment lines: model size, improving bounds,
//! search statistics, and wall-clock time.

use std::path::Path;

use qayd::constraints::primitives::not_equal_offset;
use qayd::{count_solutions, solve, SearchControl, Solver, VarId};

fn build_queens(solver: &mut Solver, n: i32) -> Vec<VarId> {
    let q: Vec<VarId> = (0..n).map(|_| solver.new_var_range(0, n - 1)).collect();
    for i in 0..n as usize {
        for j in (i + 1)..n as usize {
            let (di, dj) = (i as i32, j as i32);
            not_equal_offset(solver, q[i], q[j], 0);
            not_equal_offset(solver, q[i], q[j], di - dj);
            not_equal_offset(solver, q[i], q[j], dj - di);
        }
    }
    q
}

fn run_queens(n: i32, verbose: bool) {
    if n < 1 {
        eprintln!("board size must be >= 1");
        std::process::exit(1);
    }
    let mut solver = Solver::new();
    let q = build_queens(&mut solver, n);

    let mut first = None;
    let stats = solve(&mut solver, &q, |s| {
        first = Some(q.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
        SearchControl::Stop
    });
    match first {
        Some(sol) => println!("{n}-queens: first solution = {sol:?}"),
        None => println!("{n}-queens: UNSAT"),
    }
    if verbose {
        eprintln!(
            "c first-solution search: nodes {} failures {}",
            stats.nodes, stats.failures
        );
    }
    let total = count_solutions(&mut solver, &q);
    println!("{n}-queens: {total} solution(s)");
}

/// Read an instance, transparently decompressing `.lzma` / `.xz`.
fn read_instance(path: &str) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let xml = if path.ends_with(".lzma") {
        let mut out = Vec::new();
        lzma_rs::lzma_decompress(&mut &bytes[..], &mut out).map_err(|e| e.to_string())?;
        String::from_utf8(out).map_err(|e| e.to_string())?
    } else if path.ends_with(".xz") {
        let mut out = Vec::new();
        lzma_rs::xz_decompress(&mut &bytes[..], &mut out).map_err(|e| e.to_string())?;
        String::from_utf8(out).map_err(|e| e.to_string())?
    } else {
        String::from_utf8(bytes).map_err(|e| e.to_string())?
    };
    Ok(xml)
}

fn run_instance(path: &str, verbose: bool) {
    let xml = match read_instance(path) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    if let Err(e) = qayd::xcsp::run_to(&xml, verbose, &mut lock) {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}

fn is_instance(arg: &str) -> bool {
    Path::new(arg).is_file()
        || arg.ends_with(".xml")
        || arg.ends_with(".lzma")
        || arg.ends_with(".xz")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verbose = args.iter().any(|a| a == "-v" || a == "--verbose");
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();

    match positional.first() {
        Some(arg) if is_instance(arg) => run_instance(arg, verbose),
        Some(arg) => match arg.parse::<i32>() {
            Ok(n) => run_queens(n, verbose),
            Err(_) => {
                eprintln!("usage: qayd [-v] <instance.xml[.lzma|.xz]> | qayd [-v] <board-size>");
                std::process::exit(1);
            }
        },
        None => run_queens(8, verbose),
    }
}
