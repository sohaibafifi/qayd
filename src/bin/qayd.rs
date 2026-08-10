//! `qayd` CLI: `qayd [-h] [-v] [-t SECONDS] [--seed SEED] [-p THREADS] [--ls] [--split] [--probe N] [--lns N] [--no-learn-csp] [--semantic-branching] <instance.xml[.lzma|.xz]>`.

use std::path::Path;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Track live allocations process-wide so `--mem-limit` can act on real usage.
#[global_allocator]
static ALLOC: qayd::mem::Tracking = qayd::mem::Tracking;

/// Read an instance, transparently decompressing `.lzma` / `.xz`.
fn read_instance(path: &str) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let bytes = if path.ends_with(".lzma") {
        let mut out = Vec::new();
        lzma_rs::lzma_decompress(&mut &bytes[..], &mut out).map_err(|e| e.to_string())?;
        out
    } else if path.ends_with(".xz") {
        let mut out = Vec::new();
        lzma_rs::xz_decompress(&mut &bytes[..], &mut out).map_err(|e| e.to_string())?;
        out
    } else {
        bytes
    };
    String::from_utf8(bytes).map_err(|e| e.to_string())
}

fn run_instance(path: &str, verbose: bool, stop: &AtomicBool, options: qayd::frontends::xcsp::RunOptions) {
    let xml = match read_instance(path) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    if let Err(e) = qayd::frontends::xcsp::run_to_with_options(&xml, verbose, stop, &mut lock, options) {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}

fn is_instance(arg: &str) -> bool {
    Path::new(arg).is_file() || arg.ends_with(".xml") || arg.ends_with(".lzma") || arg.ends_with(".xz")
}

const USAGE: &str =
    "usage: qayd [-h] [-v] [-t SECONDS] [--seed SEED] [-p THREADS] [--mem-limit MB] [--ls] [--split] [--probe N] [--lns N] [--no-learn-csp] [--semantic-branching] [--force-scope-reasons] [--shared-pool-cap N] <instance.xml[.lzma|.xz]>";

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

fn parse<T: FromStr>(value: Option<&str>, message: &str) -> T {
    value.and_then(|s| s.parse().ok()).unwrap_or_else(|| fail(message))
}

fn positive(value: Option<&str>, message: &str) -> usize {
    let value = parse(value, message);
    if value == 0 {
        fail(message);
    }
    value
}

fn usage() -> ! {
    fail(USAGE)
}

fn help() -> ! {
    println!("{USAGE}");
    std::process::exit(0);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut verbose = false;
    let mut time_limit: Option<u64> = None;
    let mut seed: Option<u64> = None;
    let mut workers: Option<usize> = None;
    let mut ls = false;
    let mut split = false;
    let mut probes = 0;
    let mut lns = 0;
    let mut no_learn_csp = false;
    let mut semantic_branching = false;
    let mut force_scope_reasons = false;
    let mut shared_pool_capacity = 1 << 14;
    let mut mem_limit: Option<usize> = None;
    let mut path: Option<String> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => help(),
            "-v" | "--verbose" => verbose = true,
            "-t" | "--time" => time_limit = Some(parse(it.next().map(String::as_str), "-t/--time needs a number of seconds")),
            "--seed" => seed = Some(parse(it.next().map(String::as_str), "--seed needs an unsigned integer")),
            "-p" | "--threads" => workers = Some(positive(it.next().map(String::as_str), "-p/--threads needs a positive integer")),
            "--ls" => ls = true,
            "--split" => split = true,
            "--probe" => probes = positive(it.next().map(String::as_str), "--probe needs a positive integer"),
            "--lns" => lns = positive(it.next().map(String::as_str), "--lns needs a positive integer"),
            "--no-learn-csp" => no_learn_csp = true,
            "--semantic-branching" => semantic_branching = true,
            "--force-scope-reasons" => force_scope_reasons = true,
            "--shared-pool-cap" => {
                shared_pool_capacity = positive(it.next().map(String::as_str), "--shared-pool-cap needs a positive integer")
            }
            "--mem-limit" => mem_limit = Some(positive(it.next().map(String::as_str), "--mem-limit needs a positive integer (megabytes)")),
            other if other.starts_with('-') => {
                eprintln!("unknown option {other}");
                usage();
            }
            other if path.is_none() => path = Some(other.to_string()),
            _ => usage(),
        }
    }

    let Some(path) = path else { usage() };
    if !is_instance(&path) {
        usage();
    }
    let seed = seed.unwrap_or(0);
    if ls && time_limit.is_none() {
        time_limit = Some(240);
    }
    let workers = workers.unwrap_or(1);

    let stop = Arc::new(AtomicBool::new(false));
    {
        let s = stop.clone();
        let _ = ctrlc::set_handler(move || s.store(true, Ordering::SeqCst));
    }

    let mode = if ls { qayd::frontends::xcsp::Mode::Ls } else { qayd::frontends::xcsp::Mode::Default };
    run_instance(
        &path,
        verbose,
        &stop,
        qayd::frontends::xcsp::RunOptions {
            seed,
            workers,
            mode,
            split,
            probes,
            lns,
            no_learn_csp,
            semantic_branching,
            force_scope_reasons,
            shared_pool_capacity,
            time_limit: time_limit.map(Duration::from_secs),
            mem_limit,
        },
    );
}
