//! `qayd` CLI: `qayd [-h] [-v] [-t SECONDS] [--seed SEED] [-p THREADS] [--split] [--probe N] <instance.xml[.lzma|.xz]>`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

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

fn run_instance(
    path: &str,
    verbose: bool,
    stop: &AtomicBool,
    seed: u64,
    workers: usize,
    split: bool,
    probes: usize,
) {
    let xml = match read_instance(path) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let options = qayd::xcsp::RunOptions {
        seed,
        workers,
        split,
        probes,
    };
    if let Err(e) = qayd::xcsp::run_to_with_options(&xml, verbose, stop, &mut lock, options) {
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

const USAGE: &str =
    "usage: qayd [-h] [-v] [-t SECONDS] [--seed SEED] [-p THREADS] [--split] [--probe N] <instance.xml[.lzma|.xz]>";

fn usage() -> ! {
    eprintln!("{USAGE}");
    std::process::exit(1);
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
    let mut split = false;
    let mut probes = 0;
    let mut path: Option<String> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => help(),
            "-v" | "--verbose" => verbose = true,
            "-t" | "--time" => match it.next().and_then(|s| s.parse::<u64>().ok()) {
                Some(secs) => time_limit = Some(secs),
                None => {
                    eprintln!("-t/--time needs a number of seconds");
                    std::process::exit(1);
                }
            },
            "--seed" => match it.next().and_then(|s| s.parse::<u64>().ok()) {
                Some(value) => seed = Some(value),
                None => {
                    eprintln!("--seed needs an unsigned integer");
                    std::process::exit(1);
                }
            },
            "-p" | "--threads" => match it.next().and_then(|s| s.parse::<usize>().ok()) {
                Some(value) if value > 0 => workers = Some(value),
                _ => {
                    eprintln!("-p/--threads needs a positive integer");
                    std::process::exit(1);
                }
            },
            "--split" => split = true,
            "--probe" => match it.next().and_then(|s| s.parse::<usize>().ok()) {
                Some(value) if value > 0 => probes = value,
                _ => {
                    eprintln!("--probe needs a positive integer");
                    std::process::exit(1);
                }
            },
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
    let seed = seed.unwrap_or_else(|| {
        std::env::var("RANDOMSEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    });
    let workers = workers.unwrap_or_else(|| match std::env::var("NBCORE") {
        Ok(s) => match s.parse::<usize>() {
            Ok(0) | Err(_) => {
                eprintln!("NBCORE needs a positive integer");
                std::process::exit(1);
            }
            Ok(n) => n,
        },
        Err(_) => 1,
    });

    let stop = Arc::new(AtomicBool::new(false));
    {
        let s = stop.clone();
        let _ = ctrlc::set_handler(move || s.store(true, Ordering::SeqCst));
    }
    if let Some(secs) = time_limit {
        let s = stop.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(secs));
            s.store(true, Ordering::SeqCst);
        });
    }

    run_instance(&path, verbose, &stop, seed, workers, split, probes);
}
