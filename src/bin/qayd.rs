//! `qayd` CLI.
//!
//! `qayd [-v] [-t SECONDS] <instance.xml[.lzma|.xz]>`
//!
//! - `-v` / `--verbose` — emit `c` comment lines (model size, improving bounds,
//!   search statistics, wall-clock time).
//! - `-t` / `--time SECONDS` — stop after a time budget, reporting the best
//!   solution found so far. Ctrl+C does the same on demand.

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

fn run_instance(path: &str, verbose: bool, stop: &AtomicBool) {
    let xml = match read_instance(path) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    if let Err(e) = qayd::xcsp::run_to(&xml, verbose, stop, &mut lock) {
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

fn usage() -> ! {
    eprintln!("usage: qayd [-v] [-t SECONDS] <instance.xml[.lzma|.xz]>");
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut verbose = false;
    let mut time_limit: Option<u64> = None;
    let mut path: Option<String> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-v" | "--verbose" => verbose = true,
            "-t" | "--time" => match it.next().and_then(|s| s.parse::<u64>().ok()) {
                Some(secs) => time_limit = Some(secs),
                None => {
                    eprintln!("-t/--time needs a number of seconds");
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

    // Shared stop flag, set by Ctrl+C and (optionally) the time-limit thread.
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

    run_instance(&path, verbose, &stop);
}
