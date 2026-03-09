//! `qayd` CLI.
//!
//! - `qayd [-v] <instance.xml[.lzma|.xz]>` — parse and solve an XCSP3 instance.
//!
//! `-v` / `--verbose` adds `c` comment lines: model size, improving bounds,
//! search statistics, and wall-clock time.

use std::path::Path;

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
        Some(_) | None => {
            eprintln!("usage: qayd [-v] <instance.xml[.lzma|.xz]>");
            std::process::exit(1);
        }
    }
}
