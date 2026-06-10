//! `qayd-fzn`: a minimal FlatZinc front-end for the `qayd` solver.
//!
//! Reads a `.fzn` model, posts it onto a [`qayd::Solver`], searches, and prints
//! FlatZinc-style status / solution lines.

mod model;
mod parse;
mod solve;
mod text;

use std::time::Duration;

use crate::solve::Options;

fn main() {
    let mut path: Option<String> = None;
    let mut opts = Options { verbose: false, time_limit: None };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-v" | "--verbose" => opts.verbose = true,
            "-t" | "--time-limit" => {
                let sec = args.next().and_then(|s| s.parse::<u64>().ok()).unwrap_or_else(|| {
                    eprintln!("error: --time-limit expects a value in seconds");
                    std::process::exit(1);
                });
                opts.time_limit = Some(Duration::from_secs(sec));
            }
            "-h" | "--help" => {
                print_usage();
                return;
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

    let Some(path) = path else {
        print_usage();
        std::process::exit(1);
    };
    let input = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    match parse::parse(&input) {
        Ok(model) => solve::solve(model, &opts),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}

fn print_usage() {
    eprintln!(
        "usage: qayd-fzn [options] <model.fzn>\n\
         \n\
         options:\n\
         \x20 -v, --verbose           stream improving bounds and print statistics\n\
         \x20 -t, --time-limit <sec>  stop the search after <sec> seconds\n\
         \x20 -h, --help              show this help"
    );
}
