//! `qayd` CLI for XCSP3 instances.

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

fn run_instance(
    path: &str,
    verbose: bool,
    stop: &AtomicBool,
    options: qayd::frontends::xcsp::RunOptions,
    search_policy: qayd::orchestrator::SearchPolicy,
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
    if let Err(e) = qayd::frontends::xcsp::run_to_with_options_and_search_policy(&xml, verbose, stop, &mut lock, options, search_policy) {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}

fn is_instance(arg: &str) -> bool {
    Path::new(arg).is_file() || arg.ends_with(".xml") || arg.ends_with(".lzma") || arg.ends_with(".xz")
}

const USAGE: &str = r#"Qayd XCSP3 solver

Usage:
  qayd [OPTIONS] <INSTANCE>

Arguments:
  <INSTANCE>                         XCSP3 file (.xml, .xml.lzma, or .xml.xz)

General options:
  -h, --help                         Show this help and exit
  -v, --verbose                      Print progress and search statistics
  -t, --time <SECONDS>               Set the wall-clock time limit
                                     [default: unlimited; --ls: 240]
      --seed <SEED>                  Set the reproducible search seed [default: 0]
  -p, --threads <N>                  Set the worker count [default: 1; N >= 1]
      --mem-limit <MiB>              Set the process memory limit [MiB >= 1]
      --core                         Print an exact source-constraint MUS after UNSAT

Search options:
      --ls                           Run incumbent-only local search
                                     Uses a 240-second limit when --time is omitted
      --split                        Enable split workers (COP, threads > 1)
      --probe <N>                    Reserve N probing workers [N >= 1]
      --lns <N>                      Reserve N LNS workers [N >= 1]
      --no-learn-csp                 Use chronological DFS for compatible CSPs
      --semantic-branching           Branch first on XCSP semantic variables
      --search-phase <SPEC>          Add an ordered exact-search phase; repeatable
      --force-scope-reasons          Use whole-scope LCG reasons (ablation)
      --shared-pool-cap <N>          Set the shared learned-clause capacity
                                     [default: 16384; N >= 1]

  Split and probing require materialized COP objectives.
  Split, probing, and LNS require exact mode and multiple threads.
  Probing and LNS roles must leave at least one complete worker.
  A parallel COP may allocate probing and LNS roles when counts are omitted.

Search phase format:
  SPEC      VARS:VARIABLE:VALUE
  VARS      Comma-separated zero-based semantic integer indices
  VARIABLE  auto | input-order | first-fail | max-regret | dom-wdeg | activity
  VALUE     auto | min | max | median | random-seeded | hint

  Unscoped variables are completed by the automatic fallback.
  Explicit search phases cannot be combined with --ls or --semantic-branching.
  Semantic branching requires exact mode; activity phases cannot use --no-learn-csp.

Linear relaxation options:
      --linear-backend <MODE>        Select auto, native, or amthal [default: auto]
      --lp-root-ms <MS>              Limit the root relaxation [default: 50; 0 disables]
      --lp-node-ms <MS>              Limit in-search LP reoptimization
                                     [default: 0, disabled]
      --lp-node-depth <N>            Set the minimum depth interval [default: 8; N >= 1]
      --lp-max-vars <N>              Limit active LP columns [default: 2000; N >= 1]
      --lp-max-rows <N>              Limit retained LP rows [default: 1000; N >= 1]
      --lp-max-nonzeros <N>          Limit retained LP nonzeros [default: 100000; N >= 1]
      --lp-min-coverage <PERCENT>    Require objective coverage [default: 1; 0..100]
      --lp-phase-max-vars <N>        Limit CP size for LP phase guidance
                                     [default: 1000; 0 disables]
      --lp-route-ng-size <N>         Set routing ng-neighborhood size [default: 8; 1..16]
      --lp-route-max-labels <N>      Set the routing pricing label limit
                                     [default: 2000000; N >= 1]
      --lp-route-dual-stabilization-percent <PERCENT>
                                     Set the restricted-master dual weight
                                     [default: 75; 1..100]

  The amthal backend requires a build with the lp-relaxation feature."#;

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

fn bounded_positive(value: Option<&str>, maximum: usize, message: &str) -> usize {
    let value = positive(value, message);
    if value > maximum {
        fail(message);
    }
    value
}

fn linear_backend(value: Option<&str>) -> qayd::orchestrator::LinearBackendMode {
    match value {
        Some("auto") => qayd::orchestrator::LinearBackendMode::Auto,
        Some("native") => qayd::orchestrator::LinearBackendMode::Native,
        Some("amthal") => qayd::orchestrator::LinearBackendMode::Amthal,
        _ => fail("--linear-backend must be auto, native, or amthal"),
    }
}

fn search_phase(value: Option<&str>) -> qayd::orchestrator::SearchPhase {
    let value = value.unwrap_or_else(|| fail("--search-phase needs VARS:VARIABLE:VALUE"));
    let parts = value.split(':').collect::<Vec<_>>();
    if parts.len() != 3 || parts[0].is_empty() {
        fail("--search-phase needs VARS:VARIABLE:VALUE with a non-empty comma-separated VARS scope");
    }
    let scope = parts[0]
        .split(',')
        .map(|variable| {
            variable.parse::<usize>().unwrap_or_else(|_| fail("--search-phase VARS must contain zero-based non-negative integer indices"))
        })
        .collect();
    let variable = qayd::orchestrator::VariableSelector::from_str(parts[1]).unwrap_or_else(|error| fail(&error));
    let value = qayd::orchestrator::ValueSelector::from_str(parts[2]).unwrap_or_else(|error| fail(&error));
    qayd::orchestrator::SearchPhase::new(scope, variable, value)
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
    let mut core = false;
    let mut semantic_branching = false;
    let mut force_scope_reasons = false;
    let mut shared_pool_capacity = 1 << 14;
    let mut mem_limit: Option<usize> = None;
    let mut linear = qayd::orchestrator::LinearControls::default();
    let mut search_phases = Vec::new();
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
            "--core" => core = true,
            "--semantic-branching" => semantic_branching = true,
            "--search-phase" => search_phases.push(search_phase(it.next().map(String::as_str))),
            "--force-scope-reasons" => force_scope_reasons = true,
            "--shared-pool-cap" => {
                shared_pool_capacity = positive(it.next().map(String::as_str), "--shared-pool-cap needs a positive integer")
            }
            "--mem-limit" => mem_limit = Some(positive(it.next().map(String::as_str), "--mem-limit needs a positive integer (megabytes)")),
            "--linear-backend" => linear.backend = linear_backend(it.next().map(String::as_str)),
            "--lp-root-ms" => {
                linear.root_time = Duration::from_millis(parse(it.next().map(String::as_str), "--lp-root-ms needs milliseconds"))
            }
            "--lp-node-ms" => {
                linear.node_time =
                    Duration::from_millis(parse(it.next().map(String::as_str), "--lp-node-ms needs non-negative milliseconds"))
            }
            "--lp-node-depth" => {
                linear.node_depth_interval = positive(it.next().map(String::as_str), "--lp-node-depth needs a positive integer")
            }
            "--lp-max-vars" => linear.max_variables = positive(it.next().map(String::as_str), "--lp-max-vars needs a positive integer"),
            "--lp-max-rows" => linear.max_rows = positive(it.next().map(String::as_str), "--lp-max-rows needs a positive integer"),
            "--lp-max-nonzeros" => {
                linear.max_nonzeros = positive(it.next().map(String::as_str), "--lp-max-nonzeros needs a positive integer")
            }
            "--lp-min-coverage" => {
                linear.min_coverage_percent = parse(it.next().map(String::as_str), "--lp-min-coverage needs a percentage")
            }
            "--lp-phase-max-vars" => {
                linear.phase_max_variables = parse(it.next().map(String::as_str), "--lp-phase-max-vars needs a non-negative integer")
            }
            "--lp-route-ng-size" => {
                linear.route_ng_size =
                    bounded_positive(it.next().map(String::as_str), 16, "--lp-route-ng-size needs an integer between 1 and 16")
            }
            "--lp-route-max-labels" => {
                linear.route_max_labels = positive(it.next().map(String::as_str), "--lp-route-max-labels needs a positive integer")
            }
            "--lp-route-dual-stabilization-percent" => {
                linear.route_dual_stabilization_percent = bounded_positive(
                    it.next().map(String::as_str),
                    100,
                    "--lp-route-dual-stabilization-percent needs an integer between 1 and 100",
                )
            }
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
    let search_policy = qayd::orchestrator::SearchPolicy::new(search_phases);
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
            linear,
            core,
        },
        search_policy,
    );
}
