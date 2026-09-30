//! Run the frozen I7 pilot. Build/run is coordinated by the research parent.
//! Usage: cp_certificate_pilot --out NEW_DIRECTORY [--repetitions 3]

use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use qayd::certificate::{solve_with_certificate, CertificateOptions, CertificateStats};
use qayd::model::ModelPackage;
use qayd::orchestrator::{solve_model_silent, SolveMode, SolveRequest, SolveResult};

#[path = "../tests/common/certificate_models.rs"]
mod certificate_models;
use certificate_models::{all_cases, Case};

#[derive(Clone, Copy)]
enum Arm {
    Baseline,
    Certificate,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Certificate => "certificate",
        }
    }
}

struct Sample {
    status: String,
    objective: Option<i64>,
    elapsed_ms: f64,
    stats: CertificateStats,
    accepted: bool,
    artifact_directory: Option<PathBuf>,
    errors: Vec<String>,
}

fn measure(case: Case, package: &ModelPackage, request: &SolveRequest, arm: Arm, directory: &Path) -> Sample {
    let options = CertificateOptions::new(directory);
    let stop = Arc::new(AtomicBool::new(false));
    let mut stats = CertificateStats::default();
    let mut errors = Vec::new();
    let mut accepted = false;
    let start = Instant::now();
    let (result, elapsed_ms): (Option<SolveResult>, f64) = match arm {
        Arm::Baseline => {
            let outcome = solve_model_silent(package, request);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            stats.solve_seconds = elapsed / 1000.0;
            let result = match outcome {
                Ok(result) => Some(result),
                Err(error) => {
                    errors.push(error.to_string());
                    None
                }
            };
            (result, elapsed)
        }
        Arm::Certificate => {
            let outcome = solve_with_certificate(package, request, &options, stop);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let result = match outcome {
                Ok(run) => {
                    stats = run.stats;
                    match run.certificate {
                        Ok(checked) => {
                            accepted = true;
                            let claim = match case.expected_status() {
                                qayd::orchestrator::SolveStatus::Optimal => "optimal",
                                qayd::orchestrator::SolveStatus::Unsatisfiable => "unsat",
                                _ => "sat",
                            };
                            if checked.claim().token() != claim {
                                errors.push(format!("expected claim {claim}, got {}", checked.claim().token()));
                            }
                            if !checked.assumptions().is_empty() || !checked.model_path().is_file() || !checked.proof_path().is_file() {
                                errors.push("invalid unscoped artifact ownership".to_string());
                            }
                        }
                        Err(error) => errors.push(format!("certificate: {error}")),
                    }
                    Some(run.result)
                }
                Err(error) => {
                    errors.push(format!("wrapper: {error}"));
                    None
                }
            };
            (result, elapsed)
        }
    };
    // All exhaustive oracle work and full witness validation are outside the
    // wall-clock interval above, for both arms and warmups.
    let (status, objective) = if let Some(result) = result {
        if let Err(error) = case.validate(&result) {
            errors.push(error);
        }
        (result.status().as_str().to_string(), result.primal().and_then(|candidate| candidate.objectives().first().copied()))
    } else {
        ("ERROR".to_string(), None)
    };
    Sample {
        status,
        objective,
        elapsed_ms,
        stats,
        accepted,
        artifact_directory: matches!(arm, Arm::Certificate).then(|| directory.to_path_buf()),
        errors,
    }
}

fn json_string(value: &str) -> String {
    let mut output = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{1f}' => write!(output, "\\u{:04x}", u32::from(character)).unwrap(),
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

fn emit(case: Case, seed: u64, arm: Arm, repetition: usize, sample: &Sample) {
    let objective = sample.objective.map_or_else(|| "null".to_string(), |value| value.to_string());
    let directory = sample.artifact_directory.as_ref().map_or_else(|| "null".to_string(), |path| json_string(&path.to_string_lossy()));
    let errors = sample.errors.iter().map(|error| json_string(error)).collect::<Vec<_>>().join(",");
    println!(
        "I7_TIMING {{\"family\":{},\"size\":{},\"seed\":{},\"arm\":{},\"repetition\":{},\"status\":{},\"objective\":{},\"elapsed_ms\":{:.9},\"solve_ms\":{:.9},\"generation_ms\":{:.9},\"checker_ms\":{:.9},\"proof_bytes\":{},\"model_bytes\":{},\"nodes\":{},\"steps\":{},\"accepted\":{},\"artifact_directory\":{},\"errors\":[{}]}}",
        json_string(case.family),
        case.size,
        seed,
        json_string(arm.name()),
        repetition,
        json_string(&sample.status),
        objective,
        sample.elapsed_ms,
        sample.stats.solve_seconds * 1000.0,
        sample.stats.generation_seconds * 1000.0,
        sample.stats.checker_seconds * 1000.0,
        sample.stats.proof_bytes,
        sample.stats.model_bytes,
        sample.stats.nodes,
        sample.stats.steps,
        sample.accepted,
        directory,
        errors,
    );
}

fn run() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let mut output = None;
    let mut repetitions = 3usize;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--out" if output.is_none() => output = Some(PathBuf::from(arguments.next().ok_or("--out requires a directory")?)),
            "--repetitions" => {
                repetitions =
                    arguments.next().ok_or("--repetitions requires a positive integer")?.parse().map_err(|_| "invalid repetition count")?;
                if repetitions == 0 {
                    return Err("repetitions must be positive".to_string());
                }
            }
            _ => return Err(format!("unknown or repeated argument {argument}")),
        }
    }
    let output = output.ok_or("usage: cp_certificate_pilot --out NEW_DIRECTORY [--repetitions 3]")?;
    fs::create_dir(&output).map_err(|error| format!("create new output directory {}: {error}", output.display()))?;
    let output = output.canonicalize().map_err(|error| error.to_string())?;
    let warmups = output.join("warmup");
    let timings = output.join("timing");
    fs::create_dir(&warmups).map_err(|error| error.to_string())?;
    fs::create_dir(&timings).map_err(|error| error.to_string())?;
    let mut rows = 0usize;
    let mut failures = 0usize;
    let cases = all_cases();
    for case in &cases {
        let objective = case.expected_objective().map_or_else(|| "null".to_string(), |value| value.to_string());
        println!(
            "I7_INSTANCE {{\"family\":{},\"size\":{},\"expected_status\":{},\"expected_objective\":{},\"oracle_solutions\":{}}}",
            json_string(case.family),
            case.size,
            json_string(case.expected_status().as_str()),
            objective,
            case.solutions().len(),
        );
    }
    for (case_index, case) in cases.into_iter().enumerate() {
        let package = case.package();
        for (seed_index, seed) in [0, 1, 7].into_iter().enumerate() {
            let request = SolveRequest { mode: SolveMode::Exact, threads: 1, seed, ..SolveRequest::default() };
            let name = format!("{}-{}-seed{seed}", case.family, case.size);
            for arm in [Arm::Baseline, Arm::Certificate] {
                let sample = measure(case, &package, &request, arm, &warmups.join(&name));
                if !sample.errors.is_empty() {
                    failures += 1;
                    eprintln!("I7_WARMUP_ERROR {name} {} {:?}", arm.name(), sample.errors);
                }
            }
            for repetition in 0..repetitions {
                let arms = if (case_index + seed_index + repetition) % 2 == 0 {
                    [Arm::Baseline, Arm::Certificate]
                } else {
                    [Arm::Certificate, Arm::Baseline]
                };
                for arm in arms {
                    let sample = measure(case, &package, &request, arm, &timings.join(format!("{name}-rep{repetition}")));
                    failures += usize::from(!sample.errors.is_empty());
                    emit(case, seed, arm, repetition, &sample);
                    rows += 1;
                }
            }
        }
    }
    let expected_rows = 15 * 3 * 2 * repetitions;
    if rows != expected_rows {
        return Err(format!("expected {expected_rows} timing rows, got {rows}"));
    }
    println!("I7_COMPLETE {{\"timing_rows\":{rows},\"failures\":{failures},\"repetitions\":{repetitions}}}");
    if failures != 0 {
        return Err(format!("{failures} measured or warmup runs failed validation; all rows and artifacts were retained"));
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("I7_ERROR {error}");
        std::process::exit(1);
    }
}
