use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use crate::engines::sat::proof::{ProofFormat, ProofWriter};
use crate::engines::sat::{solve_cnf_native_seeded_options, solve_cnf_native_seeded_with_proof_options, Cnf, PreprocessOptions, Status};

fn balanced_ring_cnf(vars: usize) -> Cnf {
    let mut clauses = Vec::with_capacity(vars * 2);
    for idx in 0..vars {
        let a = idx as i32 + 1;
        let b = (idx + 1) as i32 % vars as i32 + 1;
        let c = (idx + 7) as i32 % vars as i32 + 1;
        clauses.push(vec![a, b, c]);
        clauses.push(vec![-a, -b, -c]);
    }
    Cnf { vars, clauses }
}

#[test]
fn full_preprocessing_honors_an_existing_stop_request() {
    let stop = AtomicBool::new(true);
    let cnf = balanced_ring_cnf(4_000);

    let result = solve_cnf_native_seeded_options(&cnf, &stop, 0, PreprocessOptions::full());

    assert_eq!(result.status, Status::Unknown);
    assert_eq!(result.preprocess.rounds, 0);
}

#[test]
fn full_preprocessing_is_interruptible_during_a_quadratic_pass() {
    let stop = Arc::new(AtomicBool::new(false));
    let cnf = balanced_ring_cnf(6_000);
    let timer_stop = Arc::clone(&stop);
    let timer = thread::spawn(move || {
        thread::sleep(Duration::from_millis(10));
        timer_stop.store(true, Ordering::Relaxed);
    });

    let started = Instant::now();
    let result = solve_cnf_native_seeded_options(&cnf, &stop, 0, PreprocessOptions::full());
    let elapsed = started.elapsed();
    timer.join().unwrap();

    assert_eq!(result.status, Status::Unknown);
    assert!(elapsed < Duration::from_secs(2), "preprocessing stopped after {elapsed:?}");
}

#[test]
fn proof_preprocessing_honors_stop_and_flushes_the_writer() {
    let stop = AtomicBool::new(true);
    let cnf = balanced_ring_cnf(4_000);
    let mut writer = ProofWriter::new(Vec::new(), ProofFormat::Drat);

    let result = solve_cnf_native_seeded_with_proof_options(&cnf, &stop, 0, PreprocessOptions::full(), &mut writer).unwrap();

    assert_eq!(result.status, Status::Unknown);
    assert_eq!(result.preprocess.rounds, 0);
    assert!(writer.into_inner().is_empty());
}
