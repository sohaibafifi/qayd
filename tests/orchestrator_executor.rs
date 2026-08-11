use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use qayd::orchestrator::{execute_workers, execute_workers_silent, merge_search_stats, EventControl};
use qayd::SolveStats;

#[test]
fn worker_reports_are_lossless_and_ordered_by_worker() {
    let stop = AtomicBool::new(false);
    let cancel = Arc::new(AtomicBool::new(false));
    let summary = execute_workers_silent(vec![4_u64, 3, 2, 1], &stop, cancel, 41, |context, delay| {
        std::thread::sleep(Duration::from_millis(delay));
        (context.worker(), context.seed())
    });

    let reports = summary.reports.into_iter().map(|report| (report.worker, report.seed, report.result)).collect::<Vec<_>>();
    assert_eq!(reports, vec![(0, 41, (0, 41)), (1, 42, (1, 42)), (2, 43, (2, 43)), (3, 44, (3, 44))]);
}

#[test]
fn progress_is_coalesced_but_keeps_the_latest_value() {
    let stop = AtomicBool::new(false);
    let cancel = Arc::new(AtomicBool::new(false));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let handler_seen = Arc::clone(&seen);

    let summary = execute_workers(
        vec![()],
        &stop,
        cancel,
        0,
        |context, ()| {
            for value in 0..1_000 {
                assert!(context.publish_latest(value));
            }
        },
        move |event| {
            handler_seen.lock().unwrap().push(event.payload);
            std::thread::sleep(Duration::from_millis(1));
            Ok::<_, ()>(EventControl::Continue)
        },
    )
    .unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(seen.last(), Some(&999));
    assert!(seen.len() < 1_000);
    assert_eq!(summary.reports.len(), 1);
}

#[test]
fn handler_stop_cancels_the_worker_set() {
    let stop = AtomicBool::new(false);
    let cancel = Arc::new(AtomicBool::new(false));
    let observed_cancel = Arc::clone(&cancel);

    let summary = execute_workers(
        vec![()],
        &stop,
        cancel,
        0,
        |context, ()| {
            context.publish_latest(());
            while !context.is_cancelled() {
                std::thread::yield_now();
            }
        },
        |_event| Ok::<_, ()>(EventControl::Stop),
    )
    .unwrap();

    assert!(summary.stopped_by_handler);
    assert!(observed_cancel.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(summary.reports.len(), 1);
}

#[test]
fn merge_search_stats_covers_every_counter() {
    let mut total = SolveStats {
        solutions: 1,
        nodes: 2,
        failures: 3,
        learned_lits: 4,
        vivified_clauses: 5,
        vivified_lits: 6,
        binary_clause_visits: 7,
        watched_clause_visits: 8,
        watched_literal_scans: 9,
        binary_implications: 10,
        lp_rows: 11,
        lp_solves: 12,
        lp_certified: 13,
        lp_timeouts: 14,
        lp_refactorizations: 15,
        lp_micros: 16,
        lp_root_bound: Some(17),
        lp_node_prunes: 18,
    };
    let part = SolveStats {
        solutions: 10,
        nodes: 20,
        failures: 30,
        learned_lits: 40,
        vivified_clauses: 50,
        vivified_lits: 60,
        binary_clause_visits: 70,
        watched_clause_visits: 80,
        watched_literal_scans: 90,
        binary_implications: 100,
        lp_rows: 110,
        lp_solves: 120,
        lp_certified: 130,
        lp_timeouts: 140,
        lp_refactorizations: 150,
        lp_micros: 160,
        lp_root_bound: Some(170),
        lp_node_prunes: 180,
    };

    merge_search_stats(&mut total, part);

    assert_eq!(
        total,
        SolveStats {
            solutions: 11,
            nodes: 22,
            failures: 33,
            learned_lits: 44,
            vivified_clauses: 55,
            vivified_lits: 66,
            binary_clause_visits: 77,
            watched_clause_visits: 88,
            watched_literal_scans: 99,
            binary_implications: 110,
            lp_rows: 121,
            lp_solves: 132,
            lp_certified: 143,
            lp_timeouts: 154,
            lp_refactorizations: 165,
            lp_micros: 176,
            lp_root_bound: Some(17),
            lp_node_prunes: 198,
        }
    );
}
