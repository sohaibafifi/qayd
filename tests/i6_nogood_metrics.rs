//! Fixed four-arm I6 pilot. The ignored sweep must run alone.
use qayd::orchestrator::{
    solve_model_silent, CpControls, ProvenConclusion, SearchPhase, SearchPolicy, SolveMode, SolveRequest, SolveResult, SolveStatus,
    ValueSelector, VariableSelector, VerificationLevel,
};
use qayd::SolveStats;
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU64, Ordering::Relaxed};
use std::time::Instant;
#[path = "common/nogood_models.rs"]
pub mod nogood_models;
use nogood_models::{permutation, pigeonhole, ModelCase};

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_DELTA: AtomicIsize = AtomicIsize::new(0);
static PEAK_DELTA: AtomicIsize = AtomicIsize::new(0);

struct CountingAllocator;

fn changed_live(delta: isize) {
    let live = LIVE_DELTA.fetch_add(delta, Relaxed) + delta;
    PEAK_DELTA.fetch_max(live, Relaxed);
}

// The ignored sweep runs alone with --test-threads=1. During normal tests this
// wrapper only forwards allocations. No solver state depends on these counters.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc(layout);
        if !pointer.is_null() && COUNTING.load(Relaxed) {
            ALLOCATIONS.fetch_add(1, Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Relaxed);
            changed_live(layout.size() as isize);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc_zeroed(layout);
        if !pointer.is_null() && COUNTING.load(Relaxed) {
            ALLOCATIONS.fetch_add(1, Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Relaxed);
            changed_live(layout.size() as isize);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if COUNTING.load(Relaxed) {
            changed_live(-(layout.size() as isize));
        }
        System.dealloc(pointer, layout);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = System.realloc(pointer, layout, size);
        if !result.is_null() && COUNTING.load(Relaxed) {
            REALLOCATIONS.fetch_add(1, Relaxed);
            ALLOCATED_BYTES.fetch_add(size.saturating_sub(layout.size()) as u64, Relaxed);
            changed_live(size as isize - layout.size() as isize);
        }
        result
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn begin_allocations() {
    assert!(!COUNTING.load(Relaxed));
    ALLOCATIONS.store(0, Relaxed);
    REALLOCATIONS.store(0, Relaxed);
    ALLOCATED_BYTES.store(0, Relaxed);
    LIVE_DELTA.store(0, Relaxed);
    PEAK_DELTA.store(0, Relaxed);
    COUNTING.store(true, Relaxed);
}

fn end_allocations() -> [u64; 4] {
    COUNTING.store(false, Relaxed);
    [ALLOCATIONS.load(Relaxed), REALLOCATIONS.load(Relaxed), ALLOCATED_BYTES.load(Relaxed), PEAK_DELTA.load(Relaxed).max(0) as u64]
}

const ARMS: [(bool, bool, &str); 4] = [(false, false, "baseline"), (true, false, "extended"), (false, true, "cpip"), (true, true, "both")];

fn request(case: &ModelCase, seed: u64, extended: bool, cpip: bool) -> SolveRequest {
    SolveRequest {
        mode: SolveMode::Exact,
        seed,
        threads: 1,
        cp: CpControls { extended_nogoods: extended, cpip, ..CpControls::default() },
        search_policy: SearchPolicy::new(vec![SearchPhase::new(
            (0..case.variable_count).collect(),
            VariableSelector::Activity,
            ValueSelector::RandomSeeded,
        )]),
        ..SolveRequest::default()
    }
}

fn validate(case: &ModelCase, result: &SolveResult) {
    result.validate_contract().unwrap();
    assert!(!case.objective);
    if case.feasible {
        assert_eq!(result.status(), SolveStatus::Satisfiable);
        let candidate = result.primal().unwrap();
        assert_eq!(candidate.verification(), VerificationLevel::Final);
        let values = candidate.assignment().integers.iter().map(|v| i32::try_from(v.unwrap()).unwrap()).collect::<Vec<_>>();
        assert!(case.accepts(&values));
    } else {
        assert_eq!(result.status(), SolveStatus::Unsatisfiable);
        assert!(result.primal().is_none());
        assert_eq!(result.proof().map(|p| p.conclusion()), Some(ProvenConclusion::Unsatisfiable));
    }
}

fn counters(s: SolveStats) -> BTreeMap<&'static str, u64> {
    let n = s.nogoods;
    BTreeMap::from([
        ("nodes", s.nodes),
        ("failures", s.failures),
        ("learned_literals", s.learned_lits),
        ("atoms", s.lcg_atoms),
        ("propagator_calls", s.propagator_calls),
        ("binary_visits", s.binary_clause_visits),
        ("watched_visits", s.watched_clause_visits),
        ("watched_scans", s.watched_literal_scans),
        ("extended_scans", n.extended_scans),
        ("clause_visits", n.clause_visits),
        ("predicate_visits", n.predicate_visits),
        ("value_visits", n.value_visits),
        ("propagations", n.propagations),
        ("removed_values", n.removed_values),
        ("reasons", n.reasons),
        ("reason_literals", n.reason_literals),
        ("conflicts", n.conflicts),
        ("skipped_clause_limit", n.skipped_clause_limit),
        ("skipped_domain_limit", n.skipped_domain_limit),
        ("interrupted", n.interrupted),
        ("cpip_attempts", n.cpip_attempts),
        ("cpip_resolution_steps", n.cpip_resolution_steps),
        ("cpip_learned", n.cpip_learned),
        ("cpip_multi_predicate", n.cpip_multi_predicate),
        ("cpip_asserting_predicates", n.cpip_asserting_predicates),
        ("cpip_fallbacks", n.cpip_fallbacks),
        ("peak_scratch_bytes", n.peak_scratch_bytes),
        ("skipped_proof_logging", n.skipped_proof_logging),
    ])
}

#[test]
#[ignore]
fn nogood_metrics_sweep() {
    for family in ["permutation", "pigeonhole"] {
        let sizes = if family == "permutation" { [4, 5, 6] } else { [2, 3, 4] };
        for size in sizes {
            for impossible in [false, true] {
                let case = if family == "permutation" { permutation(size, impossible, false) } else { pigeonhole(size, impossible, false) };
                for (seed_index, seed) in [0, 1, 7].into_iter().enumerate() {
                    let mut expected = BTreeMap::new();
                    for &(extended, cpip, arm) in &ARMS {
                        let options = request(&case, seed, extended, cpip);
                        let result = solve_model_silent(&case.package, &options).unwrap();
                        validate(&case, &result);
                        expected.insert(arm, counters(result.aggregate_search_stats()));
                    }
                    for repetition in 0..3 {
                        for offset in 0..4 {
                            let (extended, cpip, arm) = ARMS[(seed_index + repetition + offset) % 4];
                            let options = request(&case, seed, extended, cpip);
                            let started = Instant::now();
                            let result = solve_model_silent(&case.package, &options);
                            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                            let result = result.unwrap();
                            validate(&case, &result);
                            let metrics = counters(result.aggregate_search_stats());
                            assert_eq!(
                                expected[arm], metrics,
                                "nondeterministic timing trajectory {family} {size} {impossible} {seed} {arm}"
                            );
                            eprintln!(
                                r#"I6_TIMING {{"family":"{family}","size":{size},"impossible":{impossible},"seed":{seed},"arm":"{arm}","repetition":{repetition},"status":"{:?}","elapsed_ms":{elapsed_ms},"counters":{metrics:?}}}"#,
                                result.status()
                            );
                        }
                    }
                    for offset in 0..4 {
                        let (extended, cpip, arm) = ARMS[(seed_index + offset) % 4];
                        let options = request(&case, seed, extended, cpip);
                        begin_allocations();
                        let result = solve_model_silent(&case.package, &options);
                        let allocations = end_allocations();
                        let result = result.unwrap();
                        validate(&case, &result);
                        let metrics = counters(result.aggregate_search_stats());
                        assert_eq!(expected[arm], metrics, "different allocation trajectory {family} {size} {impossible} {seed} {arm}");
                        eprintln!(
                            r#"I6_ALLOC {{"family":"{family}","size":{size},"impossible":{impossible},"seed":{seed},"arm":"{arm}","status":"{:?}","allocation_calls":{},"reallocation_calls":{},"allocated_growth_bytes":{},"peak_extra_live_bytes":{},"counters":{metrics:?}}}"#,
                            result.status(),
                            allocations[0],
                            allocations[1],
                            allocations[2],
                            allocations[3]
                        );
                    }
                }
            }
        }
    }
}
