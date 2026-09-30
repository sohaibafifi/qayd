//! Fixed I5 profile and direct-LCG pilots. Run the ignored sweep alone.
use crate::constraints::cumulative_profile::{CumulativeStats, IncrementalProfile, ProfileUpdate};
use crate::constraints::resource_profile::{build_profile, ProfileSegment};
use crate::{SearchControl, SolveStats, Store};
use std::alloc::{GlobalAlloc, Layout};
use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU64, Ordering::Relaxed};
use std::time::Instant;
#[path = "../common/cumulative_models.rs"]
#[allow(dead_code)]
mod cumulative_models;
use cumulative_models::Case;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_DELTA: AtomicIsize = AtomicIsize::new(0);
static PEAK_DELTA: AtomicIsize = AtomicIsize::new(0);

pub(crate) struct CountingAllocator;

fn changed_live(delta: isize) {
    let live = LIVE_DELTA.fetch_add(delta, Relaxed) + delta;
    PEAK_DELTA.fetch_max(live, Relaxed);
}

// The ignored sweep runs alone with --test-threads=1. During normal tests this
// wrapper only forwards allocations. No solver state depends on these counters.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = crate::mem::Tracking.alloc(layout);
        if !pointer.is_null() && COUNTING.load(Relaxed) {
            ALLOCATIONS.fetch_add(1, Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Relaxed);
            changed_live(layout.size() as isize);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = crate::mem::Tracking.alloc_zeroed(layout);
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
        crate::mem::Tracking.dealloc(pointer, layout);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = crate::mem::Tracking.realloc(pointer, layout, size);
        if !result.is_null() && COUNTING.load(Relaxed) {
            REALLOCATIONS.fetch_add(1, Relaxed);
            ALLOCATED_BYTES.fetch_add(size.saturating_sub(layout.size()) as u64, Relaxed);
            changed_live(size as isize - layout.size() as isize);
        }
        result
    }
}

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

fn counters(c: CumulativeStats, s: SolveStats) -> BTreeMap<&'static str, u64> {
    BTreeMap::from([
        ("propagation_calls", c.propagation_calls),
        ("profile_rebuilds", c.profile_rebuilds),
        ("incremental_updates", c.incremental_updates),
        ("task_comparisons", c.task_comparisons),
        ("changed_tasks", c.changed_tasks),
        ("endpoint_updates", c.endpoint_updates),
        ("segments_emitted", c.segments_emitted),
        ("undo_records", c.undo_records),
        ("restored_tasks", c.restored_tasks),
        ("peak_undo_records", c.peak_undo_records),
        ("peak_event_points", c.peak_event_points),
        ("fallback_calls", c.fallback_calls),
        ("interrupted", c.interrupted),
        ("reason_calls", c.reason_calls),
        ("original_premises", c.original_premises),
        ("emitted_premises", c.emitted_premises),
        ("lifted_bounds", c.lifted_bounds),
        ("omitted_tautologies", c.omitted_tautologies),
        ("timetable_updates", c.timetable_updates),
        ("timetable_conflicts", c.timetable_conflicts),
        ("nodes", s.nodes),
        ("failures", s.failures),
        ("learned_literals", s.learned_lits),
        ("atoms", s.lcg_atoms),
        ("propagator_calls", s.propagator_calls),
        ("binary_visits", s.binary_clause_visits),
        ("watched_visits", s.watched_clause_visits),
        ("watched_scans", s.watched_literal_scans),
    ])
}

const SEEDS: [u64; 3] = [0, 1, 7];
const ARMS: [(bool, bool, &str); 4] =
    [(false, false, "baseline"), (true, false, "incremental"), (false, true, "lifted"), (true, true, "both")];
static NEVER_STOP: AtomicBool = AtomicBool::new(false);
type Part = Option<(i128, i128, i128)>;

#[derive(Clone, Copy)]
enum Level {
    Stay,
    Push,
    Pop,
}
struct Checkpoint {
    level: Level,
    parts: Vec<Part>,
}
struct Trace {
    family: &'static str,
    size: usize,
    seed: u64,
    initial: Vec<Part>,
    steps: Vec<Checkpoint>,
}

impl Trace {
    fn new(family: &'static str, size: usize, seed: u64) -> Self {
        let initial = (0..size)
            .map(|i| {
                let start = -1_000_000_000 + ((i % 16) * 4) as i128;
                Some((start, start + 4, (i % 3 + 1) as i128))
            })
            .collect::<Vec<_>>();
        let mut state = initial.clone();
        let mut stack = Vec::new();
        let mut steps = Vec::new();
        if family != "rollback" {
            stack.push(state.clone());
            steps.push(Checkpoint { level: Level::Push, parts: state.clone() });
        }
        for step in 0..64 {
            let level = if family == "rollback" {
                if step % 8 < 4 {
                    stack.push(state.clone());
                    Level::Push
                } else {
                    state = stack.pop().unwrap();
                    Level::Pop
                }
            } else {
                Level::Stay
            };
            let indices = if family == "batch" { (0..size).collect::<Vec<_>>() } else { vec![(step * 17 + seed as usize * 7) % size] };
            for i in indices {
                let base = -1_000_000_000 + ((i % 16) * 4) as i128;
                let phase = (step + i + seed as usize) % 7;
                state[i] = if phase == 0 {
                    None
                } else {
                    let start = base + (phase % 3) as i128;
                    Some((start, start + (phase % 4 + 1) as i128, (i % 3 + 1) as i128))
                };
            }
            steps.push(Checkpoint { level, parts: state.clone() });
        }
        while let Some(restored) = stack.pop() {
            steps.push(Checkpoint { level: Level::Pop, parts: restored });
        }
        Self { family, size, seed, initial, steps }
    }
}

fn fingerprint(profile: &[ProfileSegment]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for segment in profile {
        for value in [segment.start, segment.end, segment.usage] {
            for byte in value.to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
    }
    hash
}

struct Observation {
    elapsed_ms: f64,
    counters: BTreeMap<&'static str, u64>,
    allocations: [u64; 4],
    solutions: u64,
    checksum: u64,
}

fn render_profile(
    store: &mut Store,
    cache: &mut Option<IncrementalProfile>,
    parts: &[Part],
    events: &mut Vec<(i128, i128)>,
    output: &mut Vec<ProfileSegment>,
    stats: &mut CumulativeStats,
) {
    if let Some(cache) = cache {
        assert_eq!(cache.update(store, parts.iter().copied(), stats, None), ProfileUpdate::Ready);
        assert_eq!(cache.materialize(store, output, stats, None), ProfileUpdate::Ready);
    } else {
        build_profile(parts.iter().flatten().copied(), events, output);
        stats.profile_rebuilds += 1;
        stats.endpoint_updates += events.len() as u64;
        stats.peak_event_points = stats.peak_event_points.max(
            events
                .iter()
                .map(|(time, _)| time)
                .fold((None, 0u64), |(previous, count), time| (Some(time), count + u64::from(previous != Some(time))))
                .1,
        );
        stats.segments_emitted += output.len() as u64;
    }
    black_box(&*output);
}

fn profile_run(trace: &Trace, incremental: bool, verify: bool, allocation: bool) -> Observation {
    // Expected profiles are constructed entirely outside the measured region.
    let expected = if verify {
        std::iter::once(&trace.initial)
            .chain(trace.steps.iter().map(|s| &s.parts))
            .map(|parts| {
                let mut events = Vec::new();
                let mut profile = Vec::new();
                build_profile(parts.iter().flatten().copied(), &mut events, &mut profile);
                profile
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    if allocation {
        begin_allocations();
    }
    let started = Instant::now();
    let mut store = Store::new();
    let mut cache = if incremental { IncrementalProfile::new(&mut store, trace.size) } else { None };
    assert_eq!(cache.is_some(), incremental);
    let mut events = Vec::new();
    let mut output = Vec::new();
    let mut stats = CumulativeStats::default();
    render_profile(&mut store, &mut cache, &trace.initial, &mut events, &mut output, &mut stats);
    if verify {
        assert_eq!(output, expected[0]);
    }
    for (i, step) in trace.steps.iter().enumerate() {
        match step.level {
            Level::Stay => {}
            Level::Push => store.push_level(),
            Level::Pop => store.pop_level(),
        }
        render_profile(&mut store, &mut cache, &step.parts, &mut events, &mut output, &mut stats);
        if verify {
            assert_eq!(output, expected[i + 1], "{} {} {} step{i}", trace.family, trace.size, trace.seed);
        }
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
    let allocations = if allocation { end_allocations() } else { [0; 4] };
    Observation { elapsed_ms, counters: counters(stats, SolveStats::default()), allocations, solutions: 0, checksum: fingerprint(&output) }
}

fn search_run(
    case: &Case,
    seed: u64,
    incremental: bool,
    lifted: bool,
    expected: Option<&BTreeSet<Vec<i32>>>,
    allocation: bool,
) -> Observation {
    let (mut solver, variables) = case.solver(incremental, lifted, true);
    let mut found = BTreeSet::new();
    let mut solutions = 0u64;
    if allocation {
        begin_allocations();
    }
    let started = Instant::now();
    let stats = crate::search::solve_interruptible_seeded(
        &mut solver,
        &variables,
        |state| {
            solutions += 1;
            if expected.is_some() {
                found.insert(variables.iter().map(|&v| state.store.value(v)).collect::<Vec<_>>());
            }
            SearchControl::Continue
        },
        &NEVER_STOP,
        seed,
    );
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
    let allocations = if allocation { end_allocations() } else { [0; 4] };
    if let Some(expected) = expected {
        assert_eq!(&found, expected, "{} {} {} seed{seed}", case.family, case.n, case.capacity);
    }
    Observation { elapsed_ms, counters: counters(solver.cumulative_stats(), stats), allocations, solutions, checksum: 0 }
}

struct Key<'a> {
    kind: &'static str,
    family: &'a str,
    size: usize,
    capacity: i32,
    seed: u64,
    arm: &'static str,
}
fn emit(key: &Key<'_>, row: &Observation, repetition: Option<usize>) {
    let shared = format!(
        r#""kind":"{}","family":"{}","size":{},"capacity":{},"seed":{},"arm":"{}","solutions":{},"checksum":{},"counters":{:?}"#,
        key.kind, key.family, key.size, key.capacity, key.seed, key.arm, row.solutions, row.checksum, row.counters
    );
    if let Some(repetition) = repetition {
        println!(r#"I5_TIMING {{{shared},"repetition":{repetition},"elapsed_ms":{}}}"#, row.elapsed_ms);
    } else {
        println!(
            r#"I5_ALLOC {{{shared},"allocation_calls":{},"reallocation_calls":{},"allocated_growth_bytes":{},"peak_extra_live_bytes":{}}}"#,
            row.allocations[0], row.allocations[1], row.allocations[2], row.allocations[3]
        );
    }
}
fn consistent(reference: &Observation, row: &Observation) {
    assert_eq!(reference.counters, row.counters);
    assert_eq!(reference.solutions, row.solutions);
    assert_eq!(reference.checksum, row.checksum);
    assert_eq!(row.counters["interrupted"], 0);
    assert_eq!(row.counters["fallback_calls"], 0);
}

#[test]
#[ignore]
fn i5_metrics_sweep() {
    for family in ["single", "batch", "rollback"] {
        for size in [16, 64, 256] {
            for seed in SEEDS {
                let trace = Trace::new(family, size, seed);
                let reference = [profile_run(&trace, false, true, false), profile_run(&trace, true, true, false)];
                assert_eq!(reference[0].checksum, reference[1].checksum);
                for repetition in 0..3 {
                    for offset in 0..2 {
                        let index = (seed as usize + repetition + offset) % 2;
                        let row = profile_run(&trace, index == 1, false, false);
                        consistent(&reference[index], &row);
                        emit(
                            &Key {
                                kind: "profile",
                                family,
                                size,
                                capacity: 0,
                                seed,
                                arm: if index == 1 { "incremental" } else { "baseline" },
                            },
                            &row,
                            Some(repetition),
                        );
                    }
                }
                for (index, baseline) in reference.iter().enumerate() {
                    let row = profile_run(&trace, index == 1, false, true);
                    consistent(baseline, &row);
                    emit(
                        &Key { kind: "profile", family, size, capacity: 0, seed, arm: if index == 1 { "incremental" } else { "baseline" } },
                        &row,
                        None,
                    );
                }
            }
        }
    }
    for family in ["dense", "anchored"] {
        for size in [4, 5, 6] {
            for capacity in [2, 3] {
                let case = Case::new(family, size, capacity);
                let expected = case.solutions();
                for seed in SEEDS {
                    let reference =
                        ARMS.iter().map(|&(inc, lift, _)| search_run(&case, seed, inc, lift, Some(&expected), false)).collect::<Vec<_>>();
                    for row in &reference {
                        assert_eq!(row.solutions, expected.len() as u64);
                    }
                    for repetition in 0..3 {
                        for offset in 0..4 {
                            let index = (seed as usize + repetition + offset) % 4;
                            let (inc, lift, arm) = ARMS[index];
                            let row = search_run(&case, seed, inc, lift, None, false);
                            consistent(&reference[index], &row);
                            emit(&Key { kind: "search", family, size, capacity, seed, arm }, &row, Some(repetition));
                        }
                    }
                    for (index, &(inc, lift, arm)) in ARMS.iter().enumerate() {
                        let row = search_run(&case, seed, inc, lift, None, true);
                        consistent(&reference[index], &row);
                        emit(&Key { kind: "search", family, size, capacity, seed, arm }, &row, None);
                    }
                }
            }
        }
    }
}
