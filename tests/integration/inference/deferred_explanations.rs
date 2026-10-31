//! Public and paired measurement gates for deferred Boolean explanations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use qayd::model::{Constraint, IntExpr, Model, ModelPackage, Objective, Relation};
use qayd::orchestrator::{
    solve_model_silent, CpControls, IgnoreEvents, ProofRequest, ProvenConclusion, SearchPhase, SearchPolicy, SemanticAssumption,
    SemanticAssumptionOp, SemanticSolveSession, SolveLimits, SolveMode, SolveRequest, SolveResult, SolveStatus, ValueSelector,
    VariableSelector, VerificationLevel,
};

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

fn cardinality(n: usize, quota: usize, objective: bool) -> ModelPackage {
    let mut model = Model::new();
    let vars = (0..n).map(|_| model.int_range(0, 1)).collect::<Vec<_>>();
    model.add_constraint(Constraint::Linear {
        terms: vars.iter().map(|&var| (1, var)).collect(),
        relation: Relation::Le,
        rhs: quota as i64,
    });
    if objective {
        model.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Variable(vars[0]) });
    }
    ModelPackage::new(model)
}

fn pigeonhole(holes: usize, impossible: bool) -> ModelPackage {
    let mut model = Model::new();
    let pigeons = holes + usize::from(impossible);
    let vars = (0..holes * pigeons).map(|_| model.int_range(0, 1)).collect::<Vec<_>>();
    for row in vars.chunks(holes) {
        model.add_constraint(Constraint::Linear { terms: row.iter().map(|&var| (1, var)).collect(), relation: Relation::Eq, rhs: 1 });
    }
    for hole in 0..holes {
        model.add_constraint(Constraint::Linear {
            terms: (0..pigeons).map(|pigeon| (1, vars[pigeon * holes + hole])).collect(),
            relation: Relation::Le,
            rhs: 1,
        });
    }
    ModelPackage::new(model)
}

// Enumerate every assignment satisfying exactly one hole per pigeon, then
// independently discard repeated holes. This does not run any Qayd constraint.
fn pigeon_solutions(holes: usize, impossible: bool) -> Vec<Vec<i32>> {
    let pigeons = holes + usize::from(impossible);
    let mut solutions = Vec::new();
    for encoded in 0..holes.pow(pigeons as u32) {
        let mut remainder = encoded;
        let mut used = vec![false; holes];
        let mut assignment = vec![0; holes * pigeons];
        let mut valid = true;
        for pigeon in 0..pigeons {
            let hole = remainder % holes;
            remainder /= holes;
            valid &= !used[hole];
            used[hole] = true;
            assignment[pigeon * holes + hole] = 1;
        }
        if valid {
            solutions.push(assignment);
        }
    }
    solutions
}

fn request(n: usize, deferred: bool, checking: bool, seed: u64, cardinality: bool) -> SolveRequest {
    SolveRequest {
        mode: SolveMode::Exact,
        seed,
        threads: 1,
        cp: CpControls { deferred_explanations: deferred, explanation_metrics: true, check_inferences: checking, ..CpControls::default() },
        search_policy: SearchPolicy::new(vec![SearchPhase::new(
            (0..n).collect(),
            VariableSelector::Activity,
            if cardinality { ValueSelector::Max } else { ValueSelector::RandomSeeded },
        )]),
        ..SolveRequest::default()
    }
}

fn count(result: &SolveResult, key: &str) -> u64 {
    result
        .reports()
        .iter()
        .flat_map(|report| &report.metadata)
        .filter(|(name, _)| name == key)
        .map(|(_, value)| value.parse::<u64>().unwrap())
        .sum()
}

fn assignment(result: &SolveResult) -> Vec<i32> {
    let candidate = result.primal().unwrap();
    assert_eq!(candidate.verification(), VerificationLevel::Final);
    candidate.assignment().integers.iter().map(|value| i32::try_from(value.unwrap()).unwrap()).collect()
}

fn verify_pigeon(result: &SolveResult, expected: &[Vec<i32>]) {
    result.validate_contract().unwrap();
    if expected.is_empty() {
        assert_eq!(result.status(), SolveStatus::Unsatisfiable);
        assert!(result.primal().is_none());
        assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Unsatisfiable));
    } else {
        assert_eq!(result.status(), SolveStatus::Satisfiable);
        assert!(expected.contains(&assignment(result)));
    }
}

fn verify_cardinality(result: &SolveResult, quota: usize, objective: bool) {
    result.validate_contract().unwrap();
    assert_eq!(result.status(), if objective { SolveStatus::Optimal } else { SolveStatus::Satisfiable });
    let values = assignment(result);
    assert!(values.iter().all(|&value| value == 0 || value == 1));
    assert!(values.iter().sum::<i32>() <= quota as i32);
    if objective {
        assert_eq!(result.primal().unwrap().objectives(), &[1]);
        assert_eq!(result.proof().map(|proof| proof.conclusion()), Some(ProvenConclusion::Optimal));
    }
}

#[test]
fn canonical_workers_preserve_boolean_witnesses_proofs_and_i2_checks() {
    assert!(!CpControls::default().deferred_explanations);
    assert!(!CpControls::default().explanation_metrics);
    let mut handles = 0;
    let mut checker_expansions = 0;
    for impossible in [false, true] {
        let expected = pigeon_solutions(3, impossible);
        let package = pigeonhole(3, impossible);
        for threads in [1, 2] {
            for deferred in [false, true] {
                for checking in [false, true] {
                    let mut options = request(3 * (3 + usize::from(impossible)), deferred, checking, 7, false);
                    options.threads = threads;
                    options.cp.semantic_minimization = true;
                    if impossible {
                        options.proof = ProofRequest::Require;
                    }
                    let result = solve_model_silent(&package, &options).unwrap();
                    verify_pigeon(&result, &expected);
                    handles += count(&result, "explanation_created_handles");
                    checker_expansions += count(&result, "explanation_checker_expansions");
                    if !deferred {
                        assert_eq!(count(&result, "explanation_created_handles"), 0);
                    }
                    if checking {
                        assert_eq!(count(&result, "inference_resource_limited"), 0);
                        assert!(count(&result, "inference_checked_inferences") > 0);
                    }
                }
            }
        }
    }
    assert!(handles > 0 && checker_expansions > 0);
}

#[test]
fn public_cardinality_and_optimization_exercise_the_deferred_path() {
    for objective in [false, true] {
        for deferred in [false, true] {
            let result = solve_model_silent(&cardinality(8, 2, objective), &request(8, deferred, true, 1, true)).unwrap();
            verify_cardinality(&result, 2, objective);
            assert_eq!(count(&result, "inference_resource_limited"), 0);
            if deferred {
                assert!(count(&result, "explanation_created_handles") > 0);
            }
        }
    }
}

#[test]
fn default_requests_do_not_enable_explanation_instrumentation() {
    let mut options = request(8, false, false, 0, true);
    options.cp.explanation_metrics = false;
    let result = solve_model_silent(&cardinality(8, 2, false), &options).unwrap();
    verify_cardinality(&result, 2, false);
    assert!(result.reports().iter().flat_map(|report| &report.metadata).all(|(key, _)| !key.starts_with("explanation_")));
}

#[test]
fn persistent_sessions_toggle_deferral_with_temporary_assumptions() {
    let mut session = SemanticSolveSession::new(pigeonhole(3, false)).unwrap();
    let expected = pigeon_solutions(3, false);
    let stop = AtomicBool::new(false);
    for (deferred, impossible) in [(true, false), (false, true), (true, true), (false, false), (true, false)] {
        let mut options = request(9, deferred, true, 7, false);
        options.assumptions = vec![SemanticAssumption { variable: 0, operation: SemanticAssumptionOp::Eq, value: 1 }];
        if impossible {
            options.assumptions.push(SemanticAssumption { variable: 3, operation: SemanticAssumptionOp::Eq, value: 1 });
        }
        let result = session.solve_with_external_stop(&options, &stop, &mut IgnoreEvents).unwrap();
        let filtered = expected.iter().filter(|values| values[0] == 1 && (!impossible || values[3] == 1)).cloned().collect::<Vec<_>>();
        verify_pigeon(&result, &filtered);
        if !deferred {
            assert_eq!(count(&result, "explanation_created_handles"), 0);
        }
    }
}

#[test]
fn zero_budget_does_not_publish_a_completion_claim_with_deferral() {
    let mut options = request(20, true, true, 7, false);
    options.limits = SolveLimits { time: Some(Duration::ZERO), ..SolveLimits::default() };
    let result = solve_model_silent(&pigeonhole(4, true), &options).unwrap();
    result.validate_contract().unwrap();
    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(result.primal().is_none() && result.proof().is_none());
}

struct TemporaryInstance(PathBuf);

impl Drop for TemporaryInstance {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn cli_exposes_deferred_and_eager_instrumented_arms() {
    let instance = TemporaryInstance(std::env::temp_dir().join(format!("qayd-deferred-{}.xml", std::process::id())));
    std::fs::write(
        &instance.0,
        r#"<instance format="XCSP3" type="CSP">
      <variables><array id="x" size="[8]"> 0 1 </array></variables>
      <constraints><sum><list>x[]</list><condition>(le,2)</condition></sum></constraints>
    </instance>"#,
    )
    .unwrap();
    for deferred in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_qayd"));
        command.args(["--verbose", "--explanation-metrics", "--check-inferences", "--search-phase", "0,1,2,3,4,5,6,7:activity:max"]);
        if deferred {
            command.arg("--deferred-explanations");
        }
        let output = command.arg(&instance.0).output().unwrap();
        assert!(output.status.success(), "{:?}: {}", output.status, String::from_utf8_lossy(&output.stderr));
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("s SATISFIABLE"), "{stdout}");
        assert!(stdout.contains(if deferred { "c explanations mode deferred" } else { "c explanations mode eager" }), "{stdout}");
    }
}

#[test]
#[ignore]
fn deferred_metrics_sweep() {
    let fields = [
        "explanation_created_handles",
        "explanation_lcg_requests",
        "explanation_lcg_first_expansions",
        "explanation_lcg_cache_hits",
        "explanation_checker_expansions",
        "explanation_materialized_premises",
        "explanation_eager_premise_buffers",
        "explanation_eager_premises",
        "explanation_eager_lcg_buffers",
        "explanation_eager_lcg_literals",
        "explanation_peak_retained_upper_bytes",
    ];
    eprintln!("family,n,parameter,seed,deferred,status,nodes,failures,elapsed_ms,allocation_calls,reallocation_calls,allocated_bytes,peak_extra_live_bytes,handles,requests,expansions,cache_hits,checker_expansions,materialized_premises,eager_buffers,eager_premises,eager_lcg_buffers,eager_lcg_literals,deferred_storage_upper_bytes");
    let mut cases = Vec::new();
    for n in [32, 128, 256] {
        for quota in [1, n / 8] {
            cases.push(("cardinality", n, quota, cardinality(n, quota, false), None));
        }
    }
    for holes in [2, 3, 4] {
        for impossible in [false, true] {
            cases.push((
                "pigeonhole",
                holes * (holes + usize::from(impossible)),
                usize::from(impossible),
                pigeonhole(holes, impossible),
                Some(pigeon_solutions(holes, impossible)),
            ));
        }
    }
    for (family, n, parameter, package, expected) in cases {
        for seed in [0, 1, 7] {
            for deferred in if seed == 0 { [false, true] } else { [true, false] } {
                let options = request(n, deferred, false, seed, family == "cardinality");
                let started = Instant::now();
                let timed = solve_model_silent(&package, &options).unwrap();
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                begin_allocations();
                let allocated_result = solve_model_silent(&package, &options);
                let allocations = end_allocations();
                let allocated_result = allocated_result.unwrap();
                for result in [&timed, &allocated_result] {
                    match &expected {
                        Some(expected) => verify_pigeon(result, expected),
                        None => verify_cardinality(result, parameter, false),
                    }
                }
                let stats = timed.aggregate_search_stats();
                let allocation_stats = allocated_result.aggregate_search_stats();
                assert_eq!((stats.nodes, stats.failures), (allocation_stats.nodes, allocation_stats.failures));
                for key in fields {
                    assert_eq!(count(&timed, key), count(&allocated_result, key), "different measurement trajectory for {key}");
                }
                let counters = fields.map(|key| count(&timed, key).to_string()).join(",");
                eprintln!(
                    "{family},{n},{parameter},{seed},{deferred},{:?},{},{},{elapsed_ms:.3},{},{},{},{},{counters}",
                    timed.status(),
                    stats.nodes,
                    stats.failures,
                    allocations[0],
                    allocations[1],
                    allocations[2],
                    allocations[3]
                );
            }
        }
    }
}
