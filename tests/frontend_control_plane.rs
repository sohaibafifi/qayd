use std::fs;
use std::path::{Path, PathBuf};

fn rust_sources(root: &Path, output: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).unwrap_or_else(|error| panic!("cannot read {}: {error}", root.display())) {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            rust_sources(&path, output);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            output.push(path);
        }
    }
}

#[test]
fn every_frontend_is_parser_builder_and_renderer_only() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources = Vec::new();
    rust_sources(&root.join("src/frontends"), &mut sources);
    for entry in fs::read_dir(root.join("frontends")).expect("frontend crates directory") {
        let source = entry.expect("frontend crate entry").path().join("src");
        if source.is_dir() {
            rust_sources(&source, &mut sources);
        }
    }
    sources.sort();

    let forbidden = [
        "crate::constraints",
        "qayd::constraints",
        "crate::engines",
        "qayd::engines",
        "crate::problem",
        "qayd::problem",
        "crate::search",
        "qayd::search",
        "crate::Solver",
        "qayd::Solver",
        "LocalSearchSpec",
        "CompiledCollection",
        "CollectionBudget",
        "BackendSelection",
        "execute_specialized",
        "solve_compiled_problem",
        "solve_physical_",
        "std::time::Instant",
        "std::thread",
        "thread::spawn",
        "std::env::var(",
        "std::env::var_os(",
    ];
    for path in sources {
        let source = fs::read_to_string(&path).unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        for needle in forbidden {
            assert!(
                !source.contains(needle),
                "frontend {} crosses the control-plane boundary through `{needle}`",
                path.strip_prefix(root).unwrap_or(&path).display()
            );
        }
    }
}

#[test]
fn retired_parallel_module_cannot_reappear() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(!root.join("src/parallel.rs").exists());
    let library = fs::read_to_string(root.join("src/lib.rs")).expect("crate root");
    assert!(!library.contains("mod parallel"));
    assert!(root.join("src/engines/cp/portfolio.rs").is_file());
    assert!(root.join("src/orchestrator/executor.rs").is_file());
}

#[test]
fn integer_search_has_one_engine_and_one_control_plane() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(root.join("src/engines/ls/integer.rs").is_file());
    assert!(root.join("src/orchestrator/integer_search.rs").is_file());
    assert!(!root.join("src/orchestrator/integer_ls.rs").exists());
    assert!(!root.join("src/orchestrator/physical.rs").exists());
}

#[test]
fn engines_cannot_construct_the_public_solve_protocol() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources = Vec::new();
    rust_sources(&root.join("src/engines"), &mut sources);
    for path in sources {
        let source = fs::read_to_string(&path).unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        for needle in ["SolveResult", "ProofClaim", "SolveBudget", "finalize_decision_result"] {
            assert!(
                !source.contains(needle),
                "engine {} constructs the orchestrator protocol through `{needle}`",
                path.strip_prefix(root).unwrap_or(&path).display()
            );
        }
    }
    let list_engine = fs::read_to_string(root.join("src/engines/ls/lists/local_search.rs")).expect("list local-search source");
    assert!(!list_engine.contains("solve_schedule"), "the list engine launches the scheduling engine");
}

#[test]
fn portfolio_allocation_and_schedule_analysis_have_one_owner() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let solve = fs::read_to_string(root.join("src/orchestrator/solve.rs")).expect("prepared solve source");
    let executor = fs::read_to_string(root.join("src/orchestrator/executor.rs")).expect("worker executor source");
    assert!(solve.contains("WorkerAllocation::portfolio(*workers)"));
    assert!(executor.contains("struct WorkerAllocation"));

    for relative in ["src/engines/ls/disjunctive_schedule.rs", "src/engines/ls/scenario_schedule.rs"] {
        let source = fs::read_to_string(root.join(relative)).expect("schedule engine source");
        for duplicate in ["fn constraint_scope", "fn domain_ceiling", "fn topological_order", "fn collect_affine"] {
            assert!(!source.contains(duplicate), "{relative} reintroduced the shared scheduling helper `{duplicate}`");
        }
        assert!(source.contains("PrecedenceDag"), "{relative} bypasses the shared scheduling graph");
    }

    let list_eval = fs::read_to_string(root.join("src/engines/ls/lists/eval.rs")).expect("list scorer source");
    assert!(list_eval.contains("eval_reduction_on_contents"));
    assert!(!list_eval.contains("match reduction.iterable"), "list LS reintroduced a second reduction evaluator");
}
