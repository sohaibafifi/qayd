use qayd::orchestrator::{ScheduleJsspSearch, SolveMode, SolveRequest};

fn adaptive_request() -> SolveRequest {
    SolveRequest { mode: SolveMode::LocalSearch, schedule_jssp_search: ScheduleJsspSearch::TsabAdaptive, ..SolveRequest::default() }
}

#[test]
fn adaptive_jssp_request_is_explicit_and_single_threaded() {
    assert_eq!(SolveRequest::default().schedule_jssp_search, ScheduleJsspSearch::Legacy);
    adaptive_request().validate().expect("adaptive JSSP supports one worker without profiling");
    for threads in [2, 6, 7, 8] {
        let request = SolveRequest { threads, ..adaptive_request() };
        let error = request.validate().expect_err("adaptive JSSP must not silently select a different worker layout");
        assert!(error.to_string().contains("requires threads=1"));
    }
}

#[test]
fn adaptive_jssp_rejects_unqualified_hybrid_controls() {
    for request in [
        SolveRequest { schedule_path_relink: true, ..adaptive_request() },
        SolveRequest { schedule_lns_shadow: true, ..adaptive_request() },
        SolveRequest { schedule_constructor_workers: 1, ..adaptive_request() },
    ] {
        assert!(request.validate().is_err());
    }
}

#[test]
fn existing_tsab_pilots_keep_their_seven_worker_contract() {
    for strategy in [ScheduleJsspSearch::TsabCandidate, ScheduleJsspSearch::TsabMultiCandidate] {
        let request = SolveRequest { threads: 7, schedule_jssp_search: strategy, ..SolveRequest::default() };
        request.validate().expect("existing seven-worker TSAB configuration remains supported");
        let invalid = SolveRequest { threads: 1, ..request };
        assert!(invalid
            .validate()
            .expect_err("old pilots must not become monothread implicitly")
            .to_string()
            .contains("requires threads=7"));
    }
    SolveRequest {
        threads: 7,
        schedule_jssp_search: ScheduleJsspSearch::TsabMultiCandidate,
        schedule_path_relink: true,
        ..SolveRequest::default()
    }
    .validate()
    .expect("the qualified multi-candidate macro-relink control remains supported");
}
