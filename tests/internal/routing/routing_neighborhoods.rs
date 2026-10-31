use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use qayd::engines::ls::lists::{
    audit_completed_canonical_rebuild_accounting, audit_direct_guided_exchange_enumeration, audit_interruptible_routing_budget_admission,
    NeighborhoodKind, RoutingAuditOutcome, RoutingNeighborhoodAudit, ScanMode, WorkBudget,
};
use qayd::model::list::{
    register_external_function, CollectionModel, Constraint, ExprArena, GlobalConstraint, Iterable, ObjectiveTier, Op, ReduceOp, Reduction,
};

fn edge_reduction(list: usize, matrix: Arc<Vec<Vec<i64>>>) -> Reduction {
    let mut arena = ExprArena::default();
    let from = arena.arg(0);
    let to = arena.arg(1);
    let body = arena.matrix(matrix, from, to);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
}

fn demand_reduction(list: usize, demands: Arc<Vec<i64>>) -> Reduction {
    let mut arena = ExprArena::default();
    let item = arena.arg(0);
    let body = arena.array(demands, item);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

fn routing_model(items: usize, lists: usize, matrix: Arc<Vec<Vec<i64>>>, capacity: Option<i64>) -> CollectionModel {
    let demands = Arc::new(vec![1; items + 1]);
    CollectionModel {
        items: (1..=items as i32).collect(),
        lists,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: (0..lists).map(|list| edge_reduction(list, Arc::clone(&matrix))).collect(),
            max_terms: None,
        }],
        constraints: capacity.map_or_else(Vec::new, |rhs| {
            (0..lists).map(|list| Constraint { reduction: demand_reduction(list, Arc::clone(&demands)), op: Op::Le, rhs }).collect()
        }),
        globals: Vec::new(),
        schedule: None,
    }
}

fn zero_matrix(items: usize) -> Arc<Vec<Vec<i64>>> {
    Arc::new(vec![vec![0; items + 1]; items + 1])
}

fn partition_only_model(items: usize, lists: usize) -> CollectionModel {
    CollectionModel {
        items: (1..=items as i32).collect(),
        lists,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    }
}

fn external_pair_reduction(list: usize, name: &str) -> Reduction {
    let mut arena = ExprArena::default();
    let left = arena.arg(0);
    let body = arena.external(name, vec![left]);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Pairs(list), arena, body, coeff: 1 }
}

fn clustered_model() -> (CollectionModel, Vec<Vec<i32>>) {
    const ITEMS: usize = 60;
    const SPLIT: usize = 31;
    let matrix = Arc::new(
        (0..=ITEMS)
            .map(|from| {
                (0..=ITEMS)
                    .map(|to| {
                        if from == to {
                            return 0;
                        }
                        if from == 0 || to == 0 {
                            let customer = from.max(to);
                            return if customer > SPLIT { 1 } else { 1_000 };
                        }
                        let from_left = from <= SPLIT;
                        let to_left = to <= SPLIT;
                        if from_left == to_left {
                            0
                        } else {
                            100
                        }
                    })
                    .collect()
            })
            .collect(),
    );
    let model = routing_model(ITEMS, 2, matrix, Some(30));
    let lists = vec![(1..=SPLIT as i32).collect(), ((SPLIT + 1) as i32..=ITEMS as i32).collect()];
    (model, lists)
}

#[test]
fn routing_budget_setup_is_interruptible_and_admitted_atomically() {
    assert!(audit_interruptible_routing_budget_admission());
}

#[test]
fn exhausted_budget_is_not_reported_as_a_complete_scan() {
    let model = routing_model(6, 1, zero_matrix(6), None);
    let mut audit = RoutingNeighborhoodAudit::new(&model, vec![vec![1, 2, 3, 4, 5, 6]]);
    let stop = AtomicBool::new(false);

    let first = audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Global, WorkBudget::new(1, 1));
    assert_eq!(first.outcome, RoutingAuditOutcome::BudgetExhausted);
    assert_eq!((first.generated, first.generation_work, first.evaluated), (0, 1, 0));

    let mut final_outcome = first.outcome;
    for _ in 0..256 {
        let run = audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Global, WorkBudget::new(1, 1));
        final_outcome = run.outcome;
        if final_outcome == RoutingAuditOutcome::Complete {
            break;
        }
    }
    assert_eq!(final_outcome, RoutingAuditOutcome::Complete);
}

#[test]
fn generation_and_evaluation_budgets_resume_through_a_pending_candidate() {
    let model = routing_model(6, 1, zero_matrix(6), None);
    let mut audit = RoutingNeighborhoodAudit::new(&model, vec![vec![1, 2, 3, 4, 5, 6]]);
    let stop = AtomicBool::new(false);

    let generated = audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Global, WorkBudget::new(2, 0));
    assert_eq!(generated.outcome, RoutingAuditOutcome::BudgetExhausted);
    assert_eq!((generated.generated, generated.generation_work, generated.evaluated), (1, 2, 0));

    let evaluated = audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Global, WorkBudget::new(0, 1));
    assert_eq!(evaluated.outcome, RoutingAuditOutcome::BudgetExhausted);
    assert_eq!((evaluated.generated, evaluated.generation_work, evaluated.evaluated), (0, 0, 1));
}

#[test]
fn one_work_unit_bounds_every_global_cursor_on_empty_routes() {
    let model = routing_model(1, 8, zero_matrix(1), None);
    let lists = vec![vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![1]];
    let stop = AtomicBool::new(false);

    for kind in NeighborhoodKind::ALL {
        let mut audit = RoutingNeighborhoodAudit::new(&model, lists.clone());
        let run = audit.run(&stop, kind, ScanMode::Global, WorkBudget::new(1, 1));
        assert_eq!(run.outcome, RoutingAuditOutcome::BudgetExhausted, "{kind:?}");
        assert_eq!(run.generation_work, 1, "{kind:?}");
        assert!(run.generated <= 1, "{kind:?}");
        assert!(run.evaluated <= run.generated, "{kind:?}");
    }
}

#[test]
fn global_cursor_resumes_one_empty_route_at_a_time() {
    let model = routing_model(1, 4, zero_matrix(1), None);
    let mut audit = RoutingNeighborhoodAudit::new(&model, vec![vec![], vec![], vec![], vec![1]]);
    let stop = AtomicBool::new(false);

    for _ in 0..3 {
        let run = audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Global, WorkBudget::new(1, 1));
        assert_eq!(run.outcome, RoutingAuditOutcome::BudgetExhausted);
        assert_eq!((run.generated, run.generation_work, run.evaluated), (0, 1, 0));
    }

    let candidate = audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Global, WorkBudget::new(1, 1));
    assert_eq!(candidate.outcome, RoutingAuditOutcome::BudgetExhausted);
    assert_eq!((candidate.generated, candidate.generation_work, candidate.evaluated), (1, 1, 1));
}

#[test]
fn unit_slices_reproduce_each_complete_global_neighborhood_exactly() {
    let model = routing_model(5, 3, zero_matrix(5), None);
    let lists = vec![vec![1, 2, 3], vec![4, 5], vec![]];
    let stop = AtomicBool::new(false);
    let expected_candidates = [30, 6, 19, 16, 12, 4];

    for (kind, expected) in NeighborhoodKind::ALL.into_iter().zip(expected_candidates) {
        let mut one_shot = RoutingNeighborhoodAudit::new(&model, lists.clone());
        let complete = one_shot.run(&stop, kind, ScanMode::Global, WorkBudget::new(u64::MAX, u64::MAX));
        assert_eq!(complete.outcome, RoutingAuditOutcome::Complete, "{kind:?}");
        assert_eq!((complete.generated, complete.evaluated), (expected, expected), "{kind:?}");

        let mut sliced = RoutingNeighborhoodAudit::new(&model, lists.clone());
        let mut generated = 0;
        let mut generation_work = 0;
        let mut evaluated = 0;
        let mut outcome = RoutingAuditOutcome::BudgetExhausted;
        for _ in 0..10_000 {
            let run = sliced.run(&stop, kind, ScanMode::Global, WorkBudget::new(1, 1));
            generated += run.generated;
            generation_work += run.generation_work;
            evaluated += run.evaluated;
            outcome = run.outcome;
            if outcome == RoutingAuditOutcome::Complete {
                break;
            }
        }
        assert_eq!(outcome, RoutingAuditOutcome::Complete, "{kind:?}");
        assert_eq!((generated, evaluated), (complete.generated, complete.evaluated), "{kind:?}");
        assert_eq!(generation_work, complete.generation_work, "{kind:?}");
    }
}

#[test]
fn granular_relocate_generation_is_linear_in_items_times_k() {
    let (model, lists) = clustered_model();
    let mut audit = RoutingNeighborhoodAudit::new(&model, lists);
    let stop = AtomicBool::new(false);
    let run = audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Granular, WorkBudget::new(u64::MAX, u64::MAX));

    assert_eq!(run.outcome, RoutingAuditOutcome::Complete);
    assert_eq!(run.generated, run.evaluated);
    // PerList's default k is 24. Four directed adjacency placements per
    // undirected anchor fit comfortably below this causal O(n*k) ceiling.
    assert!(run.generated <= 8 * 60 * 24, "generated {} candidates", run.generated);
}

#[test]
fn granular_empty_destinations_are_bounded_and_deterministic() {
    const ITEMS: usize = 3;
    const EMPTY_ROUTES: usize = 4_096;
    let model = partition_only_model(ITEMS, EMPTY_ROUTES + 1);
    let mut lists = vec![Vec::new(); EMPTY_ROUTES + 1];
    lists[0] = vec![1, 2, 3];

    let mut first = RoutingNeighborhoodAudit::new(&model, lists.clone());
    first.set_interchangeable_lists(false);
    let first =
        first.granular_empty_destination_audit(NeighborhoodKind::Relocate).expect("the granular empty-destination cursor must complete");

    let mut or_opt = RoutingNeighborhoodAudit::new(&model, lists.clone());
    or_opt.set_interchangeable_lists(false);
    let or_opt = or_opt
        .granular_empty_destination_audit(NeighborhoodKind::OrOpt)
        .expect("the granular Or-opt empty-destination cursor must complete");

    let mut second = RoutingNeighborhoodAudit::new(&model, lists);
    second.set_interchangeable_lists(false);
    let second = second.granular_empty_destination_audit(NeighborhoodKind::Relocate).expect("the repeated granular cursor must complete");

    assert_eq!(first, second, "the sampled empty routes must not depend on hash iteration order");
    assert_eq!(first.0, 8 * ITEMS as u64);
    assert!(first.1 <= first.0 + 9, "cursor work {} exceeded the item-by-probe bound", first.1);
    assert_eq!(first.2.len(), 8);
    assert_eq!(first.2.first(), Some(&1));
    assert_eq!(first.2.last(), Some(&3_585));
    assert_eq!(or_opt.0, 8 * ITEMS as u64);
    assert!(or_opt.1 <= 8 * ITEMS as u64 * 4 + 9, "Or-opt cursor work {} exceeded its bounded segment scan", or_opt.1);
    assert_eq!(or_opt.2, first.2);
}

#[test]
fn global_relocate_scan_retains_complete_empty_route_coverage() {
    const ITEMS: usize = 3;
    const EMPTY_ROUTES: usize = 32;
    let model = partition_only_model(ITEMS, EMPTY_ROUTES + 1);
    let mut lists = vec![Vec::new(); EMPTY_ROUTES + 1];
    lists[0] = vec![1, 2, 3];
    let stop = AtomicBool::new(false);
    let mut audit = RoutingNeighborhoodAudit::new(&model, lists);
    audit.set_interchangeable_lists(false);

    let run = audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Global, WorkBudget::new(u64::MAX, u64::MAX));

    assert_eq!(run.outcome, RoutingAuditOutcome::Complete);
    assert_eq!(run.generated, (ITEMS * EMPTY_ROUTES + ITEMS * (ITEMS - 1)) as u64);
    assert_eq!(run.evaluated, run.generated);
}

#[test]
fn granular_index_preparation_is_resumable_and_shared_by_all_operators() {
    let model = routing_model(6, 3, zero_matrix(6), None);
    let mut audit = RoutingNeighborhoodAudit::new(&model, vec![vec![1, 2, 3], vec![4, 5], vec![6]]);
    let stop = AtomicBool::new(false);
    let mut preparation_work = 0u64;

    for _ in 0..64 {
        let run = audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Granular, WorkBudget::new(1, 0));
        assert_eq!(run.outcome, RoutingAuditOutcome::BudgetExhausted);
        assert_eq!(run.generation_work, 1);
        preparation_work += run.generation_work;
        if audit.completed_index_builds() == 1 {
            break;
        }
    }

    assert_eq!(audit.completed_index_builds(), 1);
    assert!(preparation_work > 6, "route transitions and allocation must be charged as structural work");
    assert_eq!(audit.cached_index_shape(), Some((6, 6)));

    let _ = audit.run(&stop, NeighborhoodKind::Swap, ScanMode::Granular, WorkBudget::new(8, 8));
    let _ = audit.run(&stop, NeighborhoodKind::OrOpt, ScanMode::Granular, WorkBudget::new(8, 8));
    assert_eq!(audit.completed_index_builds(), 1, "all granular operators must reuse one topology index");

    audit.reset();
    for _ in 0..64 {
        let _ = audit.run(&stop, NeighborhoodKind::Reverse, ScanMode::Granular, WorkBudget::new(1, 0));
        if audit.completed_index_builds() == 2 {
            break;
        }
    }
    assert_eq!(audit.completed_index_builds(), 2, "a topology invalidation must rebuild the shared index exactly once");
}

#[test]
fn depot_candidates_generate_bounded_inter_route_boundary_moves() {
    let model = routing_model(6, 2, zero_matrix(6), None);
    let mut audit = RoutingNeighborhoodAudit::new(&model, vec![vec![1, 2, 3], vec![4, 5, 6]]);

    for kind in [
        NeighborhoodKind::Relocate,
        NeighborhoodKind::Swap,
        NeighborhoodKind::OrOpt,
        NeighborhoodKind::TwoOptStar,
        NeighborhoodKind::CrossExchange,
    ] {
        let (total, inter_route) = audit.boundary_move_counts(kind, 2, 0).expect("item and depot must be indexed candidates");
        assert!(total > 0, "{kind:?}");
        assert!(inter_route > 0, "{kind:?} must not discard a client-depot candidate at another route boundary");
    }

    let (reverse, inter_route) = audit.boundary_move_counts(NeighborhoodKind::Reverse, 2, 0).expect("reverse boundary moves");
    assert!(reverse > 0);
    assert_eq!(inter_route, 0);
}

#[test]
fn physical_boundary_values_keep_virtual_boundary_variants() {
    let mut model = routing_model(6, 2, zero_matrix(6), None);
    model.items.insert(0, 0);
    let mut audit = RoutingNeighborhoodAudit::new(&model, vec![vec![0, 1, 2], vec![3, 4, 5, 6]]);

    for kind in NeighborhoodKind::ALL {
        let (total, boundary, present, unique) =
            audit.anchor_boundary_coverage(kind, 2, 0).expect("both candidate endpoints must have physical locations");
        assert!(boundary > 0, "{kind:?} fixture must have virtual boundary variants");
        assert_eq!(present, boundary, "{kind:?} lost a virtual boundary variant behind the physical depot anchor");
        assert_eq!(unique, total, "{kind:?} emitted duplicate physical and virtual variants");
        if kind == NeighborhoodKind::Relocate {
            assert!(total > boundary, "the physical depot anchor must remain in addition to virtual boundaries");
        }
    }
}

#[test]
fn granular_reverse_finds_a_candidate_edge_improvement() {
    let mut matrix = vec![vec![20; 4]; 4];
    for (node, row) in matrix.iter_mut().enumerate() {
        row[node] = 0;
    }
    for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
        matrix[a][b] = 1;
        matrix[b][a] = 1;
    }
    matrix[1][3] = 15;
    matrix[3][1] = 15;
    matrix[2][0] = 15;
    matrix[0][2] = 15;
    let model = routing_model(3, 1, Arc::new(matrix), None);
    let mut audit = RoutingNeighborhoodAudit::new(&model, vec![vec![1, 3, 2]]);
    let stop = AtomicBool::new(false);

    let run = audit.run(&stop, NeighborhoodKind::Reverse, ScanMode::Granular, WorkBudget::new(128, 128));
    assert_eq!(run.outcome, RoutingAuditOutcome::Improved);
    assert!(run.evaluated > 0);
}

#[test]
fn bounded_global_scan_resumes_and_finds_move_missing_from_granular_graph() {
    let (model, lists) = clustered_model();
    let mut audit = RoutingNeighborhoodAudit::new(&model, lists);
    let stop = AtomicBool::new(false);

    let granular = audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Granular, WorkBudget::new(u64::MAX, u64::MAX));
    assert_eq!(granular.outcome, RoutingAuditOutcome::Complete);

    let mut exhausted_slices = 0;
    let mut outcome = RoutingAuditOutcome::BudgetExhausted;
    for _ in 0..128 {
        let run = audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Global, WorkBudget::new(1, 1));
        outcome = run.outcome;
        match outcome {
            RoutingAuditOutcome::BudgetExhausted => exhausted_slices += 1,
            RoutingAuditOutcome::Improved => break,
            other => panic!("global scan ended unexpectedly: {other:?}"),
        }
    }
    assert_eq!(outcome, RoutingAuditOutcome::Improved);
    assert!(exhausted_slices > 1, "the cursor must resume across multiple bounded slices");
}

#[test]
fn routing_scan_honours_an_already_fired_stop_token() {
    let model = routing_model(6, 1, zero_matrix(6), None);
    let mut audit = RoutingNeighborhoodAudit::new(&model, vec![vec![1, 2, 3, 4, 5, 6]]);
    let stop = AtomicBool::new(true);

    let run = audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Global, WorkBudget::new(100, 100));
    assert_eq!(run.outcome, RoutingAuditOutcome::Interrupted);
    assert_eq!((run.generated, run.generation_work, run.evaluated), (0, 0, 0));
}

#[test]
fn routing_scan_honours_stop_between_resumable_structural_steps() {
    let model = routing_model(1, 4, zero_matrix(1), None);
    let mut audit = RoutingNeighborhoodAudit::new(&model, vec![vec![], vec![], vec![], vec![1]]);
    let stop = AtomicBool::new(false);

    let first = audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Global, WorkBudget::new(1, 1));
    assert_eq!((first.outcome, first.generation_work), (RoutingAuditOutcome::BudgetExhausted, 1));

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let interrupted = audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Global, WorkBudget::new(100, 100));
    assert_eq!(interrupted.outcome, RoutingAuditOutcome::Interrupted);
    assert_eq!((interrupted.generated, interrupted.generation_work, interrupted.evaluated), (0, 0, 0));
}

#[test]
fn interrupted_candidate_score_is_not_counted_and_remains_pending() {
    const NAME: &str = "qayd_routing_candidate_score_interrupter";
    const ITEMS: usize = 64;
    let stop = Arc::new(AtomicBool::new(false));
    let armed = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_stop = Arc::clone(&stop);
    let callback_armed = Arc::clone(&armed);
    let callback_calls = Arc::clone(&calls);
    register_external_function(NAME, move |_| {
        if callback_armed.load(Ordering::Acquire) {
            callback_calls.fetch_add(1, Ordering::Relaxed);
            callback_stop.store(true, Ordering::Release);
        }
        Ok(1)
    })
    .expect("the score-interruption oracle has a unique evaluator name");

    let mut model = routing_model(ITEMS, 1, zero_matrix(ITEMS), None);
    model.objectives[0].terms.push(external_pair_reduction(0, NAME));
    let mut audit = RoutingNeighborhoodAudit::new(&model, vec![(1..=ITEMS as i32).collect()]);
    assert_eq!(calls.load(Ordering::Relaxed), 0, "the callback must stay disarmed while accepted caches are built");
    armed.store(true, Ordering::Release);

    let interrupted = audit.run(stop.as_ref(), NeighborhoodKind::Relocate, ScanMode::Global, WorkBudget::new(u64::MAX, u64::MAX));

    assert_eq!(interrupted.outcome, RoutingAuditOutcome::Interrupted);
    assert_eq!(interrupted.generated, 1);
    assert_eq!(interrupted.evaluated, 0, "a partial score must not be reported as a completed evaluation");
    assert!(calls.load(Ordering::Relaxed) > 0, "the stop token must fire from inside candidate scoring");

    armed.store(false, Ordering::Release);
    stop.store(false, Ordering::Release);
    let resumed = audit.run(stop.as_ref(), NeighborhoodKind::Relocate, ScanMode::Global, WorkBudget::new(0, 1));
    assert_eq!(resumed.outcome, RoutingAuditOutcome::BudgetExhausted);
    assert_eq!((resumed.generated, resumed.generation_work, resumed.evaluated), (0, 0, 1));
}

#[test]
fn heterogeneous_vehicles_keep_distinct_empty_destinations_within_the_granular_cap() {
    let homogeneous = routing_model(1, 3, zero_matrix(1), None);
    let mut heterogeneous = homogeneous.clone();
    heterogeneous.constraints.push(Constraint { reduction: demand_reduction(1, Arc::new(vec![1; 2])), op: Op::Le, rhs: 1 });
    let lists = vec![vec![1], vec![], vec![]];
    let stop = AtomicBool::new(false);

    let mut homogeneous_audit = RoutingNeighborhoodAudit::new(&homogeneous, lists.clone());
    let homogeneous_run = homogeneous_audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Granular, WorkBudget::new(u64::MAX, u64::MAX));
    assert_eq!(homogeneous_run.outcome, RoutingAuditOutcome::Complete);
    assert_eq!(homogeneous_run.generated, 1, "one representative empty route is enough for interchangeable vehicles");

    homogeneous_audit.set_interchangeable_lists(false);
    let symmetry_disabled =
        homogeneous_audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Granular, WorkBudget::new(u64::MAX, u64::MAX));
    assert_eq!(symmetry_disabled.outcome, RoutingAuditOutcome::Complete);
    assert_eq!(symmetry_disabled.generated, 2, "changing symmetry status invalidates the cached granular index");

    let mut heterogeneous_audit = RoutingNeighborhoodAudit::new(&heterogeneous, lists);
    let heterogeneous_run =
        heterogeneous_audit.run(&stop, NeighborhoodKind::Relocate, ScanMode::Granular, WorkBudget::new(u64::MAX, u64::MAX));
    assert_eq!(heterogeneous_run.outcome, RoutingAuditOutcome::Complete);
    assert_eq!(heterogeneous_run.generated, 2, "heterogeneous empty routes are distinct destinations");
}

#[test]
fn composite_raw_scoring_matches_canonical_materialization() {
    let items = 7usize;
    let matrix = Arc::new(
        (0..=items)
            .map(|from| {
                (0..=items)
                    .map(|to| {
                        if from == to {
                            0
                        } else {
                            (from as i64 + 3).saturating_mul(17).saturating_add((to as i64 + 5).saturating_mul(11))
                        }
                    })
                    .collect()
            })
            .collect(),
    );
    let mut model = routing_model(items, 3, matrix, Some(3));
    model.globals = vec![
        GlobalConstraint::SameList { a: 1, b: 2 },
        GlobalConstraint::ListLe { before: 3, after: 6 },
        GlobalConstraint::DifferentList { a: 4, b: 7 },
    ];
    let audit = RoutingNeighborhoodAudit::new(&model, vec![vec![1, 2, 3], vec![4, 5], vec![6, 7]]);

    let (or_opt, cross_exchange, guided_differences) = audit.audit_composite_raw_scores();

    assert_eq!(or_opt, 34);
    assert_eq!(cross_exchange, 45);
    assert!(guided_differences > 0, "the fixture must distinguish raw scoring from guided penalties");
}

#[test]
fn interrupted_canonical_rebuild_is_not_counted_as_completed() {
    assert!(audit_completed_canonical_rebuild_accounting());
}

#[test]
fn guided_exchange_reaches_a_direct_candidate_before_the_legacy_cap() {
    assert!(audit_direct_guided_exchange_enumeration());
}
