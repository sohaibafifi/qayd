use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
#[cfg(feature = "lp-relaxation")]
use std::time::Duration;

#[cfg(feature = "lp-relaxation")]
use qayd::engines::dual::compute_with_linear;
use qayd::engines::dual::{
    attach, audit_reset_routing_relaxation_edge_evaluations, audit_routing_relaxation_edge_evaluations, compute, DualAuditGuard,
};
use qayd::model::list::{
    CollectionModel, CollectionSolution, Constraint, ExprArena, IntervalVar, Iterable, ObjectiveTier, Op, ReduceOp, Reduction, Schedule,
};
use qayd::model::Model;
use qayd::orchestrator::{
    audit_watch_local_search_dual, compile_collection_plan, solve_collection_plan, EngineKind, EventCallback, EventControl,
    LocalSearchDualAudit, SolveBudget, SolveError, SolveEvent, SolveLimits, SolveMode, SolveRequest,
};
#[cfg(feature = "lp-relaxation")]
use qayd::orchestrator::{LinearBackendMode, LinearControls};

fn edge_term(list: usize, costs: &[Vec<i64>]) -> Reduction {
    let mut arena = ExprArena::default();
    let from = arena.arg(0);
    let to = arena.arg(1);
    let body = arena.matrix(Arc::new(costs.to_vec()), from, to);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
}

fn item_term(list: usize, values: &[i64], op: ReduceOp) -> Reduction {
    let mut arena = ExprArena::default();
    let item = arena.arg(0);
    let body = arena.array(Arc::new(values.to_vec()), item);
    Reduction { op, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

fn vrptw_constraint(list: usize, travel: &[Vec<i64>], earliest: &[i64], latest: &[i64], service: &[i64]) -> Constraint {
    let mut arena = ExprArena::default();
    let current = arena.arg(0);
    let accumulator = arena.arg(1);
    let previous = arena.arg(2);
    let release = arena.array(Arc::new(earliest.to_vec()), current);
    let travel_time = arena.matrix(Arc::new(travel.to_vec()), previous, current);
    let arrival = arena.add(accumulator, travel_time);
    let start = arena.max(release, arrival);
    let service_time = arena.array(Arc::new(service.to_vec()), current);
    let departure = arena.add(start, service_time);
    let emit_service = arena.array(Arc::new(service.to_vec()), current);
    let start_again = arena.sub(accumulator, emit_service);
    let deadline = arena.array(Arc::new(latest.to_vec()), current);
    let lateness = arena.sub(start_again, deadline);
    let zero = arena.constant(0);
    let violation = arena.max(zero, lateness);
    Constraint {
        reduction: Reduction {
            op: ReduceOp::Sum,
            iterable: Iterable::Scan { list, init: earliest[0], boundary: 0, step: departure, end: Some(0) },
            arena,
            body: violation,
            coeff: 1,
        },
        op: Op::Le,
        rhs: 0,
    }
}

fn routing_model(costs: &[Vec<i64>], demands: &[i64], routes: usize, capacity: Option<i64>) -> CollectionModel {
    let items = (1..costs.len() as i32).collect::<Vec<_>>();
    let terms = (0..routes).map(|list| edge_term(list, costs)).collect();
    let constraints = capacity.map_or_else(Vec::new, |capacity| {
        (0..routes).map(|list| Constraint { reduction: item_term(list, demands, ReduceOp::Sum), op: Op::Le, rhs: capacity }).collect()
    });
    CollectionModel {
        items,
        lists: routes,
        objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
        constraints,
        globals: Vec::new(),
        schedule: None,
    }
}

fn shared_edge_term(list: usize, costs: &Arc<Vec<Vec<i64>>>) -> Reduction {
    let mut arena = ExprArena::default();
    let from = arena.arg(0);
    let to = arena.arg(1);
    let body = arena.matrix(Arc::clone(costs), from, to);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
}

fn offset_edge_term(list: usize, costs: &Arc<Vec<Vec<i64>>>) -> Reduction {
    let mut arena = ExprArena::default();
    let from = arena.arg(0);
    let to = arena.arg(1);
    let zero = arena.constant(0);
    let base = arena.matrix(Arc::clone(costs), from, to);
    let body = arena.add(base, zero);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
}

fn permutations(values: &mut [i32], start: usize, visit: &mut dyn FnMut(&[i32])) {
    if start == values.len() {
        visit(values);
        return;
    }
    for index in start..values.len() {
        values.swap(start, index);
        permutations(values, start + 1, visit);
        values.swap(start, index);
    }
}

fn route_cost(route: &[i32], costs: &[Vec<i64>]) -> i64 {
    let mut total = 0;
    let mut previous = 0usize;
    for &item in route {
        total += costs[previous][item as usize];
        previous = item as usize;
    }
    total + costs[previous][0]
}

fn brute_two_route_cvrp(costs: &[Vec<i64>], demands: &[i64], capacity: i64) -> i64 {
    let mut items = (1..costs.len() as i32).collect::<Vec<_>>();
    let mut best = i64::MAX;
    permutations(&mut items, 0, &mut |permutation| {
        for cut in 0..=permutation.len() {
            let left = &permutation[..cut];
            let right = &permutation[cut..];
            let load = |route: &[i32]| route.iter().map(|&item| demands[item as usize]).sum::<i64>();
            if load(left) <= capacity && load(right) <= capacity {
                best = best.min(route_cost(left, costs) + route_cost(right, costs));
            }
        }
    });
    best
}

fn vrptw_route_is_feasible(
    route: &[i32],
    travel: &[Vec<i64>],
    earliest: &[i64],
    latest: &[i64],
    service: &[i64],
    demands: &[i64],
    capacity: i64,
) -> bool {
    if route.iter().map(|&customer| demands[customer as usize]).sum::<i64>() > capacity {
        return false;
    }
    let mut departure = earliest[0];
    let mut previous = 0usize;
    for &customer in route {
        let customer = customer as usize;
        let start = earliest[customer].max(departure + travel[previous][customer]);
        if start > latest[customer] {
            return false;
        }
        departure = start + service[customer];
        previous = customer;
    }
    departure + travel[previous][0] <= latest[0]
}

fn brute_vrptw_fleet(travel: &[Vec<i64>], earliest: &[i64], latest: &[i64], service: &[i64], demands: &[i64], capacity: i64) -> usize {
    let mut items = (1..travel.len() as i32).collect::<Vec<_>>();
    let mut best = items.len();
    permutations(&mut items, 0, &mut |permutation| {
        let gaps = permutation.len().saturating_sub(1);
        for cuts in 0..(1usize << gaps) {
            let mut start = 0usize;
            let mut routes = 0usize;
            let mut feasible = true;
            for end in 1..=permutation.len() {
                if end == permutation.len() || cuts & (1usize << (end - 1)) != 0 {
                    routes += 1;
                    feasible &= vrptw_route_is_feasible(&permutation[start..end], travel, earliest, latest, service, demands, capacity);
                    start = end;
                }
            }
            if feasible {
                best = best.min(routes);
            }
        }
    });
    best
}

#[test]
fn additive_assignment_relaxation_is_exact_without_side_constraints() {
    let model = CollectionModel {
        items: vec![1, 2, 3],
        lists: 2,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: vec![item_term(0, &[0, 8, 2, 7], ReduceOp::Sum), item_term(1, &[0, 3, 9, 4], ReduceOp::Sum)],
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let bound = compute(&model, &AtomicBool::new(false)).expect("assignment bound");
    assert_eq!(bound.value, 3 + 2 + 4);
    assert_eq!(bound.method, "assignment relaxation");
}

#[test]
fn used_bins_get_the_capacity_lower_bound() {
    let demands = [0, 4, 4, 4, 4, 4];
    let mut terms = Vec::new();
    let mut constraints = Vec::new();
    for list in 0..4 {
        terms.push(item_term(list, &demands, ReduceOp::Used));
        constraints.push(Constraint { reduction: item_term(list, &demands, ReduceOp::Sum), op: Op::Le, rhs: 8 });
    }
    let model = CollectionModel {
        items: vec![1, 2, 3, 4, 5],
        lists: 4,
        objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
        constraints,
        globals: Vec::new(),
        schedule: None,
    };
    let bound = compute(&model, &AtomicBool::new(false)).expect("capacity bound");
    assert_eq!(bound.value, 3);
    assert_eq!(bound.method, "capacity assignment relaxation");
}

#[test]
fn exact_bin_packing_strengthens_the_volume_bound() {
    let demands = [0, 6, 6, 6, 6];
    let mut terms = Vec::new();
    let mut constraints = Vec::new();
    for list in 0..4 {
        terms.push(item_term(list, &demands, ReduceOp::Used));
        constraints.push(Constraint { reduction: item_term(list, &demands, ReduceOp::Sum), op: Op::Le, rhs: 10 });
    }
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 4,
        objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
        constraints,
        globals: Vec::new(),
        schedule: None,
    };
    let bound = compute(&model, &AtomicBool::new(false)).expect("bin-packing bound");
    assert_eq!(bound.value, 4);
    assert_eq!(bound.method, "bin-packing relaxation");
}

#[test]
fn vrptw_route_cover_certifies_temporal_fleet_pressure() {
    let customers = 4;
    let travel = vec![vec![0; customers + 1]; customers + 1];
    let earliest = vec![0; customers + 1];
    let mut latest = vec![0; customers + 1];
    latest[0] = 10;
    let mut service = vec![1; customers + 1];
    service[0] = 0;
    let demands = vec![0, 1, 1, 1, 1];
    let mut terms = Vec::new();
    let mut constraints = Vec::new();
    for list in 0..customers {
        terms.push(item_term(list, &demands, ReduceOp::Used));
        constraints.push(Constraint { reduction: item_term(list, &demands, ReduceOp::Sum), op: Op::Le, rhs: 10 });
        constraints.push(vrptw_constraint(list, &travel, &earliest, &latest, &service));
    }
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: customers,
        objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
        constraints,
        globals: Vec::new(),
        schedule: None,
    };
    let bound = compute(&model, &AtomicBool::new(false)).expect("VRPTW bound");
    assert_eq!(bound.value, 4);
    assert_eq!(bound.method, "exact VRPTW route-cover dual");
}

#[test]
fn vrptw_duals_stay_below_random_small_optima() {
    let customers = 4usize;
    let mut state = 19u64;
    let mut random = |limit: i64| {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((state >> 32) % limit as u64) as i64
    };
    for _ in 0..32 {
        let coordinates = (0..=customers).map(|_| (random(6), random(6))).collect::<Vec<_>>();
        let travel = coordinates
            .iter()
            .map(|&(x1, y1)| coordinates.iter().map(|&(x2, y2)| (x1 - x2).abs() + (y1 - y2).abs()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut earliest = vec![0; customers + 1];
        let mut latest = vec![100; customers + 1];
        let mut service = vec![0; customers + 1];
        let mut demands = vec![0; customers + 1];
        for customer in 1..=customers {
            earliest[customer] = random(20);
            latest[customer] = earliest[customer] + 12 + random(14);
            service[customer] = 1 + random(4);
            demands[customer] = 1 + random(6);
        }
        let capacity = 8;
        let optimum = brute_vrptw_fleet(&travel, &earliest, &latest, &service, &demands, capacity);
        let mut terms = Vec::new();
        let mut constraints = Vec::new();
        for list in 0..customers {
            terms.push(item_term(list, &demands, ReduceOp::Used));
            constraints.push(Constraint { reduction: item_term(list, &demands, ReduceOp::Sum), op: Op::Le, rhs: capacity });
            constraints.push(vrptw_constraint(list, &travel, &earliest, &latest, &service));
        }
        let model = CollectionModel {
            items: (1..=customers as i32).collect(),
            lists: customers,
            objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
            constraints,
            globals: Vec::new(),
            schedule: None,
        };
        let bound = compute(&model, &AtomicBool::new(false)).expect("VRPTW relaxation");
        assert!(bound.value <= optimum as i64, "dual {} crossed VRPTW optimum {optimum}", bound.value);
    }
}

#[test]
fn held_karp_one_tree_closes_a_symmetric_tsp_relaxation() {
    let costs = vec![vec![0, 10, 10, 10], vec![10, 0, 1, 1], vec![10, 1, 0, 1], vec![10, 1, 1, 0]];
    let model = routing_model(&costs, &[0, 1, 1, 1], 1, None);
    let bound = compute(&model, &AtomicBool::new(false)).expect("routing bound");
    assert_eq!(bound.value, 22);
    assert_eq!(bound.method, "Held-Karp 1-tree");
}

#[test]
fn stabilized_route_columns_strengthen_cvrp_and_never_cross_the_optimum() {
    let costs = vec![vec![0, 10, 10, 10, 10], vec![10, 0, 1, 1, 1], vec![10, 1, 0, 1, 1], vec![10, 1, 1, 0, 1], vec![10, 1, 1, 1, 0]];
    let demands = [0, 1, 1, 1, 1];
    let model = routing_model(&costs, &demands, 2, Some(2));
    let optimum = brute_two_route_cvrp(&costs, &demands, 2);
    let bound = compute(&model, &AtomicBool::new(false)).expect("column bound");
    assert!(bound.value <= optimum);
    assert!(bound.value >= 41, "column dual should dominate the 24-point arc assignment bound");
    assert_eq!(bound.method, "stabilized VRP column generation");
}

#[test]
#[allow(clippy::needless_range_loop)]
fn routing_duals_stay_below_random_small_cvrp_optima() {
    let mut state = 7u64;
    for _ in 0..24 {
        let mut costs = vec![vec![0i64; 6]; 6];
        for i in 0..6 {
            for j in (i + 1)..6 {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let cost = 1 + ((state >> 32) % 30) as i64;
                costs[i][j] = cost;
                costs[j][i] = cost;
            }
        }
        let demands = [0, 1, 1, 1, 1, 1];
        let model = routing_model(&costs, &demands, 2, Some(3));
        let optimum = brute_two_route_cvrp(&costs, &demands, 3);
        let bound = compute(&model, &AtomicBool::new(false)).expect("routing relaxation");
        assert!(bound.value <= optimum, "dual {} crossed optimum {optimum}", bound.value);
    }
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn q_route_master_lp_is_recertified_before_publication() {
    let customers = 17usize;
    let mut costs = vec![vec![1i64; customers + 1]; customers + 1];
    for (node, row) in costs.iter_mut().enumerate() {
        row[node] = 0;
        if node == 0 {
            row.iter_mut().skip(1).for_each(|cost| *cost = 10);
        } else {
            row[0] = 10;
        }
    }
    let demands = vec![1i64; customers + 1];
    let model = routing_model(&costs, &demands, customers, Some(2));
    let controls = LinearControls {
        backend: LinearBackendMode::Amthal,
        root_time: Duration::from_secs(2),
        max_rows: 256,
        ..LinearControls::default()
    };
    let bound = compute_with_linear(&model, controls, &AtomicBool::new(false)).expect("routing bound");

    // Eight two-customer routes and one singleton give a feasible solution of
    // 8 * 21 + 20 = 188. Every published route-master value must stay below it.
    assert!(bound.value <= 188, "certified dual {} crossed a feasible CVRP solution", bound.value);
    assert_eq!(bound.stats.lp_model_status, qayd::search::LinearModelStatus::Ready);
    assert!(bound.stats.lp_solves > 0);
    assert!(bound.stats.lp_certified > 0);
    assert!(bound.stats.lp_root_bound.is_some_and(|lp| lp <= 188));
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn routing_local_search_reports_the_certified_route_lp() {
    let customers = 17usize;
    let mut costs = vec![vec![1i64; customers + 1]; customers + 1];
    for (node, row) in costs.iter_mut().enumerate() {
        row[node] = 0;
        if node == 0 {
            row.iter_mut().skip(1).for_each(|cost| *cost = 10);
        } else {
            row[0] = 10;
        }
    }
    let collection = routing_model(&costs, &vec![1i64; customers + 1], customers, Some(2));
    let semantic = Model::from_collection(&collection);
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        limits: SolveLimits { iterations: Some(4), ..SolveLimits::default() },
        linear: LinearControls {
            backend: LinearBackendMode::Amthal,
            root_time: Duration::from_secs(1),
            max_rows: 256,
            ..LinearControls::default()
        },
        ..SolveRequest::default()
    };
    let budget = SolveBudget::new(None);
    let plan = compile_collection_plan(&semantic, &request, &budget).expect("routing local-search plan");
    let mut sink = EventCallback(|_| Ok::<_, SolveError>(EventControl::Continue));
    let result = solve_collection_plan(&semantic, &plan, &request, &budget, None, None, &mut sink).expect("routing result");
    let stats = result.aggregate_search_stats();

    assert!(stats.lp_solves > 0);
    assert!(stats.lp_certified > 0);
    assert!(stats.lp_root_bound.is_some());
    assert!(result.bounds().first().zip(result.primal()).is_some_and(|(bound, primal)| bound.value <= primal.objectives()[0]));
}

#[test]
fn critical_path_and_resource_energy_bound_schedule_makespan() {
    let model = CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(Schedule {
            intervals: vec![
                IntervalVar { duration: 3, horizon: 20, modes: Vec::new(), optional: false },
                IntervalVar { duration: 4, horizon: 20, modes: Vec::new(), optional: false },
            ],
            precedences: vec![(0, 1)],
            resources: Vec::new(),
            minimize_makespan: true,
        }),
    };
    let bound = compute(&model, &AtomicBool::new(false)).expect("schedule bound");
    assert_eq!(bound.value, 7);
}

#[test]
fn attached_gap_uses_a_certified_dual_only() {
    let costs = vec![vec![0, 10, 10, 10], vec![10, 0, 1, 1], vec![10, 1, 0, 1], vec![10, 1, 1, 0]];
    let model = routing_model(&costs, &[0, 1, 1, 1], 1, None);
    let dual = compute(&model, &AtomicBool::new(false));
    let mut solution = CollectionSolution {
        lists: vec![vec![1, 2, 3]],
        objectives: vec![24],
        feasible: true,
        starts: Vec::new(),
        presences: Vec::new(),
        machines: Vec::new(),
        modes: Vec::new(),
        bound: None,
    };
    attach(&model, &mut solution, dual);
    let report = solution.bound.expect("gap report");
    assert_eq!(report.dual, 22);
    assert_eq!(report.absolute_gap, 2);
    assert!((report.relative_gap - 2.0 / 24.0).abs() < 1e-12);
}

#[test]
fn dual_construction_honors_a_prearmed_cancellation() {
    let costs =
        (0usize..96).map(|from| (0usize..96).map(|to| i64::try_from(from.abs_diff(to)).unwrap()).collect::<Vec<_>>()).collect::<Vec<_>>();
    let demands = vec![1; costs.len()];
    let model = routing_model(&costs, &demands, 4, Some(32));

    assert!(compute(&model, &AtomicBool::new(true)).is_none());
}

#[test]
fn routing_local_search_publishes_its_first_incumbent_before_the_dual_starts() {
    let local_audit = Arc::new(LocalSearchDualAudit::default());
    let _audit = audit_watch_local_search_dual(Arc::clone(&local_audit));

    let costs = vec![vec![0, 4, 7, 6], vec![4, 0, 2, 5], vec![7, 2, 0, 3], vec![6, 5, 3, 0]];
    let collection = routing_model(&costs, &[0, 1, 1, 1], 2, Some(3));
    let semantic = Model::from_collection(&collection);
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 1,
        limits: SolveLimits { iterations: Some(16), ..SolveLimits::default() },
        ..SolveRequest::default()
    };
    let budget = SolveBudget::new(None);
    let plan = compile_collection_plan(&semantic, &request, &budget).expect("routing local-search plan");

    let mut saw_progress = false;
    let mut sink = EventCallback(|event| -> Result<EventControl, SolveError> {
        if let SolveEvent::Progress { engine: EngineKind::RoutingLocalSearch, .. } = event {
            if !saw_progress {
                saw_progress = true;
                assert!(
                    !local_audit.dual_started.load(Ordering::Acquire),
                    "the dual started before the first incumbent publication completed"
                );
            }
        }
        Ok(EventControl::Continue)
    });

    let result = solve_collection_plan(&semantic, &plan, &request, &budget, None, None, &mut sink).expect("routing local-search result");

    assert!(saw_progress, "the routing fixture should publish a feasible incumbent");
    assert!(local_audit.dual_started.load(Ordering::Acquire), "the dual should still run once the first incumbent has been published");
    assert!(
        !local_audit.dual_started_during_progress.load(Ordering::Acquire),
        "the dual crossed the causal publication boundary while the first incumbent callback was still running"
    );
    assert_eq!(result.reports()[0].engine, Some(EngineKind::RoutingLocalSearch));
    assert_eq!(result.bounds().len(), 1, "the certified dual should still be attached when budget permits");
}

#[test]
fn homogeneous_routing_lists_build_one_relaxation_matrix() {
    let _audit = DualAuditGuard::acquire();
    let costs = Arc::new(vec![
        vec![0, 10, 10, 10, 10, 10],
        vec![10, 0, 1, 1, 1, 1],
        vec![10, 1, 0, 1, 1, 1],
        vec![10, 1, 1, 0, 1, 1],
        vec![10, 1, 1, 1, 0, 1],
        vec![10, 1, 1, 1, 1, 0],
    ]);
    let customers = costs.len() - 1;
    let routes = 48usize;
    let model = CollectionModel {
        items: (1..=customers as i32).collect(),
        lists: routes,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: (0..routes).map(|list| shared_edge_term(list, &costs)).collect(),
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };

    let bound = compute(&model, &AtomicBool::new(false)).expect("routing relaxation");

    assert!(!bound.method.is_empty());
    assert_eq!(audit_routing_relaxation_edge_evaluations(), ((customers + 1) * (customers + 1)) as u64);
}

#[test]
fn distinct_routing_expression_families_fall_back_once_per_family() {
    let costs = Arc::new(vec![
        vec![0, 10, 10, 10, 10, 10],
        vec![10, 0, 1, 1, 1, 1],
        vec![10, 1, 0, 1, 1, 1],
        vec![10, 1, 1, 0, 1, 1],
        vec![10, 1, 1, 1, 0, 1],
        vec![10, 1, 1, 1, 1, 0],
    ]);
    let customers = costs.len() - 1;
    let routes = 40usize;
    let baseline = CollectionModel {
        items: (1..=customers as i32).collect(),
        lists: routes,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: (0..routes).map(|list| shared_edge_term(list, &costs)).collect(),
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let baseline_bound = compute(&baseline, &AtomicBool::new(false)).expect("baseline routing relaxation");

    let _audit = DualAuditGuard::acquire();
    audit_reset_routing_relaxation_edge_evaluations();
    let mixed = CollectionModel {
        items: (1..=customers as i32).collect(),
        lists: routes,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: (0..routes)
                .map(|list| if list % 2 == 0 { shared_edge_term(list, &costs) } else { offset_edge_term(list, &costs) })
                .collect(),
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let mixed_bound = compute(&mixed, &AtomicBool::new(false)).expect("mixed routing relaxation");

    assert_eq!(mixed_bound, baseline_bound);
    assert_eq!(audit_routing_relaxation_edge_evaluations(), 2 * ((customers + 1) * (customers + 1)) as u64);
}
