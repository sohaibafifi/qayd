use std::collections::BTreeSet;
use std::sync::{atomic::AtomicBool, Arc};

use crate::model::list::{CollectionModel, Constraint, ExprArena, GlobalConstraint, Iterable, ObjectiveTier, Op, ReduceOp, Reduction};
use crate::search::{self, SearchControl};

use crate::engines::routing::{
    routes_from_successors, solve_collection, solve_collection_with_options, solve_compiled, CapacityCheck, CompiledRouting,
    RoutingOptions, RoutingSpec,
};
use crate::model::list::scan::scan_routing_signature;

#[test]
fn lowers_closed_tsp_to_integer_circuit() {
    let dist = vec![vec![0, 10, 15, 20], vec![10, 0, 35, 25], vec![15, 35, 0, 30], vec![20, 25, 30, 0]];
    let mut arena = ExprArena::default();
    let i = arena.arg(0);
    let j = arena.arg(1);
    let body = arena.matrix(std::sync::Arc::new(dist), i, j);
    let reduction = Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list: 0, start: 0, end: 0 }, arena, body, coeff: 1 };
    let model = CollectionModel {
        items: vec![1, 2, 3],
        lists: 1,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![reduction], max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let mut reports = Vec::new();
    let outcome = solve_collection(&model, 0, &stop, &mut |value| reports.push(value)).expect("supported route lowering");

    assert!(outcome.complete);
    assert!(outcome.solution.feasible);
    assert_eq!(outcome.solution.objectives, vec![80]);
    let mut route = outcome.solution.lists[0].clone();
    route.sort_unstable();
    assert_eq!(route, vec![1, 2, 3]);
    // With the LS warm start the verified incumbent can already be optimal, so
    // the exact search may prove optimality without itself reporting/finding a
    // new solution; only the outcome (optimum + route) is the contract here.
    let _ = &reports;
}

#[test]
fn stopped_routing_construction_returns_a_clean_unknown_outcome() {
    let dist = vec![vec![0, 10, 15, 20], vec![10, 0, 35, 25], vec![15, 35, 0, 30], vec![20, 25, 30, 0]];
    let mut arena = ExprArena::default();
    let i = arena.arg(0);
    let j = arena.arg(1);
    let body = arena.matrix(Arc::new(dist), i, j);
    let model = CollectionModel {
        items: vec![1, 2, 3],
        lists: 1,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: vec![Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list: 0, start: 0, end: 0 }, arena, body, coeff: 1 }],
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    // A compiled routing blueprint contains every capability decision but
    // materializes its mutable solver only when execution starts. A
    // pre-armed execution stop must return a clean UNKNOWN outcome.
    let spec = RoutingSpec::from_model(&model).expect("supported route shape");
    let compiled = CompiledRouting { spec, options: RoutingOptions::default() };

    let stop = AtomicBool::new(true);
    let outcome = solve_compiled(&compiled, 0, &stop, None, &mut |_| {});

    assert!(!outcome.complete);
    assert!(!outcome.solution.feasible);
    assert_eq!(outcome.solution.lists, vec![Vec::<i32>::new()]);
    assert!(outcome.solution.objectives.is_empty());
    assert_eq!(outcome.improvements, 0);
}

#[test]
fn lowers_small_cvrp_to_depot_replicated_circuit() {
    let dist = std::sync::Arc::new(vec![
        vec![0, 1, 1, 1, 1],
        vec![1, 0, 1, 50, 50],
        vec![1, 1, 0, 50, 50],
        vec![1, 50, 50, 0, 1],
        vec![1, 50, 50, 1, 0],
    ]);
    let demand = std::sync::Arc::new(vec![0, 1, 1, 1, 1]);

    let edge = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let j = arena.arg(1);
        let body = arena.matrix(Arc::clone(&dist), i, j);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
    };
    let load = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let body = arena.array(Arc::clone(&demand), i);
        Constraint {
            reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
            op: Op::Le,
            rhs: 2,
        }
    };

    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
        constraints: vec![load(0), load(1)],
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported cvrp lowering");

    assert!(outcome.complete);
    assert!(outcome.solution.feasible);
    assert_eq!(outcome.solution.objectives, vec![6]);
    assert_eq!(outcome.solution.lists.len(), 2);
    let mut served = outcome.solution.lists.iter().flatten().copied().collect::<Vec<_>>();
    served.sort_unstable();
    assert_eq!(served, vec![1, 2, 3, 4]);
    for route in &outcome.solution.lists {
        assert!(route.iter().map(|&node| demand[node as usize]).sum::<i64>() <= 2);
    }
}

// --- brute-force oracles ----------------------------------------------------
//
// These compare the engine's optimum against an independent exhaustive search,
// a stronger guard than the hand-computed-value tests above: they prove the
// engine finds the *true* optimum for the lowering, on arbitrary small matrices.

/// All permutations of `items` (each a distinct visiting order).
fn permutations(items: &[i32]) -> Vec<Vec<i32>> {
    if items.is_empty() {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(i);
        for mut tail in permutations(&rest) {
            tail.insert(0, head);
            out.push(tail);
        }
    }
    out
}

/// Minimum closed-tour cost `depot -> ... -> depot` over all orderings of
/// `items` (directed, so asymmetric matrices are handled). Empty route = 0.
fn min_closed_tour(depot: i32, items: &[i32], dist: &[Vec<i64>]) -> i64 {
    let mut best = i64::MAX;
    for perm in permutations(items) {
        let mut cost = 0i64;
        let mut prev = depot;
        for &node in &perm {
            cost += dist[prev as usize][node as usize];
            prev = node;
        }
        cost += dist[prev as usize][depot as usize];
        best = best.min(cost);
    }
    best
}

#[test]
#[ignore = "benchmark, run with --ignored --nocapture"]
fn measure_two_way() {
    // Membership-routing instance: 2 routes, 8 customers, capacity 4 per route,
    // two same_list pairs, and each route exactly 4 customers. Exercises the
    // reverse member -> succ channel.
    let coords: [(i64, i64); 9] = [(0, 0), (1, 1), (2, 5), (5, 2), (6, 6), (2, 2), (7, 1), (3, 7), (8, 5)];
    let n = coords.len();
    let dist: Vec<Vec<i64>> =
        (0..n).map(|i| (0..n).map(|j| (coords[i].0 - coords[j].0).abs() + (coords[i].1 - coords[j].1).abs()).collect()).collect();
    let items: Vec<i32> = (1..n as i32).collect();
    let dist_arc = Arc::new(dist);
    let demand_arc = Arc::new(vec![1i64; n]);
    let edge = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let j = arena.arg(1);
        let body = arena.matrix(Arc::clone(&dist_arc), i, j);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
    };
    let load = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let body = arena.array(Arc::clone(&demand_arc), i);
        Constraint {
            reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
            op: Op::Le,
            rhs: 4,
        }
    };
    let model = CollectionModel {
        items,
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
        constraints: vec![load(0), load(1), count_eq(0, 4), count_eq(1, 4)],
        globals: vec![GlobalConstraint::SameList { a: 1, b: 5 }, GlobalConstraint::SameList { a: 2, b: 6 }],
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    for two_way in [false, true] {
        for warm_start in [false, true] {
            let options = RoutingOptions { two_way, warm_start, ..RoutingOptions::default() };
            let outcome =
                solve_collection_with_options(&model, 0, &stop, &mut |_| {}, options).expect("supported membership routing model");
            eprintln!(
                "two_way={two_way} warm_start={warm_start} nodes={} failures={} learned_lits={} obj={:?} complete={}",
                outcome.stats.nodes, outcome.stats.failures, outcome.stats.learned_lits, outcome.solution.objectives, outcome.complete,
            );
        }
    }
}

#[test]
#[ignore = "measurement, run explicitly with --ignored --nocapture"]
fn measure_nn_seed() {
    let coords: [(i64, i64); 10] = [(0, 0), (1, 5), (2, 3), (5, 1), (8, 8), (3, 7), (7, 2), (6, 6), (4, 4), (9, 5)];
    let n = coords.len();
    let dist: Vec<Vec<i64>> =
        (0..n).map(|i| (0..n).map(|j| (coords[i].0 - coords[j].0).abs() + (coords[i].1 - coords[j].1).abs()).collect()).collect();
    let items: Vec<i32> = (1..n as i32).collect();
    let model = closed_tsp_model(items, 0, &Arc::new(dist));
    let stop = AtomicBool::new(false);
    for nearest_neighbor in [false, true] {
        let options = RoutingOptions { nearest_neighbor, ..RoutingOptions::default() };
        let outcome = solve_collection_with_options(&model, 0, &stop, &mut |_| {}, options).expect("supported");
        eprintln!(
            "nearest_neighbor={nearest_neighbor} nodes={} failures={} solutions={} obj={:?} complete={}",
            outcome.stats.nodes, outcome.stats.failures, outcome.stats.solutions, outcome.solution.objectives, outcome.complete,
        );
        assert!(outcome.complete);
    }
}

fn closed_tsp_model(items: Vec<i32>, depot: i32, dist: &Arc<Vec<Vec<i64>>>) -> CollectionModel {
    let mut arena = ExprArena::default();
    let i = arena.arg(0);
    let j = arena.arg(1);
    let body = arena.matrix(Arc::clone(dist), i, j);
    let reduction = Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list: 0, start: depot, end: depot }, arena, body, coeff: 1 };
    CollectionModel {
        items,
        lists: 1,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![reduction], max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    }
}

#[test]
fn closed_tsp_matches_brute_force() {
    // depot = node 0, items = nodes 1..=3. Several matrices, including an
    // asymmetric one, each checked against the exhaustive closed-tour minimum.
    let cases: Vec<Vec<Vec<i64>>> = vec![
        vec![vec![0, 10, 15, 20], vec![10, 0, 35, 25], vec![15, 35, 0, 30], vec![20, 25, 30, 0]],
        vec![vec![0, 3, 7, 5], vec![5, 0, 2, 9], vec![6, 4, 0, 1], vec![8, 7, 3, 0]],
        vec![vec![0, 2, 9, 1], vec![1, 0, 6, 4], vec![3, 8, 0, 2], vec![7, 5, 2, 0]],
    ];
    let items = vec![1, 2, 3];
    for dist in cases {
        let brute = min_closed_tour(0, &items, &dist);
        let model = closed_tsp_model(items.clone(), 0, &Arc::new(dist));
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported route lowering");

        assert!(outcome.complete, "exhausted the search");
        assert_eq!(outcome.solution.objectives, vec![brute], "engine optimum equals the exhaustive closed-tour minimum");
        let mut route = outcome.solution.lists[0].clone();
        route.sort_unstable();
        assert_eq!(route, items, "route visits every item exactly once");
    }
}

/// Minimum total cost of partitioning `items` across `n_routes` capacitated
/// closed tours: every assignment of items to routes, each route ordered
/// optimally, capacity respected. `None` if no assignment is feasible.
fn min_cvrp(depot: i32, items: &[i32], dist: &[Vec<i64>], demand: &[i64], n_routes: usize, capacity: i64) -> Option<i64> {
    let n = items.len();
    let combos = (n_routes as u64).pow(n as u32);
    let mut best: Option<i64> = None;
    for combo in 0..combos {
        let mut routes: Vec<Vec<i32>> = vec![Vec::new(); n_routes];
        let mut code = combo;
        for &item in items {
            let route = (code % n_routes as u64) as usize;
            code /= n_routes as u64;
            routes[route].push(item);
        }
        if routes.iter().any(|r| r.iter().map(|&node| demand[node as usize]).sum::<i64>() > capacity) {
            continue;
        }
        let cost: i64 = routes.iter().map(|r| min_closed_tour(depot, r, dist)).sum();
        best = Some(best.map_or(cost, |b| b.min(cost)));
    }
    best
}

#[test]
fn small_cvrp_matches_brute_force() {
    // 2 routes, 4 customers, identical per-route capacity 2 (each customer
    // demands 1). depot = node 0. Engine optimum must equal the exhaustive
    // capacitated multi-route minimum.
    let dist = vec![vec![0, 1, 1, 1, 1], vec![1, 0, 1, 50, 50], vec![1, 1, 0, 50, 50], vec![1, 50, 50, 0, 1], vec![1, 50, 50, 1, 0]];
    let demand = vec![0i64, 1, 1, 1, 1];
    let items = vec![1, 2, 3, 4];
    let brute = min_cvrp(0, &items, &dist, &demand, 2, 2).expect("a feasible packing exists");

    let dist_arc = Arc::new(dist);
    let demand_arc = Arc::new(demand.clone());
    let edge = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let j = arena.arg(1);
        let body = arena.matrix(Arc::clone(&dist_arc), i, j);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
    };
    let load = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let body = arena.array(Arc::clone(&demand_arc), i);
        Constraint {
            reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
            op: Op::Le,
            rhs: 2,
        }
    };
    let model = CollectionModel {
        items: items.clone(),
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
        constraints: vec![load(0), load(1)],
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported cvrp lowering");

    assert!(outcome.complete, "exhausted the search");
    assert_eq!(outcome.solution.objectives, vec![brute], "engine optimum equals the exhaustive CVRP minimum");
    let mut served = outcome.solution.lists.iter().flatten().copied().collect::<Vec<_>>();
    served.sort_unstable();
    assert_eq!(served, items, "every customer served exactly once");
}

/// Like `min_cvrp`, but discard any assignment that splits a `same_list` pair.
fn min_cvrp_same_list(
    depot: i32,
    items: &[i32],
    dist: &[Vec<i64>],
    demand: &[i64],
    n_routes: usize,
    capacity: i64,
    same: &[(i32, i32)],
) -> Option<i64> {
    let n = items.len();
    let combos = (n_routes as u64).pow(n as u32);
    let mut best: Option<i64> = None;
    for combo in 0..combos {
        let mut routes: Vec<Vec<i32>> = vec![Vec::new(); n_routes];
        let mut segment_by_item: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        let mut code = combo;
        for &item in items {
            let route = (code % n_routes as u64) as usize;
            code /= n_routes as u64;
            routes[route].push(item);
            segment_by_item.insert(item, route);
        }
        if same.iter().any(|&(a, b)| segment_by_item[&a] != segment_by_item[&b]) {
            continue;
        }
        if routes.iter().any(|r| r.iter().map(|&node| demand[node as usize]).sum::<i64>() > capacity) {
            continue;
        }
        let cost: i64 = routes.iter().map(|r| min_closed_tour(depot, r, dist)).sum();
        best = Some(best.map_or(cost, |b| b.min(cost)));
    }
    best
}

/// Build a 2-route CVRP-shaped model with capacity 2 and optional `same_list`
/// pairs, over the shared `dist`/`demand` tables.
fn cvrp_same_list_model(
    items: Vec<i32>,
    dist: &Arc<Vec<Vec<i64>>>,
    demand: &Arc<Vec<i64>>,
    capacity: i64,
    same: &[(i32, i32)],
) -> CollectionModel {
    let edge = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let j = arena.arg(1);
        let body = arena.matrix(Arc::clone(dist), i, j);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
    };
    let load = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let body = arena.array(Arc::clone(demand), i);
        Constraint {
            reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
            op: Op::Le,
            rhs: capacity,
        }
    };
    CollectionModel {
        items,
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
        constraints: vec![load(0), load(1)],
        globals: same.iter().map(|&(a, b)| GlobalConstraint::SameList { a, b }).collect(),
        schedule: None,
    }
}

#[test]
fn same_list_changes_cvrp_optimum_and_matches_brute_force() {
    // Unconstrained optimum pairs {1,2} and {3,4} (cost 6). same_list(1, 3)
    // forces 1 and 3 together, which is far more expensive. The engine optimum
    // must equal the same-list-filtered exhaustive minimum, and must exceed the
    // unconstrained optimum (proving the channel actually bites).
    let dist = vec![vec![0, 1, 1, 1, 1], vec![1, 0, 1, 50, 50], vec![1, 1, 0, 50, 50], vec![1, 50, 50, 0, 1], vec![1, 50, 50, 1, 0]];
    let demand = vec![0i64, 1, 1, 1, 1];
    let items = vec![1, 2, 3, 4];
    let unconstrained = min_cvrp(0, &items, &dist, &demand, 2, 2).expect("feasible");
    let same = [(1, 3)];
    let brute = min_cvrp_same_list(0, &items, &dist, &demand, 2, 2, &same).expect("feasible under same_list");
    assert!(brute > unconstrained, "same_list(1,3) is more expensive than the free optimum");

    let model = cvrp_same_list_model(items.clone(), &Arc::new(dist), &Arc::new(demand), 2, &same);
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported same_list cvrp");

    assert!(outcome.complete, "exhausted the search");
    assert_eq!(outcome.solution.objectives, vec![brute], "engine optimum equals the same-list-filtered minimum");
    let segment_of_item = |item: i32| outcome.solution.lists.iter().position(|r| r.contains(&item)).expect("served");
    assert_eq!(segment_of_item(1), segment_of_item(3), "same_list keeps 1 and 3 in the same route");
}

#[test]
fn same_list_with_capacity_can_be_unsat() {
    // Customer 1 demands the whole capacity (2). same_list(1, 2) forces 1 and 2
    // into one route, whose combined demand 3 exceeds capacity 2: infeasible.
    let dist = vec![vec![0, 1, 1, 1, 1], vec![1, 0, 1, 1, 1], vec![1, 1, 0, 1, 1], vec![1, 1, 1, 0, 1], vec![1, 1, 1, 1, 0]];
    let demand = vec![0i64, 2, 1, 1, 1];
    let items = vec![1, 2, 3, 4];
    let same = [(1, 2)];
    assert!(min_cvrp_same_list(0, &items, &dist, &demand, 2, 2, &same).is_none(), "no feasible same_list packing exists");

    let model = cvrp_same_list_model(items, &Arc::new(dist), &Arc::new(demand), 2, &same);
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported same_list cvrp");

    assert!(outcome.complete, "search exhausted");
    assert!(!outcome.solution.feasible, "same_list(1,2) over-capacity is UNSAT");
}

/// General exhaustive multi-route minimum under optional capacity, same-list,
/// and per-route length filters. `None` if no assignment is feasible.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn min_routes(
    depot: i32,
    items: &[i32],
    dist: &[Vec<i64>],
    demand: &[i64],
    n_routes: usize,
    capacity: Option<i64>,
    same: &[(i32, i32)],
    length: Option<(usize, usize)>,
    item_sum: Option<(&[(i32, i64)], i64, i64)>,
) -> Option<i64> {
    let n = items.len();
    let combos = (n_routes as u64).pow(n as u32);
    let mut best: Option<i64> = None;
    for combo in 0..combos {
        let mut routes: Vec<Vec<i32>> = vec![Vec::new(); n_routes];
        let mut segment_by_item: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        let mut code = combo;
        for &item in items {
            let route = (code % n_routes as u64) as usize;
            code /= n_routes as u64;
            routes[route].push(item);
            segment_by_item.insert(item, route);
        }
        if same.iter().any(|&(a, b)| segment_by_item[&a] != segment_by_item[&b]) {
            continue;
        }
        if let Some(cap) = capacity {
            if routes.iter().any(|r| r.iter().map(|&node| demand[node as usize]).sum::<i64>() > cap) {
                continue;
            }
        }
        if let Some((lo, hi)) = length {
            if routes.iter().any(|r| r.len() < lo || r.len() > hi) {
                continue;
            }
        }
        if let Some((weights, lo, hi)) = item_sum {
            let wmap: std::collections::HashMap<i32, i64> = weights.iter().copied().collect();
            if routes.iter().any(|r| {
                let sum: i64 = r.iter().map(|&node| wmap.get(&node).copied().unwrap_or(0)).sum();
                sum < lo || sum > hi
            }) {
                continue;
            }
        }
        let cost: i64 = routes.iter().map(|r| min_closed_tour(depot, r, dist)).sum();
        best = Some(best.map_or(cost, |b| b.min(cost)));
    }
    best
}

fn successor_cycle_from_order(n: usize, order: &[usize]) -> Vec<i32> {
    let mut succ = vec![0i32; n];
    let mut prev = 0usize;
    for &node in order {
        succ[prev] = node as i32;
        prev = node;
    }
    succ[prev] = 0;
    succ
}

fn depot_symmetry_holds(succ: &[i32], depot_count: usize) -> bool {
    let n = succ.len();
    succ[..depot_count]
        .iter()
        .map(|&head| {
            let head = head as usize;
            if head < depot_count {
                n
            } else {
                head
            }
        })
        .collect::<Vec<_>>()
        .windows(2)
        .all(|w| w[0] <= w[1])
}

fn routes_satisfy_spec(spec: &RoutingSpec, routes: &[Vec<i32>], succ: &[i32]) -> bool {
    if let Some(capacity) = &spec.capacity {
        let demands = spec.node_demands(capacity);
        if !matches!(crate::engines::routing::route_capacity_check(spec.depot_count, &demands, capacity.capacity, succ), CapacityCheck::Ok)
        {
            return false;
        }
    }
    let segment_of = |item: i32| routes.iter().position(|route| route.contains(&item));
    if spec.same_list.iter().any(|&(a, b)| segment_of(a) != segment_of(b)) {
        return false;
    }
    if let Some((min, max)) = spec.length {
        if routes.iter().any(|route| route.len() < min || route.len() > max) {
            return false;
        }
    }
    if let Some((weights, min, max)) = &spec.item_sum {
        if routes.iter().any(|route| {
            let sum =
                route.iter().map(|item| weights.iter().find_map(|&(x, weight)| (x == *item).then_some(weight)).unwrap_or(0)).sum::<i64>();
            sum < *min || sum > *max
        }) {
            return false;
        }
    }
    true
}

fn chronological_successor_set(spec: &RoutingSpec) -> BTreeSet<Vec<i32>> {
    let mut nodes = Vec::with_capacity(spec.depot_count + spec.items.len());
    nodes.extend((0..spec.depot_count).map(|_| spec.depot));
    nodes.extend_from_slice(&spec.items);
    let n = nodes.len();
    let mut order: Vec<usize> = (1..n).collect();
    let mut out = BTreeSet::new();
    fn rec(pos: usize, order: &mut [usize], nodes: &[i32], spec: &RoutingSpec, out: &mut BTreeSet<Vec<i32>>) {
        if pos == order.len() {
            let succ = successor_cycle_from_order(nodes.len(), order);
            if !depot_symmetry_holds(&succ, spec.depot_count) {
                return;
            }
            let Some(routes) = routes_from_successors(nodes, spec.depot_count, &succ) else {
                return;
            };
            if routes_satisfy_spec(spec, &routes, &succ) {
                out.insert(succ);
            }
            return;
        }
        for i in pos..order.len() {
            order.swap(pos, i);
            rec(pos + 1, order, nodes, spec, out);
            order.swap(pos, i);
        }
    }
    rec(0, &mut order, &nodes, spec, &mut out);
    out
}

fn cdcl_successor_set(model: &CollectionModel) -> BTreeSet<Vec<i32>> {
    let spec = RoutingSpec::from_model(model).expect("supported route lowering");
    let stop = AtomicBool::new(false);
    let mut lowered = spec.lower(RoutingOptions::default(), &stop).expect("route lowering completes");
    let n = lowered.nodes.len();
    let mut out = BTreeSet::new();
    search::solve(&mut lowered.solver, &lowered.search_vars, |solver| {
        out.insert(lowered.search_vars[..n].iter().map(|&var| solver.store.value(var)).collect::<Vec<_>>());
        SearchControl::Continue
    });
    out
}

fn count_eq(list: usize, k: i64) -> Constraint {
    let mut arena = ExprArena::default();
    let body = arena.constant(1); // count of items present = route length
    Constraint { reduction: Reduction { op: ReduceOp::Count, iterable: Iterable::Items(list), arena, body, coeff: 1 }, op: Op::Eq, rhs: k }
}

#[test]
fn list_len_exact_route_size_matches_brute_force() {
    // Free optimum is unbalanced (one customer alone), but `list_len` forces
    // exactly 2 customers per route. The engine optimum must equal the
    // size-filtered exhaustive minimum and exceed the free optimum.
    let dist = vec![vec![0, 1, 10, 10, 10], vec![1, 0, 10, 10, 10], vec![10, 10, 0, 1, 1], vec![10, 10, 1, 0, 1], vec![10, 10, 1, 1, 0]];
    let zero = vec![0i64; 5];
    let items = vec![1, 2, 3, 4];
    let free = min_routes(0, &items, &dist, &zero, 2, None, &[], None, None).expect("feasible");
    let balanced = min_routes(0, &items, &dist, &zero, 2, None, &[], Some((2, 2)), None).expect("feasible balanced");
    assert!(balanced > free, "forcing two-per-route is costlier than the free optimum");

    let dist_arc = Arc::new(dist);
    let edge = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let j = arena.arg(1);
        let body = arena.matrix(Arc::clone(&dist_arc), i, j);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
    };
    let model = CollectionModel {
        items: items.clone(),
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
        constraints: vec![count_eq(0, 2), count_eq(1, 2)],
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported route length model");

    assert!(outcome.complete, "exhausted the search");
    assert_eq!(outcome.solution.objectives, vec![balanced], "engine optimum equals the size-filtered minimum");
    for route in &outcome.solution.lists {
        assert_eq!(route.len(), 2, "every route holds exactly two customers");
    }
}

#[test]
fn same_list_length_and_capacity_compose() {
    // same_list(1,3) + each route exactly 2 + capacity 2: the three constraints
    // coexist; the engine optimum equals the jointly-filtered exhaustive minimum.
    let dist = vec![vec![0, 1, 1, 1, 1], vec![1, 0, 1, 50, 50], vec![1, 1, 0, 50, 50], vec![1, 50, 50, 0, 1], vec![1, 50, 50, 1, 0]];
    let demand = vec![0i64, 1, 1, 1, 1];
    let items = vec![1, 2, 3, 4];
    let same = [(1, 3)];
    let brute = min_routes(0, &items, &dist, &demand, 2, Some(2), &same, Some((2, 2)), None).expect("feasible");

    let dist_arc = Arc::new(dist);
    let demand_arc = Arc::new(demand);
    let edge = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let j = arena.arg(1);
        let body = arena.matrix(Arc::clone(&dist_arc), i, j);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
    };
    let load = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let body = arena.array(Arc::clone(&demand_arc), i);
        Constraint {
            reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
            op: Op::Le,
            rhs: 2,
        }
    };
    let model = CollectionModel {
        items: items.clone(),
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
        constraints: vec![load(0), load(1), count_eq(0, 2), count_eq(1, 2)],
        globals: vec![GlobalConstraint::SameList { a: 1, b: 3 }],
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported combined model");

    assert!(outcome.complete, "exhausted the search");
    assert_eq!(outcome.solution.objectives, vec![brute], "engine optimum equals the jointly-filtered minimum");
    let segment_of_item = |item: i32| outcome.solution.lists.iter().position(|r| r.contains(&item)).expect("served");
    assert_eq!(segment_of_item(1), segment_of_item(3), "same_list keeps 1 and 3 together");
    for route in &outcome.solution.lists {
        assert_eq!(route.len(), 2, "every route holds exactly two customers");
    }
}

#[test]
fn route_item_sum_ge_matches_brute_force() {
    // Each route must reach a value sum >= 11 (a non-capacity `Sum`, so it routes
    // through the membership channel, not the MTZ load). The cheap free optimum
    // groups the two low-value customers together, which the bound forbids, so
    // the engine optimum must equal the value-filtered exhaustive minimum and
    // exceed the free optimum.
    let dist = vec![vec![0, 1, 1, 1, 1], vec![1, 0, 1, 50, 50], vec![1, 1, 0, 50, 50], vec![1, 50, 50, 0, 1], vec![1, 50, 50, 1, 0]];
    let value = vec![0i64, 1, 1, 10, 10]; // indexed by node; customers 3,4 are high value
    let weights = vec![(1i32, 1i64), (2, 1), (3, 10), (4, 10)];
    let items = vec![1, 2, 3, 4];
    let zero = vec![0i64; 5];
    let free = min_routes(0, &items, &dist, &zero, 2, None, &[], None, None).expect("feasible");
    let bound = (weights.as_slice(), 11i64, i64::MAX / 4);
    let constrained = min_routes(0, &items, &dist, &zero, 2, None, &[], None, Some(bound)).expect("feasible under value bound");
    assert!(constrained > free, "the per-route value floor is costlier than the free optimum");

    let dist_arc = Arc::new(dist);
    let value_arc = Arc::new(value);
    let edge = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let j = arena.arg(1);
        let body = arena.matrix(Arc::clone(&dist_arc), i, j);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
    };
    let value_sum = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let body = arena.array(Arc::clone(&value_arc), i);
        Constraint {
            reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
            op: Op::Ge,
            rhs: 11,
        }
    };
    let model = CollectionModel {
        items: items.clone(),
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
        constraints: vec![value_sum(0), value_sum(1)],
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported route item-sum model");

    assert!(outcome.complete, "exhausted the search");
    assert_eq!(outcome.solution.objectives, vec![constrained], "engine optimum equals the value-filtered minimum");
    for route in &outcome.solution.lists {
        let sum: i64 = route.iter().map(|&node| value_arc[node as usize]).sum();
        assert!(sum >= 11, "every route reaches the value floor");
    }
}

#[test]
fn routing_cdcl_enumerates_chronological_successor_set() {
    // A small mixed routing model: circuit + depot symmetry + capacity + channel
    // constraints (`same_list`, homogeneous length, homogeneous item_sum). CDCL
    // enumeration uses all routing explanations; the chronological set is an
    // independent permutation scan over successor cycles with the same filters.
    let dist = vec![vec![0, 1, 2, 3, 4], vec![1, 0, 1, 5, 5], vec![2, 1, 0, 5, 1], vec![3, 5, 5, 0, 1], vec![4, 5, 1, 1, 0]];
    let demand = vec![0i64, 1, 1, 1, 1];
    let value = vec![0i64, 1, 10, 10, 1];
    let items = vec![1, 2, 3, 4];

    let dist_arc = Arc::new(dist);
    let demand_arc = Arc::new(demand);
    let value_arc = Arc::new(value);
    let edge = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let j = arena.arg(1);
        let body = arena.matrix(Arc::clone(&dist_arc), i, j);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
    };
    let load = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let body = arena.array(Arc::clone(&demand_arc), i);
        Constraint {
            reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
            op: Op::Le,
            rhs: 2,
        }
    };
    let value_sum = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let body = arena.array(Arc::clone(&value_arc), i);
        Constraint {
            reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
            op: Op::Ge,
            rhs: 11,
        }
    };
    let model = CollectionModel {
        items,
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
        constraints: vec![load(0), load(1), count_eq(0, 2), count_eq(1, 2), value_sum(0), value_sum(1)],
        globals: vec![GlobalConstraint::SameList { a: 1, b: 3 }],
        schedule: None,
    };
    let spec = RoutingSpec::from_model(&model).expect("supported mixed routing model");
    let chronological = chronological_successor_set(&spec);
    assert!(!chronological.is_empty(), "mixed routing model has feasible successor tours");

    let cdcl = cdcl_successor_set(&model);
    assert_eq!(cdcl, chronological, "CDCL routing enumeration matches chronological successor enumeration");
}

#[test]
fn route_capacity_overflow_is_unsat() {
    // 3 customers each demand 2, capacity 3, 2 routes: any route with two or more
    // customers overflows (>= 4 > 3), but three customers across two routes force
    // one such route. The explained capacity check must make this infeasible,
    // matching the exhaustive (capacity-filtered) result.
    let dist = vec![vec![0, 1, 1, 1], vec![1, 0, 1, 1], vec![1, 1, 0, 1], vec![1, 1, 1, 0]];
    let demand = vec![0i64, 2, 2, 2];
    let items = vec![1, 2, 3];
    assert!(min_routes(0, &items, &dist, &demand, 2, Some(3), &[], None, None).is_none(), "no capacity-feasible split exists");

    let dist_arc = Arc::new(dist);
    let demand_arc = Arc::new(demand);
    let edge = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let j = arena.arg(1);
        let body = arena.matrix(Arc::clone(&dist_arc), i, j);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
    };
    let load = |list| {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let body = arena.array(Arc::clone(&demand_arc), i);
        Constraint {
            reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
            op: Op::Le,
            rhs: 3,
        }
    };
    let model = CollectionModel {
        items,
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
        constraints: vec![load(0), load(1)],
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported cvrp lowering");

    assert!(outcome.complete, "exhausted the search");
    assert!(!outcome.solution.feasible, "capacity overflow makes the model infeasible");
}

// --- scan (resource-extension) oracles --------------------------------------

/// Time-window lateness scan: `step = max(earliest, acc + travel) + service`,
/// emitting `max(0, end - latest)`. Mirrors `tests/collection.rs::lateness_scan`.
fn lateness_scan(list: usize, dist: &[Vec<i64>], earliest: &[i64], latest: &[i64], service: &[i64], depot: i32) -> Reduction {
    let mut arena = ExprArena::default();
    let cur = arena.arg(0);
    let acc = arena.arg(1);
    let prev = arena.arg(2);
    let travel = arena.matrix(Arc::new(dist.to_vec()), prev, cur);
    let arrive = arena.add(acc, travel);
    let earl = arena.array(Arc::new(earliest.to_vec()), cur);
    let start = arena.max(earl, arrive);
    let svc = arena.array(Arc::new(service.to_vec()), cur);
    let step = arena.add(start, svc);
    let cur_e = arena.arg(0);
    let end = arena.arg(1);
    let lat = arena.array(Arc::new(latest.to_vec()), cur_e);
    let over = arena.sub(end, lat);
    let zero = arena.constant(0);
    let emit = arena.max(zero, over);
    Reduction {
        op: ReduceOp::Sum,
        iterable: Iterable::Scan { list, init: 0, boundary: depot, step, end: None },
        arena,
        body: emit,
        coeff: 1,
    }
}

/// Oracle replaying the lateness scan over an ordered route.
fn lateness_sum(order: &[i32], dist: &[Vec<i64>], earliest: &[i64], latest: &[i64], service: &[i64], depot: i32) -> i64 {
    let mut acc = 0i64;
    let mut prev = depot;
    let mut sum = 0i64;
    for &c in order {
        let end = earliest[c as usize].max(acc + dist[prev as usize][c as usize]) + service[c as usize];
        sum += (end - latest[c as usize]).max(0);
        acc = end;
        prev = c;
    }
    sum
}

fn ordered_closed_cost(depot: i32, order: &[i32], dist: &[Vec<i64>]) -> i64 {
    let mut cost = 0i64;
    let mut prev = depot;
    for &c in order {
        cost += dist[prev as usize][c as usize];
        prev = c;
    }
    cost + dist[prev as usize][depot as usize]
}

fn op_ok(v: i64, op: Op, rhs: i64) -> bool {
    match op {
        Op::Le => v <= rhs,
        Op::Ge => v >= rhs,
        Op::Eq => v == rhs,
    }
}

/// Exhaustive minimum total distance over two-route partitions where each
/// route has a distance-optimal ordering meeting its per-route scan bound (and
/// optional capacity). The scan is filtered per route, matching the LS spec.
#[allow(clippy::too_many_arguments)]
fn min_two_routes_scan(
    items: &[i32],
    dist: &[Vec<i64>],
    demand: &[i64],
    capacity: Option<i64>,
    earliest: &[i64],
    latest: &[i64],
    service: &[i64],
    depot: i32,
    op: Op,
    rhs: i64,
) -> Option<i64> {
    let best_route = |route: &[i32]| -> Option<i64> {
        if let Some(cap) = capacity {
            if route.iter().map(|&c| demand[c as usize]).sum::<i64>() > cap {
                return None;
            }
        }
        permutations(route)
            .iter()
            .filter(|p| op_ok(lateness_sum(p, dist, earliest, latest, service, depot), op, rhs))
            .map(|p| ordered_closed_cost(depot, p, dist))
            .min()
    };
    let n = items.len();
    let mut best: Option<i64> = None;
    for code in 0..(1u32 << n) {
        let mut routes = [Vec::new(), Vec::new()];
        for (i, &it) in items.iter().enumerate() {
            routes[((code >> i) & 1) as usize].push(it);
        }
        if let (Some(a), Some(b)) = (best_route(&routes[0]), best_route(&routes[1])) {
            best = Some(best.map_or(a + b, |x| x.min(a + b)));
        }
    }
    best
}

fn line_dist(positions: &[i64]) -> Vec<Vec<i64>> {
    positions.iter().map(|&i| positions.iter().map(|&j| (i - j).abs()).collect()).collect()
}

fn scan_edge(dist: &Arc<Vec<Vec<i64>>>, list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let i = arena.arg(0);
    let j = arena.arg(1);
    let body = arena.matrix(Arc::clone(dist), i, j);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
}

fn scan_load(demand: &Arc<Vec<i64>>, list: usize, rhs: i64) -> Constraint {
    let mut arena = ExprArena::default();
    let i = arena.arg(0);
    let body = arena.array(Arc::clone(demand), i);
    Constraint { reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 }, op: Op::Le, rhs }
}

#[test]
fn scan_cvrptw_matches_brute_force() {
    // P2 shape: cumulative service+travel time with hard time windows (lateness
    // sum <= 0) plus capacity 2. Customers 1,2 share a deadline (5) they cannot
    // both meet on one route, so a feasible plan must split them. Proven OPTIMAL.
    let dist = line_dist(&[0, 2, 4, 6, 8]);
    let demand = vec![0i64, 1, 1, 1, 1];
    let service = vec![0i64, 1, 1, 1, 1];
    let earliest = vec![0i64, 0, 0, 0, 0];
    let latest = vec![0i64, 5, 5, 20, 20];
    let items = vec![1, 2, 3, 4];
    let depot = 0;
    let brute =
        min_two_routes_scan(&items, &dist, &demand, Some(2), &earliest, &latest, &service, depot, Op::Le, 0).expect("feasible plan exists");

    let dist_arc = Arc::new(dist.clone());
    let demand_arc = Arc::new(demand.clone());
    let mut constraints = vec![scan_load(&demand_arc, 0, 2), scan_load(&demand_arc, 1, 2)];
    for l in 0..2 {
        constraints.push(Constraint { reduction: lateness_scan(l, &dist, &earliest, &latest, &service, depot), op: Op::Le, rhs: 0 });
    }
    let model = CollectionModel {
        items: items.clone(),
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![scan_edge(&dist_arc, 0), scan_edge(&dist_arc, 1)], max_terms: None }],
        constraints,
        globals: Vec::new(),
        schedule: None,
    };

    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported cvrptw scan lowering");
    assert!(outcome.complete, "exhausted the search");
    assert!(outcome.solution.feasible, "a time-window-feasible plan exists");
    assert_eq!(outcome.solution.objectives, vec![brute], "exact optimum equals the tw-filtered exhaustive minimum");
    for route in &outcome.solution.lists {
        assert!(!(route.contains(&1) && route.contains(&2)), "1 and 2 cannot share a route");
        assert!(lateness_sum(route, &dist, &earliest, &latest, &service, depot) <= 0, "each route meets its hard windows");
    }

    // LS-vs-exact: local search reaches the same optimum on this enumerable model.
    let ls = crate::engines::ls::lists::solve_collection_capped(&model, 0, &stop, 50_000, None, &mut |_| {});
    assert!(ls.feasible, "LS finds a feasible plan");
    assert_eq!(ls.objectives, vec![brute], "LS agrees with the exact optimum");
}

#[test]
fn scan_bound_is_per_route_not_fleet_total() {
    // Every arc costs 1; each customer needs service 1 and has deadline 1, so a
    // route's k-th stop finishes at time 2k and is (2k-1) late. Any 2-2 split
    // gives each route lateness 1+3 = 4 <= 5 (feasible per route) but a fleet
    // total of 8 > 5. A global-sum lowering would wrongly reject every split and
    // return UNSAT; a correct per-route lowering keeps it feasible.
    let dist = vec![vec![0, 1, 1, 1, 1], vec![1, 0, 1, 1, 1], vec![1, 1, 0, 1, 1], vec![1, 1, 1, 0, 1], vec![1, 1, 1, 1, 0]];
    let service = vec![0i64, 1, 1, 1, 1];
    let earliest = vec![0i64, 0, 0, 0, 0];
    let latest = vec![0i64, 1, 1, 1, 1];
    let demand = vec![0i64; 5];
    let items = vec![1, 2, 3, 4];
    let depot = 0;
    let rhs = 5;
    let brute = min_two_routes_scan(&items, &dist, &demand, None, &earliest, &latest, &service, depot, Op::Le, rhs).expect("feasible");

    let dist_arc = Arc::new(dist.clone());
    let mut constraints = Vec::new();
    for l in 0..2 {
        constraints.push(Constraint { reduction: lateness_scan(l, &dist, &earliest, &latest, &service, depot), op: Op::Le, rhs });
    }
    let model = CollectionModel {
        items: items.clone(),
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![scan_edge(&dist_arc, 0), scan_edge(&dist_arc, 1)], max_terms: None }],
        constraints,
        globals: Vec::new(),
        schedule: None,
    };

    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported per-route scan lowering");
    assert!(outcome.complete, "exhausted the search");
    assert!(outcome.solution.feasible, "a per-route-feasible plan exists (a global-sum lowering would be UNSAT here)");
    assert_eq!(outcome.solution.objectives, vec![brute], "exact optimum equals the per-route-filtered minimum");
    let sums: Vec<i64> = outcome.solution.lists.iter().map(|r| lateness_sum(r, &dist, &earliest, &latest, &service, depot)).collect();
    assert!(sums.iter().all(|&s| s <= rhs), "every route respects the per-route bound");
    assert!(sums.iter().sum::<i64>() > rhs, "the fleet total exceeds rhs: the scan is enforced per route, not globally");
}

#[test]
fn scan_single_route_unsat() {
    // One vehicle, two customers, each with deadline 1 and service 1 over unit
    // arcs: whichever is served first already finishes at time 2 (1 late), so no
    // ordering meets the hard windows. The exact search must prove UNSAT.
    let dist = vec![vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]];
    let service = vec![0i64, 1, 1];
    let earliest = vec![0i64, 0, 0];
    let latest = vec![0i64, 1, 1];
    let items = vec![1, 2];
    let depot = 0;

    let dist_arc = Arc::new(dist.clone());
    let model = CollectionModel {
        items: items.clone(),
        lists: 1,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![scan_edge(&dist_arc, 0)], max_terms: None }],
        constraints: vec![Constraint { reduction: lateness_scan(0, &dist, &earliest, &latest, &service, depot), op: Op::Le, rhs: 0 }],
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported single-route scan lowering");
    assert!(outcome.complete, "exhausted the search");
    assert!(!outcome.solution.feasible, "no ordering meets both hard windows");
}

#[test]
fn scan_multiple_per_route_compose() {
    // Two identical lateness scans per route (the Python API posts one scan per
    // customer time window, so a route commonly carries several). The exact
    // optimum must still equal the single-scan brute-force minimum -- the extra
    // resource thread must not corrupt the result.
    let dist = line_dist(&[0, 2, 4, 6, 8]);
    let demand = vec![0i64, 1, 1, 1, 1];
    let service = vec![0i64, 1, 1, 1, 1];
    let earliest = vec![0i64, 0, 0, 0, 0];
    let latest = vec![0i64, 5, 5, 20, 20];
    let items = vec![1, 2, 3, 4];
    let depot = 0;
    let brute =
        min_two_routes_scan(&items, &dist, &demand, Some(2), &earliest, &latest, &service, depot, Op::Le, 0).expect("feasible plan exists");

    let dist_arc = Arc::new(dist.clone());
    let demand_arc = Arc::new(demand.clone());
    let mut constraints = vec![scan_load(&demand_arc, 0, 2), scan_load(&demand_arc, 1, 2)];
    for l in 0..2 {
        for _ in 0..2 {
            constraints.push(Constraint { reduction: lateness_scan(l, &dist, &earliest, &latest, &service, depot), op: Op::Le, rhs: 0 });
        }
    }
    let model = CollectionModel {
        items: items.clone(),
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![scan_edge(&dist_arc, 0), scan_edge(&dist_arc, 1)], max_terms: None }],
        constraints,
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported multi-scan lowering");
    assert!(outcome.complete, "exhausted the search");
    assert!(outcome.solution.feasible, "a feasible plan exists");
    assert_eq!(outcome.solution.objectives, vec![brute], "duplicate scans are idempotent");
}

// --- compiler capability ----------------------------------------------------

/// A two-route edge-objective model carrying one scan constraint builder per
/// route, for routing compiler checks.
fn scan_dispatch_model(scan: impl Fn(usize) -> Option<Constraint>) -> CollectionModel {
    let dist_arc = Arc::new(line_dist(&[0, 2, 4, 6, 8]));
    let mut constraints = Vec::new();
    for l in 0..2 {
        if let Some(c) = scan(l) {
            constraints.push(c);
        }
    }
    CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![scan_edge(&dist_arc, 0), scan_edge(&dist_arc, 1)], max_terms: None }],
        constraints,
        globals: Vec::new(),
        schedule: None,
    }
}

fn tw_scan_constraint(l: usize) -> Constraint {
    let dist = line_dist(&[0, 2, 4, 6, 8]);
    let service = vec![0i64, 1, 1, 1, 1];
    let earliest = vec![0i64, 0, 0, 0, 0];
    let latest = vec![0i64, 5, 5, 20, 20];
    Constraint { reduction: lateness_scan(l, &dist, &earliest, &latest, &service, 0), op: Op::Le, rhs: 0 }
}

#[test]
fn compiler_accepts_supported_scan() {
    let model = scan_dispatch_model(|l| Some(tw_scan_constraint(l)));
    assert!(RoutingSpec::from_model(&model).is_some());
}

#[test]
fn compiler_accepts_multiple_homogeneous_scans() {
    // Each route carries two identical scans: still homogeneous, still exact.
    let model = scan_dispatch_model(|l| Some(tw_scan_constraint(l)));
    let mut model = model;
    for l in 0..2 {
        model.constraints.push(tw_scan_constraint(l));
    }
    assert!(RoutingSpec::from_model(&model).is_some());
}

#[test]
fn compiler_rejects_non_homogeneous_scan() {
    // Scan on route 0 only: route-specific, so unsafe under depot symmetry.
    let model = scan_dispatch_model(|l| (l == 0).then(|| tw_scan_constraint(l)));
    assert!(RoutingSpec::from_model(&model).is_none());
}

#[test]
fn compiler_rejects_accumulator_in_divisor() {
    let divisor_scan = |l: usize| {
        let mut arena = ExprArena::default();
        let cur = arena.arg(0);
        let acc = arena.arg(1);
        let step = arena.div(cur, acc); // divides by the accumulator: rejected
        let body = arena.constant(0);
        Some(Constraint {
            reduction: Reduction {
                op: ReduceOp::Sum,
                iterable: Iterable::Scan { list: l, init: 1, boundary: 0, step, end: None },
                arena,
                body,
                coeff: 1,
            },
            op: Op::Le,
            rhs: 0,
        })
    };
    assert!(RoutingSpec::from_model(&scan_dispatch_model(divisor_scan)).is_none());
}

#[test]
fn compiler_rejects_zero_capable_divisor_scan() {
    // `emit = cur / rate[cur]` with `rate[1] == 0`: LS reads the zero divisor
    // as the value 0, but the kernel `Div` makes that stop infeasible, so the
    // two engines would disagree. The scan must fall back to LS, not lower.
    let zero_div = |l: usize| {
        let mut arena = ExprArena::default();
        let cur = arena.arg(0);
        let acc = arena.arg(1);
        let step = arena.add(acc, cur);
        let cur_e = arena.arg(0);
        let rate = arena.array(Arc::new(vec![9, 0, 2, 2, 2]), cur_e); // rate[item 1] == 0
        let body = arena.div(cur_e, rate);
        Some(Constraint {
            reduction: Reduction {
                op: ReduceOp::Sum,
                iterable: Iterable::Scan { list: l, init: 0, boundary: 0, step, end: None },
                arena,
                body,
                coeff: 1,
            },
            op: Op::Le,
            rhs: 100,
        })
    };
    assert!(RoutingSpec::from_model(&scan_dispatch_model(zero_div)).is_none());
}

#[test]
fn compiler_accepts_nonzero_const_divisor_scan() {
    // The div guard must not over-reject: dividing by a nonzero constant lowers
    // cleanly (div-by-const with the accumulator in the numerator is supported).
    let const_div = |l: usize| {
        let mut arena = ExprArena::default();
        let cur = arena.arg(0);
        let acc = arena.arg(1);
        let step = arena.add(acc, cur);
        let acc_e = arena.arg(1);
        let two = arena.constant(2);
        let body = arena.div(acc_e, two);
        Some(Constraint {
            reduction: Reduction {
                op: ReduceOp::Sum,
                iterable: Iterable::Scan { list: l, init: 0, boundary: 0, step, end: None },
                arena,
                body,
                coeff: 1,
            },
            op: Op::Le,
            rhs: 100,
        })
    };
    assert!(RoutingSpec::from_model(&scan_dispatch_model(const_div)).is_some());
}

#[test]
fn compiler_rejects_oversized_table_scan() {
    // A table value beyond the i32-safe band could overflow the kernel's plain
    // i64 `Expr::eval` on an intermediate (LS saturates), so the scan falls back.
    let big_table = |l: usize| {
        let mut arena = ExprArena::default();
        let acc = arena.arg(1);
        let cur_s = arena.arg(0);
        let huge = arena.array(Arc::new(vec![0, i64::MAX, 0, 0, 0]), cur_s);
        let step = arena.add(acc, huge);
        let body = arena.arg(1);
        Some(Constraint {
            reduction: Reduction {
                op: ReduceOp::Sum,
                iterable: Iterable::Scan { list: l, init: 0, boundary: 0, step, end: None },
                arena,
                body,
                coeff: 1,
            },
            op: Op::Le,
            rhs: 100,
        })
    };
    assert!(RoutingSpec::from_model(&scan_dispatch_model(big_table)).is_none());
}

#[test]
fn signature_rejects_zero_capable_and_oversized_leaves() {
    // Direct predicate checks: `scan_routing_signature` returns `None` for a
    // zero-capable divisor and for an out-of-band table leaf.
    let items = vec![1, 2, 3, 4];
    let zero_div = {
        let mut arena = ExprArena::default();
        let cur = arena.arg(0);
        let acc = arena.arg(1);
        let step = arena.add(acc, cur);
        let cur_e = arena.arg(0);
        let rate = arena.array(Arc::new(vec![9, 0, 2, 2, 2]), cur_e);
        let body = arena.div(cur_e, rate);
        Constraint {
            reduction: Reduction {
                op: ReduceOp::Sum,
                iterable: Iterable::Scan { list: 0, init: 0, boundary: 0, step, end: None },
                arena,
                body,
                coeff: 1,
            },
            op: Op::Le,
            rhs: 100,
        }
    };
    assert!(scan_routing_signature(&zero_div, &items).is_none(), "zero-capable divisor is rejected");
}

#[test]
fn compiler_rejects_select_kth_scan() {
    let kth_scan = |l: usize| {
        let mut arena = ExprArena::default();
        let cur = arena.arg(0);
        let acc = arena.arg(1);
        let step = arena.add(acc, cur);
        let body = arena.arg(1);
        Some(Constraint {
            reduction: Reduction {
                op: ReduceOp::SelectKth(0),
                iterable: Iterable::Scan { list: l, init: 0, boundary: 0, step, end: None },
                arena,
                body,
                coeff: 1,
            },
            op: Op::Le,
            rhs: 0,
        })
    };
    assert!(RoutingSpec::from_model(&scan_dispatch_model(kth_scan)).is_none());
}

#[test]
fn compiler_rejects_out_of_i32_range_scan() {
    // acc multiplies by 100000 each step, leaving the i32-safe band immediately.
    let huge_scan = |l: usize| {
        let mut arena = ExprArena::default();
        let acc = arena.arg(1);
        let big = arena.constant(100_000);
        let step = arena.mul(acc, big);
        let body = arena.constant(0);
        Some(Constraint {
            reduction: Reduction {
                op: ReduceOp::Sum,
                iterable: Iterable::Scan { list: l, init: 100_000, boundary: 0, step, end: None },
                arena,
                body,
                coeff: 1,
            },
            op: Op::Le,
            rhs: 0,
        })
    };
    assert!(RoutingSpec::from_model(&scan_dispatch_model(huge_scan)).is_none());
}

// --- optional-list pool oracles (G9) ----------------------------------------
//
// The pool depot forms a zero-cost route that absorbs dropped items. Every test
// asserts a three-way equality: engine integer optimum == exact list enumeration
// optimum == an independent hand brute-force where each item goes to one of the
// k real routes OR the pool (cost 0). This is the primary defence against a
// silently mis-gated pooled item.

/// Best closed-route value for one real route under a pool: the minimum
/// closed-tour cost over orderings passing `filter`, minus a constant reward per
/// served stop. `None` if no ordering passes the filter; empty route scores 0.
fn best_pool_route(depot: i32, route: &[i32], dist: &[Vec<i64>], reward: i64, filter: &dyn Fn(&[i32]) -> bool) -> Option<i64> {
    if route.is_empty() {
        return Some(0);
    }
    permutations(route)
        .iter()
        .filter(|p| filter(p))
        .map(|p| ordered_closed_cost(depot, p, dist))
        .min()
        .map(|cost| cost - reward * route.len() as i64)
}

/// Exhaustive optimum for a `k`-real-route optional model: each item goes to one
/// of `k` real routes OR the pool (contributing 0). Mirrors the in_pool gating.
fn min_pool(depot: i32, items: &[i32], dist: &[Vec<i64>], k: usize, reward: i64, filter: &dyn Fn(&[i32]) -> bool) -> i64 {
    let choices = k + 1;
    let combos = (choices as u64).pow(items.len() as u32);
    let mut best = i64::MAX;
    for combo in 0..combos {
        let mut routes = vec![Vec::new(); k];
        let mut code = combo;
        for &item in items {
            let choice = (code % choices as u64) as usize;
            code /= choices as u64;
            if choice < k {
                routes[choice].push(item);
            }
        }
        let total: Option<i64> = routes.iter().map(|r| best_pool_route(depot, r, dist, reward, filter)).sum();
        if let Some(total) = total {
            best = best.min(total);
        }
    }
    best
}

/// The exact list-enumeration optimum for a pool model (the G8 oracle). Asserts
/// the instance is enumerable (proven OPTIMAL), scoring only the objective-
/// referenced real lists so the free pool list contributes 0.
fn enum_pool_optimum(model: &CollectionModel) -> i64 {
    let tiers = crate::model::list_objective_tiers(&model.objectives, &model.items).expect("objective parses for enumeration");
    let stop = AtomicBool::new(false);
    let outcome =
        crate::engines::list_exact::solve(model, &tiers, &stop, |_| {}).expect("enumeration runs").expect("enumeration supports the model");
    assert_eq!(outcome.status, crate::engines::list_exact::Status::Optimal, "enumeration proves optimality on the small instance");
    outcome.objectives[0]
}

/// A `k`-real-route pool model whose objective is edges only (list `k` is the
/// free pool, referenced by nothing).
fn pool_edge_model(items: Vec<i32>, dist: &Arc<Vec<Vec<i64>>>, k: usize) -> CollectionModel {
    let terms = (0..k).map(|list| scan_edge(dist, list)).collect();
    CollectionModel {
        items,
        lists: k + 1,
        objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    }
}

/// A reified scan objective term paying a constant `reward` per served stop
/// (`emit = -reward`, order-independent). Its `Sum` over a real route is
/// `-reward * length`, so summing the served customers rewards serving them.
fn reward_scan(list: usize, reward: i64) -> Reduction {
    let mut arena = ExprArena::default();
    let step = arena.arg(1); // accumulator threaded unchanged (unused)
    let emit = arena.constant(-reward);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list, init: 0, boundary: 0, step, end: None }, arena, body: emit, coeff: 1 }
}

/// A `k`-real-route pool model: edges plus a reified reward scan per real route.
fn pool_reward_model(items: Vec<i32>, dist: &Arc<Vec<Vec<i64>>>, k: usize, reward: i64) -> CollectionModel {
    let mut terms: Vec<Reduction> = (0..k).map(|list| scan_edge(dist, list)).collect();
    terms.extend((0..k).map(|list| reward_scan(list, reward)));
    CollectionModel {
        items,
        lists: k + 1,
        objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    }
}

fn allow_all(_: &[i32]) -> bool {
    true
}

#[test]
fn pool_all_items_pooled_is_zero() {
    // No reward, every served edge strictly positive: dropping everything is
    // optimal. The whole pool route must be zero-cost (optimum 0).
    let dist = vec![vec![0, 7, 9, 8], vec![7, 0, 5, 6], vec![9, 5, 0, 4], vec![8, 6, 4, 0]];
    let items = vec![1, 2, 3];
    let brute = min_pool(0, &items, &dist, 1, 0, &allow_all);
    assert_eq!(brute, 0, "with no reward the best plan pools every item");
    let model = pool_edge_model(items.clone(), &Arc::new(dist), 1);
    let enumerated = enum_pool_optimum(&model);
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported pool lowering");
    assert!(outcome.complete, "exhausted the search");
    assert!(outcome.solution.feasible);
    assert_eq!(outcome.solution.objectives, vec![0], "engine optimum is zero");
    assert_eq!(outcome.solution.objectives[0], enumerated, "engine == enumeration");
    assert_eq!(outcome.solution.objectives[0], brute, "engine == hand brute-force");
    assert!(outcome.solution.lists[0].is_empty(), "the real route is empty");
    let mut pooled = outcome.solution.lists[1].clone();
    pooled.sort_unstable();
    assert_eq!(pooled, items, "every item lands in the pool list");
}

#[test]
fn pool_single_node_serve_or_drop() {
    // One customer, round trip 20. Reward 25 > 20 -> serve (net -5); reward 5 <
    // 20 -> drop (optimum 0). The gate must flip with the reward.
    let dist = vec![vec![0, 10], vec![10, 0]];
    let items = vec![1];

    let served_brute = min_pool(0, &items, &dist, 1, 25, &allow_all);
    assert_eq!(served_brute, -5);
    let served = pool_reward_model(items.clone(), &Arc::new(dist.clone()), 1, 25);
    let enumerated = enum_pool_optimum(&served);
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&served, 0, &stop, &mut |_| {}).expect("supported");
    assert!(outcome.complete);
    assert_eq!(outcome.solution.objectives[0], served_brute, "serve: engine == brute");
    assert_eq!(outcome.solution.objectives[0], enumerated, "serve: engine == enumeration");
    assert_eq!(outcome.solution.lists[0], vec![1], "the customer is served");

    let dropped_brute = min_pool(0, &items, &dist, 1, 5, &allow_all);
    assert_eq!(dropped_brute, 0);
    let dropped = pool_reward_model(items.clone(), &Arc::new(dist), 1, 5);
    let enumerated = enum_pool_optimum(&dropped);
    let outcome = solve_collection(&dropped, 0, &stop, &mut |_| {}).expect("supported");
    assert!(outcome.complete);
    assert_eq!(outcome.solution.objectives[0], dropped_brute, "drop: engine == brute");
    assert_eq!(outcome.solution.objectives[0], enumerated, "drop: engine == enumeration");
    assert!(outcome.solution.lists[0].is_empty(), "the customer is pooled");
    assert_eq!(outcome.solution.lists[1], vec![1]);
}

#[test]
fn pool_reward_scan_changes_optimum_and_matches_oracles() {
    // Customers on a line at positions 1, 2, 10 (depot 0). Reward 6/stop makes
    // the near cluster {1,2} profitable but the far one a loss, so the optimum
    // serves some and pools some -- and beats the edges-only optimum of 0,
    // proving the reified scan term actually bites (exercises the gating).
    let dist = line_dist(&[0, 1, 2, 10]);
    let items = vec![1, 2, 3];
    let reward = 6;
    let brute = min_pool(0, &items, &dist, 1, reward, &allow_all);
    assert!(brute < 0, "the reward makes serving profitable, changing the optimum from the edges-only 0");

    let model = pool_reward_model(items, &Arc::new(dist), 1, reward);
    let enumerated = enum_pool_optimum(&model);
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported pool + reward lowering");
    assert!(outcome.complete, "exhausted the search");
    assert_eq!(outcome.solution.objectives[0], brute, "engine == hand brute-force");
    assert_eq!(outcome.solution.objectives[0], enumerated, "engine == enumeration");
    assert!(!outcome.solution.lists[0].is_empty(), "some customers are served");
    assert!(!outcome.solution.lists[1].is_empty(), "some customers are pooled");
}

#[test]
fn pool_multi_route_matches_oracles() {
    // Two real routes + pool. Two near clusters go to the two routes; a far
    // customer is pooled. Reward 8/stop.
    let dist = line_dist(&[0, 1, 2, 20, 21, 50]);
    let items = vec![1, 2, 3, 4, 5];
    let reward = 8;
    let brute = min_pool(0, &items, &dist, 2, reward, &allow_all);
    let model = pool_reward_model(items.clone(), &Arc::new(dist), 2, reward);
    let enumerated = enum_pool_optimum(&model);
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported multi-route pool");
    assert!(outcome.complete, "exhausted the search");
    assert_eq!(outcome.solution.objectives[0], brute, "engine == hand brute-force");
    assert_eq!(outcome.solution.objectives[0], enumerated, "engine == enumeration");
    let mut served: Vec<i32> = outcome.solution.lists[..2].iter().flatten().copied().collect();
    served.sort_unstable();
    let mut pooled = outcome.solution.lists[2].clone();
    pooled.sort_unstable();
    let mut all: Vec<i32> = served.iter().chain(pooled.iter()).copied().collect();
    all.sort_unstable();
    assert_eq!(all, items, "every item is either served or pooled exactly once");
}

#[test]
fn pool_empty_real_route_matches_oracles() {
    // Two real routes but only one near cluster is profitable, so one real route
    // stays empty and the far customer is pooled.
    let dist = line_dist(&[0, 1, 2, 50]);
    let items = vec![1, 2, 3];
    let reward = 6;
    let brute = min_pool(0, &items, &dist, 2, reward, &allow_all);
    let model = pool_reward_model(items, &Arc::new(dist), 2, reward);
    let enumerated = enum_pool_optimum(&model);
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported pool with empty route");
    assert!(outcome.complete, "exhausted the search");
    assert_eq!(outcome.solution.objectives[0], brute, "engine == hand brute-force");
    assert_eq!(outcome.solution.objectives[0], enumerated, "engine == enumeration");
    let empty_real_routes = outcome.solution.lists[..2].iter().filter(|route| route.is_empty()).count();
    assert!(empty_real_routes >= 1, "at least one real route is empty");
}

#[test]
fn pool_scan_constraint_and_reward_objective_matches_oracles() {
    // The combined decisive case: optional + edges + a scan CONSTRAINT (hard
    // time windows) + a reified scan OBJECTIVE (reward). Customers 1,2 share a
    // deadline they cannot both meet in one route, so the window forbids serving
    // both; the reward still makes serving the reachable ones profitable, and the
    // far ones (3,4) are cheap to reach late. Three-way equality throughout.
    let dist = line_dist(&[0, 2, 4, 6, 8]);
    let service = vec![0i64, 1, 1, 1, 1];
    let earliest = vec![0i64, 0, 0, 0, 0];
    let latest = vec![0i64, 5, 5, 20, 20];
    let items = vec![1, 2, 3, 4];
    let depot = 0;
    let reward = 20;
    let windows = |route: &[i32]| lateness_sum(route, &dist, &earliest, &latest, &service, depot) <= 0;
    let brute = min_pool(depot, &items, &dist, 1, reward, &windows);
    assert!(brute < 0, "serving the reachable customers is profitable under the windows");

    let dist_arc = Arc::new(dist.clone());
    let mut terms = vec![scan_edge(&dist_arc, 0)];
    terms.push(reward_scan(0, reward));
    let model = CollectionModel {
        items: items.clone(),
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
        constraints: vec![Constraint { reduction: lateness_scan(0, &dist, &earliest, &latest, &service, depot), op: Op::Le, rhs: 0 }],
        globals: Vec::new(),
        schedule: None,
    };
    let enumerated = enum_pool_optimum(&model);
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported combined pool lowering");
    assert!(outcome.complete, "exhausted the search");
    assert_eq!(outcome.solution.objectives[0], brute, "engine == hand brute-force");
    assert_eq!(outcome.solution.objectives[0], enumerated, "engine == enumeration");
    assert!(
        !(outcome.solution.lists[0].contains(&1) && outcome.solution.lists[0].contains(&2)),
        "the window forbids serving both 1 and 2 in the single real route"
    );
    assert!(windows(&outcome.solution.lists[0]), "the served route meets its hard windows");
}

#[test]
fn pool_scale_proves_optimal_where_enumeration_times_out() {
    // 12 customers + a pool, every served edge strictly positive. Enumeration
    // cannot exhaust the ordered partitions within a short budget and returns
    // UNKNOWN; the integer path proves the optimum is 0 (pool everything) from
    // the non-negative edge bound. This is the scale win: an instance
    // enumeration cannot close but the integer path proves OPTIMAL.
    let count = 12usize;
    let positions: Vec<i64> = (0..=count as i64).collect();
    let dist = line_dist(&positions);
    let items: Vec<i32> = (1..=count as i32).collect();
    let model = pool_edge_model(items, &Arc::new(dist), 1);

    // Enumeration under a short wall-clock budget cannot finish the exact
    // search (10! ordered partitions). The interruption contract discards
    // an in-flight enumeration result, so it returns UNKNOWN rather than a
    // partially replayed incumbent.
    let tiers = crate::model::list_objective_tiers(&model.objectives, &model.items).expect("objective parses");
    let enum_stop = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&enum_stop);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(400));
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        });
    }
    let enum_start = std::time::Instant::now();
    let enum_out = crate::engines::list_exact::solve(&model, &tiers, &enum_stop, |_| {}).expect("runs").expect("supported");
    let enum_elapsed = enum_start.elapsed();
    assert_eq!(enum_out.status, crate::engines::list_exact::Status::Unknown, "interrupted enumeration cannot prove optimality");
    assert!(enum_out.solution.is_none(), "an interrupted exact enumeration does not publish a partial assignment");

    // Integer path: proves OPTIMAL. A safety timeout turns a hang into a clear
    // failure rather than blocking CI.
    let int_stop = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&int_stop);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(30));
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        });
    }
    let int_start = std::time::Instant::now();
    let outcome = solve_collection(&model, 0, &int_stop, &mut |_| {}).expect("supported pool lowering");
    let int_elapsed = int_start.elapsed();
    assert!(outcome.complete, "the integer path proves optimality");
    assert_eq!(outcome.solution.objectives[0], 0, "pooling every customer is optimal");
    eprintln!(
        "SCALE pool: enumeration UNKNOWN in {:?}, integer OPTIMAL ({}) in {:?}",
        enum_elapsed, outcome.solution.objectives[0], int_elapsed
    );
}

#[test]
fn pool_compiler_accepts_only_supported_shapes() {
    let dist4 = Arc::new(vec![vec![0, 7, 9, 8], vec![7, 0, 5, 6], vec![9, 5, 0, 4], vec![8, 6, 4, 0]]);
    let dist_line = Arc::new(line_dist(&[0, 2, 4, 6, 8]));
    let demand = Arc::new(vec![0i64, 1, 1, 1, 1]);

    // A pool + capacity model (must route to enumeration, not integer).
    let mut cap_pool = pool_reward_model(vec![1, 2, 3, 4], &dist_line, 1, 10);
    cap_pool.constraints.push(scan_load(&demand, 0, 2));

    // A pool + route-specific scan (non-homogeneous over the two real routes).
    let mut route_specific = pool_reward_model(vec![1, 2, 3, 4], &dist_line, 2, 10);
    route_specific.constraints.push(Constraint {
        reduction: lateness_scan(0, &line_dist(&[0, 2, 4, 6, 8]), &[0, 0, 0, 0, 0], &[0, 5, 5, 20, 20], &[0, 1, 1, 1, 1], 0),
        op: Op::Le,
        rhs: 0,
    });

    let battery: Vec<CollectionModel> = vec![
        pool_edge_model(vec![1, 2, 3], &dist4, 1),                                    // pool, single route
        pool_edge_model(vec![1], &dist4, 1),                                          // pool, single node
        pool_edge_model(vec![1, 2, 3], &dist4, 2),                                    // pool, multi route
        pool_reward_model(vec![1, 2, 3], &Arc::new(line_dist(&[0, 1, 2, 10])), 1, 6), // pool + reward objective
        pool_reward_model(vec![1, 2, 3, 4, 5], &Arc::new(line_dist(&[0, 1, 2, 20, 21, 50])), 2, 8), // pool + reward, multi
        cap_pool,                                                                     // pool + capacity -> enumeration
        route_specific,                                                               // pool + route-specific scan -> enumeration
        closed_tsp_model(vec![1, 2, 3], 0, &dist4),                                   // non-pool TSP
    ];

    let expected = [true, true, true, true, true, false, false, true];
    for (model, expected) in battery.iter().zip(expected) {
        assert_eq!(RoutingSpec::from_model(model).is_some(), expected);
    }
    // The two exclusions are handled by another already-compiled candidate.
    assert!(RoutingSpec::from_model(&battery[5]).is_none(), "pool + capacity is not integer-lowerable");
    assert!(RoutingSpec::from_model(&battery[6]).is_none(), "route-specific scan is not integer-lowerable");
}

/// Per-route cumulative-load scan `Σ prefix_load`, as a constraint reduction.
/// `emit = acc` (running load after each stop); `step = acc + item_value`.
fn prefix_load_scan(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let cur = arena.arg(0);
    let acc = arena.arg(1);
    let step = arena.add(acc, cur);
    let emit = arena.arg(1); // the new accumulator = load after this stop
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list, init: 0, boundary: 0, step, end: None }, arena, body: emit, coeff: 1 }
}

#[test]
fn pool_scan_constraint_ge_matches_oracle() {
    // A `Ge` scan constraint on a pool model (only `Le` was covered): each real
    // route's cumulative load must reach at least `rhs`, so an empty (all-pooled)
    // route is forbidden when rhs > 0. A reward objective incentivises serving,
    // and three-way equality must hold including this feasibility coupling.
    let dist = line_dist(&[0, 1, 2, 3]);
    let items = vec![1, 2, 3];
    let reward = 12;
    let rhs = 3;
    let dist_arc = Arc::new(dist.clone());
    let mut model = pool_reward_model(items.clone(), &dist_arc, 1, reward);
    model.constraints.push(Constraint { reduction: prefix_load_scan(0), op: Op::Ge, rhs });

    // Hand brute-force: the real route's Σ prefix-load must be >= rhs.
    let prefix_ok = |route: &[i32]| {
        let (mut acc, mut total) = (0i64, 0i64);
        for &c in route {
            acc += i64::from(c);
            total += acc;
        }
        route.is_empty() || total >= rhs
    };
    let brute = min_pool(0, &items, &dist, 1, reward, &prefix_ok);

    let enumerated = enum_pool_optimum(&model);
    let stop = AtomicBool::new(false);
    let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported Ge-scan pool lowering");
    assert!(outcome.complete, "exhausted the search");
    assert_eq!(outcome.solution.objectives[0], brute, "engine == hand brute-force");
    assert_eq!(outcome.solution.objectives[0], enumerated, "engine == enumeration");
    assert!(prefix_ok(&outcome.solution.lists[0]), "the served route meets the Ge load floor");
}

#[test]
fn pool_random_instances_match_enumeration() {
    // Broad differential defense for the pool gating: on random small pool
    // instances (varied topology, item count, route count, reward), the integer
    // optimum must equal the enumeration optimum. Catches gating bugs in
    // combinations the hand-written cases do not enumerate. Fixed seed, trimmed
    // to stay CI-cheap while covering the pool cost channels.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    for _ in 0..50 {
        let n = 2 + (next() % 3) as usize; // 2..=4 customers
        let k = 1 + (next() % 2) as usize; // 1..=2 real routes
        let m = n + 1; // node universe: depot + customers
        let mut dist = vec![vec![0i64; m]; m];
        #[allow(clippy::needless_range_loop)] // fills dist[a][b] over the full grid
        for a in 0..m {
            for b in 0..m {
                if a != b {
                    dist[a][b] = 1 + (next() % 9) as i64; // strictly positive
                }
            }
        }
        let items: Vec<i32> = (1..=n as i32).collect();
        let reward = (next() % 12) as i64; // 0..=11; 0 => pool everything
        let model = pool_reward_model(items, &Arc::new(dist), k, reward);

        let enumerated = enum_pool_optimum(&model);
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported random pool lowering");
        assert!(outcome.complete, "integer path proves optimality on the small instance");
        assert_eq!(outcome.solution.objectives[0], enumerated, "integer optimum == enumeration optimum (n={n}, k={k}, reward={reward})");
    }
}

/// A reified scan objective paying a per-item reward: `emit = table[cur]`, so a
/// route's `Sum` is the total reward of its served customers (each pooled one is
/// gated to 0). Distinct from the order-independent constant `reward_scan`.
fn item_reward_scan(list: usize, reward_table: &Arc<Vec<i64>>) -> Reduction {
    let mut arena = ExprArena::default();
    let step = arena.arg(1); // accumulator threaded unchanged (unused)
    let cur = arena.arg(0);
    let emit = arena.array(Arc::clone(reward_table), cur);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list, init: 0, boundary: 0, step, end: None }, arena, body: emit, coeff: 1 }
}

/// The same per-item reward as `item_reward_scan` (`emit = table[cur]`), but
/// expressed as a `Sum` over `Items` (`cp.sum(x, |i| table[i])`) -- the CIBLE 1
/// objective shape. Must normalise to the identical identity-accumulator scan.
fn item_reward_items(list: usize, reward_table: &Arc<Vec<i64>>) -> Reduction {
    let mut arena = ExprArena::default();
    let cur = arena.arg(0);
    let emit = arena.array(Arc::clone(reward_table), cur);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body: emit, coeff: 1 }
}

/// The G10 repro as a model: one real route + pool, required customers `1..=n`
/// (each paying a huge serve reward `-reward`) and disposable optionals
/// `n+1..=n+m` (zero reward, pure positive detour on the line). Objective =
/// travel - reward per served required.
fn disposable_model(n: usize, m: usize, reward: i64) -> CollectionModel {
    let total = n + m;
    let positions: Vec<i64> = (0..=total as i64).collect();
    let dist = Arc::new(line_dist(&positions));
    let mut reward_table = vec![0i64; total + 1];
    for slot in reward_table.iter_mut().take(n + 1).skip(1) {
        *slot = -reward;
    }
    let reward_table = Arc::new(reward_table);
    let items: Vec<i32> = (1..=total as i32).collect();
    let terms = vec![scan_edge(&dist, 0), item_reward_scan(0, &reward_table)];
    CollectionModel {
        items,
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    }
}

/// `disposable_model` with the per-served reward expressed as an `Items` sum
/// (`cp.sum(x, |i| table[i])`) rather than a scan -- the CIBLE 1 objective shape.
fn disposable_items_model(n: usize, m: usize, reward: i64) -> CollectionModel {
    let total = n + m;
    let positions: Vec<i64> = (0..=total as i64).collect();
    let dist = Arc::new(line_dist(&positions));
    let mut reward_table = vec![0i64; total + 1];
    for slot in reward_table.iter_mut().take(n + 1).skip(1) {
        *slot = -reward;
    }
    let reward_table = Arc::new(reward_table);
    let items: Vec<i32> = (1..=total as i32).collect();
    let terms = vec![scan_edge(&dist, 0), item_reward_items(0, &reward_table)];
    CollectionModel {
        items,
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    }
}

#[test]
fn cible1_items_reward_disposables_prove_optimal() {
    // CIBLE 1: disposable optionals under a per-item `Items` reward
    // (`cp.sum(x, |i| REQ[i]) * -reward`) must prove OPTIMAL, not just
    // SATISFIABLE -- the normalisation feeds the same dominance bound the scan
    // reward gets. The M=n optimum equals the M=0 optimum (disposables dropped).
    let reward = 100_000i64;
    for n in [6usize, 8] {
        let stop = AtomicBool::new(false);
        let v0 = solve_collection(&disposable_items_model(n, 0, reward), 0, &stop, &mut |_| {}).expect("M=0 lowering");
        assert!(v0.complete, "M=0 proves OPTIMAL (n={n})");
        let vn = solve_collection(&disposable_items_model(n, n, reward), 0, &stop, &mut |_| {}).expect("M=n lowering");
        assert!(vn.complete, "M=n proves OPTIMAL via the items-reward dominance bound (n={n})");
        assert_eq!(vn.solution.objectives[0], v0.solution.objectives[0], "disposables dropped: M=n optimum == M=0 optimum (n={n})");
    }
}

#[test]
fn pool_items_reward_matches_scan_and_enumeration() {
    // Soundness of the `Items`-reward normalisation: on random small pool
    // instances the per-item `Items` reward must lower to the SAME optimum as
    // the equivalent per-item scan reward AND as enumeration, and prove OPTIMAL.
    // Signed rewards (some serve-profitable, some not) exercise the dominance
    // gate both ways. Fixed seed, trimmed for CI.
    let mut state = 0xD1B5_4A32_D192_ED03u64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    for _ in 0..50 {
        let n = 2 + (next() % 3) as usize; // 2..=4 customers
        let k = 1 + (next() % 2) as usize; // 1..=2 real routes
        let m = n + 1;
        let mut dist = vec![vec![0i64; m]; m];
        #[allow(clippy::needless_range_loop)] // fills dist[a][b] over the full grid
        for a in 0..m {
            for b in 0..m {
                if a != b {
                    dist[a][b] = 1 + (next() % 9) as i64;
                }
            }
        }
        let dist = Arc::new(dist);
        let items: Vec<i32> = (1..=n as i32).collect();
        let mut reward_table = vec![0i64; m];
        for slot in reward_table.iter_mut().skip(1) {
            *slot = (next() % 13) as i64 - 6; // -6..=6
        }
        let reward_table = Arc::new(reward_table);
        let mk = |scan: bool| {
            let mut terms: Vec<Reduction> = (0..k).map(|l| scan_edge(&dist, l)).collect();
            terms.extend((0..k).map(|l| if scan { item_reward_scan(l, &reward_table) } else { item_reward_items(l, &reward_table) }));
            CollectionModel {
                items: items.clone(),
                lists: k + 1,
                objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
                constraints: Vec::new(),
                globals: Vec::new(),
                schedule: None,
            }
        };
        let items_model = mk(false);
        let enumerated = enum_pool_optimum(&items_model);
        let stop = AtomicBool::new(false);
        let items_out = solve_collection(&items_model, 0, &stop, &mut |_| {}).expect("items-reward lowering");
        let scan_out = solve_collection(&mk(true), 0, &stop, &mut |_| {}).expect("scan-reward lowering");
        assert!(items_out.complete, "items-reward integer path proves optimality (n={n}, k={k})");
        assert_eq!(items_out.solution.objectives[0], enumerated, "items optimum == enumeration (n={n}, k={k})");
        assert_eq!(items_out.solution.objectives[0], scan_out.solution.objectives[0], "items reward == scan reward optimum (n={n}, k={k})");
    }
}

#[test]
fn pool_disposable_optionals_prove_optimal_and_primal_never_worse() {
    // G10 acceptance: DISPOSABLE optionals (zero reward, pure detour) must not
    // collapse the optimality proof. (1) bench(8,8) proves OPTIMAL in seconds;
    // (2) for every (n, m) the incumbent is <= the same model's M=0 optimum
    // (serve required only) -- the primal is never worse than dropping all
    // optionals. Disposables must land in the pool, requireds in the real route.
    let reward = 100_000i64;
    let solve = |model: &CollectionModel| -> (i64, bool, std::time::Duration) {
        // Give each M=0 reference solve its own safety budget. A shared
        // deadline made later sizes fail solely because earlier cases had
        // consumed the cumulative wall clock.
        let stop = Arc::new(AtomicBool::new(false));
        {
            let flag = Arc::clone(&stop);
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(120));
                flag.store(true, std::sync::atomic::Ordering::Relaxed);
            });
        }
        let start = std::time::Instant::now();
        let out = solve_collection(model, 0, &stop, &mut |_| {}).expect("supported disposable-optional lowering");
        (if out.complete { out.solution.objectives[0] } else { i64::MAX }, out.complete, start.elapsed())
    };
    for n in 5..=8usize {
        let (v_star, m0_complete, _) = solve(&disposable_model(n, 0, reward));
        assert!(m0_complete, "M=0 optimum proves OPTIMAL for n={n}");
        for m in 0..=n {
            let model = disposable_model(n, m, reward);
            let stop_local = AtomicBool::new(false);
            let start = std::time::Instant::now();
            let out = solve_collection(&model, 0, &stop_local, &mut |_| {}).expect("supported");
            let elapsed = start.elapsed();
            eprintln!(
                "G10 n={n} m={m} status={} obj={} time={:?}",
                if out.complete { "OPTIMAL" } else { "SATISFIABLE" },
                out.solution.objectives[0],
                elapsed
            );
            assert!(out.complete, "OPTIMAL proven for n={n} m={m} in {elapsed:?}");
            assert!(
                out.solution.objectives[0] <= v_star,
                "primal never worse than M=0 optimum: n={n} m={m} got {} vs {v_star}",
                out.solution.objectives[0]
            );
            for &c in &out.solution.lists[0] {
                assert!((c as usize) <= n, "only required customers are served: n={n} m={m} served {c}");
            }
            let pooled = &out.solution.lists[1];
            for d in (n + 1)..=(n + m) {
                assert!(pooled.contains(&(d as i32)), "disposable {d} lands in the pool: n={n} m={m}");
            }
        }
    }
}
