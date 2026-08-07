use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use qayd::engines::dual::{attach, compute};
use qayd::model::list::{
    CollectionModel, CollectionSolution, Constraint, ExprArena, IntervalVar, Iterable, ObjectiveTier, Op, ReduceOp, Reduction, Schedule,
};

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
