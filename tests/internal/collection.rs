//! The general collection engine, exercised on several model shapes to show it
//! is not routing-specific: a TSP, a CVRP, a cardinality-balanced partition, a
//! body-arithmetic objective, and the empty `min` infeasibility rule. Each is
//! just a different set of lambda reductions over list variables.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use qayd::engines::ls::lists::solve_collection;
use qayd::model::list::{
    verify_collection_solution, CollectionModel, CollectionSolution, Constraint, ExprArena, GlobalConstraint, IntervalVar, Iterable, Mode,
    ObjectiveTier, Op, ReduceOp, Reduction, Resource, Schedule,
};
use qayd::model::{Model, ModelPackage};
use qayd::orchestrator::{solve_model_silent, ScheduleJsspSearch, SolveLimits, SolveMode, SolveRequest, SolveStatus};

#[test]
fn tsab_candidate_request_requires_exactly_seven_threads() {
    let request = SolveRequest { threads: 6, schedule_jssp_search: ScheduleJsspSearch::TsabCandidate, ..SolveRequest::default() };

    let error = request.validate().expect_err("TSAB candidate must not silently fall back to Legacy");

    assert_eq!(error.to_string(), "invalid solve request: schedule_jssp_search='tsab-candidate' currently requires threads=7");
}

/// A schedule-only collection model (no list variables).
fn schedule_model(sched: Schedule) -> CollectionModel {
    CollectionModel { items: vec![], lists: 0, objectives: vec![], constraints: vec![], globals: vec![], schedule: Some(sched) }
}

/// A single minimised objective tier from a set of reductions.
fn min_tier(terms: Vec<Reduction>) -> Vec<ObjectiveTier> {
    if terms.is_empty() {
        return Vec::new();
    }
    vec![ObjectiveTier { minimize: true, terms, max_terms: None }]
}

fn run(model: &CollectionModel, millis: u64) -> CollectionSolution {
    assert!(model.validate().is_ok(), "model should validate");
    if model.schedule.is_some() {
        let request = SolveRequest {
            mode: SolveMode::Exact,
            limits: SolveLimits { time: Some(Duration::from_millis(millis)), ..SolveLimits::default() },
            ..SolveRequest::default()
        };
        let result = solve_model_silent(&ModelPackage::new(Model::from_collection(model)), &request)
            .expect("the canonical schedule orchestrator should solve the legacy test model");
        let feasible = matches!(result.status(), SolveStatus::Optimal | SolveStatus::Satisfiable);
        let Some(primal) = result.primal() else {
            return CollectionSolution {
                lists: Vec::new(),
                objectives: Vec::new(),
                feasible: false,
                starts: Vec::new(),
                presences: Vec::new(),
                machines: Vec::new(),
                modes: Vec::new(),
                bound: None,
            };
        };
        let intervals = &primal.assignment().intervals;
        return CollectionSolution {
            lists: Vec::new(),
            objectives: primal.objectives().to_vec(),
            feasible,
            starts: intervals.iter().map(|interval| interval.start.unwrap_or_default()).collect(),
            presences: intervals.iter().map(|interval| interval.present).collect(),
            machines: intervals.iter().map(|interval| interval.machine.and_then(|value| i64::try_from(value).ok()).unwrap_or(-1)).collect(),
            modes: intervals.iter().map(|interval| interval.mode).collect(),
            bound: None,
        };
    }
    let stop = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&stop);
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(millis));
        flag.store(true, Ordering::SeqCst);
    });
    solve_collection(model, 0, &stop, &mut |_| {})
}

fn machine_no_overlap_model(spec: &[(usize, i64, bool)], horizon: i64) -> CollectionModel {
    let intervals = spec
        .iter()
        .enumerate()
        .map(|(index, &(machine, duration, optional))| IntervalVar {
            duration: 0,
            horizon,
            modes: vec![Mode { reference: Some(index), machine, duration, start_window: (0, horizon - duration) }],
            optional,
        })
        .collect();
    schedule_model(Schedule { intervals, precedences: Vec::new(), resources: vec![Resource::MachineNoOverlap], minimize_makespan: true })
}

fn machine_solution(starts: &[i64], presences: &[bool], machines: &[i64], modes: &[Option<usize>], objective: i64) -> CollectionSolution {
    CollectionSolution {
        lists: Vec::new(),
        objectives: vec![objective],
        feasible: true,
        starts: starts.to_vec(),
        presences: presences.to_vec(),
        machines: machines.to_vec(),
        modes: modes.to_vec(),
        bound: None,
    }
}

fn quadratic_machine_no_overlap_ok(starts: &[i64], durations: &[i64], horizon: i64) -> bool {
    for (&start, &duration) in starts.iter().zip(durations) {
        if start < 0 || start.checked_add(duration).is_none_or(|end| end > horizon) {
            return false;
        }
    }
    for left in 0..starts.len() {
        let left_end = starts[left].saturating_add(durations[left]);
        for right in (left + 1)..starts.len() {
            let right_end = starts[right].saturating_add(durations[right]);
            if starts[left] < right_end && starts[right] < left_end {
                return false;
            }
        }
    }
    true
}

/// `sum_edges(list, (i, j) => dist[i][j])` with the depot (node 0) at both ends.
fn route_cost(list: usize, dist: &[Vec<i64>]) -> Reduction {
    let mut arena = ExprArena::default();
    let i = arena.arg(0);
    let j = arena.arg(1);
    let body = arena.matrix(Arc::new(dist.to_vec()), i, j);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
}

/// `sum(list, i => demand[i])`.
fn load(list: usize, demand: &[i64]) -> Reduction {
    let mut arena = ExprArena::default();
    let i = arena.arg(0);
    let body = arena.array(Arc::new(demand.to_vec()), i);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

/// `count(list, _ => 1)`, i.e. the list length.
fn count_items(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let body = arena.constant(1);
    Reduction { op: ReduceOp::Count, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

/// `min(list, i => i)`: the smallest item value, undefined for an empty list.
fn min_item(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let body = arena.arg(0);
    Reduction { op: ReduceOp::Min, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

/// `max(list, i => i)`: the largest item value, undefined for an empty list.
fn max_item(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let body = arena.arg(0);
    Reduction { op: ReduceOp::Max, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

/// `op(list, i => values[i])` over the items of a list.
fn value_reduction(list: usize, op: ReduceOp, values: &[i64]) -> Reduction {
    let mut arena = ExprArena::default();
    let i = arena.arg(0);
    let body = arena.array(Arc::new(values.to_vec()), i);
    Reduction { op, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

/// `sum_edges(list, (i, j) => dist[i][j])` over the path `[start, items.., end]`.
fn edges_cost(list: usize, dist: &[Vec<i64>], start: i32, end: i32) -> Reduction {
    let mut arena = ExprArena::default();
    let i = arena.arg(0);
    let j = arena.arg(1);
    let body = arena.matrix(Arc::new(dist.to_vec()), i, j);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start, end }, arena, body, coeff: 1 }
}

/// Oracle: cost of the path `[start, route.., end]`.
fn path_cost(route: &[i32], dist: &[Vec<i64>], start: i32, end: i32) -> i64 {
    let mut nodes = vec![start];
    nodes.extend_from_slice(route);
    nodes.push(end);
    nodes.windows(2).map(|w| dist[w[0] as usize][w[1] as usize]).sum()
}

/// Time-window lateness of a route: a scan computing end-of-service times,
/// summing `max(0, end - latest)`. `step` is `max(earliest, acc + travel) +
/// service`; the body emits the lateness at each stop. Constrain to 0 for hard
/// time windows.
fn lateness_scan(list: usize, dist: &[Vec<i64>], earliest: &[i64], latest: &[i64], service: &[i64], depot: i32) -> Reduction {
    let mut arena = ExprArena::default();
    // step(cur=Arg0, acc=Arg1, prev=Arg2) -> max(earliest[cur], acc + dist[prev][cur]) + service[cur]
    let cur = arena.arg(0);
    let acc = arena.arg(1);
    let prev = arena.arg(2);
    let travel = arena.matrix(Arc::new(dist.to_vec()), prev, cur);
    let arrive = arena.add(acc, travel);
    let earl = arena.array(Arc::new(earliest.to_vec()), cur);
    let start = arena.max(earl, arrive);
    let svc = arena.array(Arc::new(service.to_vec()), cur);
    let step = arena.add(start, svc);
    // emit(cur=Arg0, end=Arg1, prev=Arg2) -> max(0, end - latest[cur])
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

/// Oracle: is `route` time-window feasible (replicates `lateness_scan`)?
fn tw_feasible(route: &[i32], dist: &[Vec<i64>], earliest: &[i64], latest: &[i64], service: &[i64], depot: i32) -> bool {
    let mut acc = 0i64;
    let mut prev = depot;
    for &c in route {
        let end = earliest[c as usize].max(acc + dist[prev as usize][c as usize]) + service[c as usize];
        if end > latest[c as usize] {
            return false;
        }
        acc = end;
        prev = c;
    }
    true
}

fn each_obj(k: usize, f: impl Fn(usize) -> Reduction) -> Vec<ObjectiveTier> {
    min_tier((0..k).map(f).collect())
}

fn each_con(k: usize, f: impl Fn(usize) -> Reduction, op: Op, rhs: i64) -> Vec<Constraint> {
    (0..k).map(|l| Constraint { reduction: f(l), op, rhs }).collect()
}

/// `1` if list `l` is non-empty, else `0` (the body is ignored).
fn used_list(l: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let body = arena.constant(0);
    Reduction { op: ReduceOp::Used, iterable: Iterable::Items(l), arena, body, coeff: 1 }
}

/// QAP cost of a permutation list: `sum[i][j] a[i][j] * b[p[i]][p[j]]`, where
/// `a` is indexed by positions (`Arg2`,`Arg3`) and `b` by the facilities at
/// those positions (`Arg0`,`Arg1`).
fn qap_cost(list: usize, a: &[Vec<i64>], b: &[Vec<i64>]) -> Reduction {
    let mut arena = ExprArena::default();
    let item_i = arena.arg(0);
    let item_j = arena.arg(1);
    let pos_i = arena.arg(2);
    let pos_j = arena.arg(3);
    let aij = arena.matrix(Arc::new(a.to_vec()), pos_i, pos_j);
    let bij = arena.matrix(Arc::new(b.to_vec()), item_i, item_j);
    let body = arena.mul(aij, bij);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Pairs(list), arena, body, coeff: 1 }
}

/// Number of conflicting ordered item pairs within a list: `sum conflict[a][b]`
/// over pairs `(a, b)` of items. Constrained to 0 to forbid conflicts in a bin.
fn conflict_pairs(list: usize, conflict: &[Vec<i64>]) -> Reduction {
    let mut arena = ExprArena::default();
    let a = arena.arg(0);
    let b = arena.arg(1);
    let body = arena.matrix(Arc::new(conflict.to_vec()), a, b);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Pairs(list), arena, body, coeff: 1 }
}

/// Count, within a list, of ordered position pairs `(i, j)` with `i < j` where
/// the item at `i` is the delivery of the item at `j` (`is_delivery_of[a][b]`),
/// i.e. a delivery placed before its pickup. Constrained to 0 for
/// pickup-before-delivery. Combines `Pairs` (P6), `Lt` and `Mul` (P2).
fn precedes_in_list(list: usize, is_delivery_of: &[Vec<i64>]) -> Reduction {
    let mut arena = ExprArena::default();
    let item_i = arena.arg(0);
    let item_j = arena.arg(1);
    let pos_i = arena.arg(2);
    let pos_j = arena.arg(3);
    let earlier = arena.lt(pos_i, pos_j); // 1 if i < j
    let is_del = arena.matrix(Arc::new(is_delivery_of.to_vec()), item_i, item_j);
    let body = arena.mul(earlier, is_del);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Pairs(list), arena, body, coeff: 1 }
}

fn line_matrix(coords: &[i64]) -> Vec<Vec<i64>> {
    let n = coords.len();
    (0..n).map(|i| (0..n).map(|j| (coords[i] - coords[j]).abs()).collect()).collect()
}

fn perms(items: &[i32]) -> Vec<Vec<i32>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(i);
        for mut p in perms(&rest) {
            p.insert(0, head);
            out.push(p);
        }
    }
    out
}

fn tour_cost(route: &[i32], dist: &[Vec<i64>]) -> i64 {
    if route.is_empty() {
        return 0;
    }
    let mut total = dist[0][route[0] as usize];
    for w in route.windows(2) {
        total += dist[w[0] as usize][w[1] as usize];
    }
    total + dist[*route.last().unwrap() as usize][0]
}

#[test]
fn tsp_one_list_matches_brute_force() {
    let dist = line_matrix(&[0, 4, 1, 3, 2]); // depot + 4 cities on a line
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 1,
        objectives: each_obj(1, |l| route_cost(l, &dist)),
        constraints: vec![],
        globals: vec![],
        schedule: None,
    };
    let opt = perms(&[1, 2, 3, 4]).iter().map(|p| tour_cost(p, &dist)).min().unwrap();
    let sol = run(&model, 300);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], opt, "TSP reaches the optimal tour");
}

#[test]
fn cvrp_distance_plus_capacity() {
    // Two clusters; capacity forces one cluster per vehicle.
    let dist = line_matrix(&[0, 10, 11, -10, -11]);
    let demand = [0, 1, 1, 1, 1];
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 2,
        objectives: each_obj(2, |l| route_cost(l, &dist)),
        constraints: each_con(2, |l| load(l, &demand), Op::Le, 2),
        globals: vec![],
        schedule: None,
    };
    // Brute force: assign each customer to a route, order optimally, respect cap.
    let mut best = i64::MAX;
    for code in 0..(2u32.pow(4)) {
        let mut routes = [Vec::new(), Vec::new()];
        for c in 0..4i32 {
            routes[((code >> c) & 1) as usize].push(c + 1);
        }
        if routes.iter().any(|r| r.iter().map(|&c| demand[c as usize]).sum::<i64>() > 2) {
            continue;
        }
        let cost: i64 = routes.iter().map(|r| perms(r).iter().map(|p| tour_cost(p, &dist)).min().unwrap_or(0)).sum();
        best = best.min(cost);
    }
    let sol = run(&model, 400);
    assert!(sol.feasible, "capacity respected");
    assert_eq!(sol.objectives[0], best, "CVRP reaches the optimum");
    let mut served: Vec<i32> = sol.lists.iter().flatten().copied().collect();
    served.sort_unstable();
    assert_eq!(served, vec![1, 2, 3, 4]);
}

#[test]
fn cardinality_balanced_partition() {
    // Four items into two lists, each forced to hold exactly two (Count == 2).
    let model = CollectionModel {
        items: vec![10, 20, 30, 40],
        lists: 2,
        objectives: min_tier(vec![]),
        constraints: each_con(2, count_items, Op::Eq, 2),
        globals: vec![],
        schedule: None,
    };
    let sol = run(&model, 200);
    assert!(sol.feasible, "a balanced partition exists");
    assert!(sol.lists.iter().all(|l| l.len() == 2), "each list holds exactly two items");
    let mut all: Vec<i32> = sol.lists.iter().flatten().copied().collect();
    all.sort_unstable();
    assert_eq!(all, vec![10, 20, 30, 40]);
}

#[test]
fn body_arithmetic_is_partition_invariant() {
    // Objective sum over items of (2 * weight[item] + 1). Each item contributes
    // regardless of which list it lands in, so the optimum is a known constant
    // and exercises Array / Mul / Add / Const in the body.
    let weight = [0, 3, 5, 2, 7];
    let items = vec![1, 2, 3, 4];
    let mut arena = ExprArena::default();
    let i = arena.arg(0);
    let w = arena.array(Arc::new(weight.to_vec()), i);
    let two = arena.constant(2);
    let one = arena.constant(1);
    let two_w = arena.mul(two, w);
    let body = arena.add(two_w, one);
    let reduction = Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(0), arena, body, coeff: 1 };

    let model = CollectionModel {
        items: items.clone(),
        lists: 2,
        objectives: min_tier(vec![reduction, {
            // second list contributes the same per-item formula over its own items
            let mut a = ExprArena::default();
            let i = a.arg(0);
            let w = a.array(Arc::new(weight.to_vec()), i);
            let two = a.constant(2);
            let one = a.constant(1);
            let tw = a.mul(two, w);
            let body = a.add(tw, one);
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(1), arena: a, body, coeff: 1 }
        }]),
        constraints: vec![],
        globals: vec![],
        schedule: None,
    };

    let expected: i64 = items.iter().map(|&c| 2 * weight[c as usize] + 1).sum();
    let sol = run(&model, 150);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], expected, "every item contributes 2*w+1 wherever it goes");
}

#[test]
fn empty_min_is_infeasible() {
    // Three lists, two items: by pigeonhole some list is empty, so its min over
    // items is undefined and no assignment is feasible.
    let model = CollectionModel {
        items: vec![5, 6],
        lists: 3,
        objectives: min_tier(vec![]),
        constraints: each_con(3, min_item, Op::Ge, 0),
        globals: vec![],
        schedule: None,
    };
    let sol = run(&model, 150);
    assert!(!sol.feasible, "an empty list has no min, so the model is infeasible");
}

#[test]
fn empty_max_is_infeasible() {
    // Same pigeonhole, exercising the Max-over-empty-Items undefined rule.
    let model = CollectionModel {
        items: vec![5, 6],
        lists: 3,
        objectives: min_tier(vec![]),
        constraints: each_con(3, max_item, Op::Le, 100),
        globals: vec![],
        schedule: None,
    };
    assert!(!run(&model, 150).feasible, "an empty list has no max, so the model is infeasible");
}

#[test]
fn max_objective_matches_brute_force() {
    // Minimise the sum over two non-empty lists of each list's largest value.
    let value = [0, 5, 1, 4, 2]; // by item id 1..4
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 2,
        objectives: each_obj(2, |l| value_reduction(l, ReduceOp::Max, &value)),
        constraints: vec![],
        globals: vec![],
        schedule: None,
    };
    let mut best = i64::MAX;
    for code in 0..16u32 {
        let mut lists = [Vec::new(), Vec::new()];
        for c in 0..4i32 {
            lists[((code >> c) & 1) as usize].push(c + 1);
        }
        if lists[0].is_empty() || lists[1].is_empty() {
            continue; // an empty list has no max => infeasible
        }
        let cost: i64 = lists.iter().map(|l| l.iter().map(|&c| value[c as usize]).max().unwrap()).sum();
        best = best.min(cost);
    }
    let sol = run(&model, 300);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], best, "Max objective reaches the optimum");
}

#[test]
fn min_max_constraints_feasible_and_unsat() {
    // SAT: values are all in [1, 4], so min >= 1 and max <= 4 hold on any
    // partition whose lists are non-empty.
    let mut sat_constraints = each_con(2, min_item, Op::Ge, 1);
    sat_constraints.extend(each_con(2, max_item, Op::Le, 4));
    let sat = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 2,
        objectives: vec![],
        constraints: sat_constraints,
        globals: vec![],
        schedule: None,
    };
    let s = run(&sat, 250);
    assert!(s.feasible, "min/max bounds are satisfiable");
    assert!(s.lists.iter().all(|l| !l.is_empty()), "both lists used");

    // UNSAT: max <= 0 is impossible since every item value is >= 1.
    let unsat = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 2,
        objectives: vec![],
        constraints: each_con(2, max_item, Op::Le, 0),
        globals: vec![],
        schedule: None,
    };
    assert!(!run(&unsat, 150).feasible, "max <= 0 is impossible");
}

#[test]
fn min_with_negative_values() {
    // One list holds every item, so the min is the global minimum value.
    let value = [0, -5, 3, -2, 1]; // id 1..4
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 1,
        objectives: min_tier(vec![value_reduction(0, ReduceOp::Min, &value)]),
        constraints: vec![],
        globals: vec![],
        schedule: None,
    };
    let sol = run(&model, 100);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], -5, "min over negative values");
}

#[test]
fn count_nonzero_with_negatives() {
    // Count is "body != 0", so a negative value counts and a zero does not.
    let value = [0, -5, 3, 0, 1]; // id 3 has value 0
    let sat = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 1,
        objectives: min_tier(vec![]),
        constraints: vec![Constraint { reduction: value_reduction(0, ReduceOp::Count, &value), op: Op::Eq, rhs: 3 }],
        globals: vec![],
        schedule: None,
    };
    assert!(run(&sat, 100).feasible, "three items have a non-zero value");
    let unsat = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 1,
        objectives: min_tier(vec![]),
        constraints: vec![Constraint { reduction: value_reduction(0, ReduceOp::Count, &value), op: Op::Eq, rhs: 4 }],
        globals: vec![],
        schedule: None,
    };
    assert!(!run(&unsat, 100).feasible, "only three values are non-zero");
}

#[test]
fn edges_with_nonzero_depot_and_open_path() {
    // Six nodes (0..5); items 1..4 are ordered, node 5 is the depot.
    let dist = line_matrix(&[0, 4, 1, 3, 2, 9]);

    // Closed tour pinned at depot 5 (start == end == 5).
    let closed = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 1,
        objectives: min_tier(vec![edges_cost(0, &dist, 5, 5)]),
        constraints: vec![],
        globals: vec![],
        schedule: None,
    };
    let opt_closed = perms(&[1, 2, 3, 4]).iter().map(|p| path_cost(p, &dist, 5, 5)).min().unwrap();
    let s = run(&closed, 300);
    assert!(s.feasible);
    assert_eq!(s.objectives[0], opt_closed, "closed tour with depot 5");

    // Open path from node 5 to node 0 (start != end).
    let open = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 1,
        objectives: min_tier(vec![edges_cost(0, &dist, 5, 0)]),
        constraints: vec![],
        globals: vec![],
        schedule: None,
    };
    let opt_open = perms(&[1, 2, 3, 4]).iter().map(|p| path_cost(p, &dist, 5, 0)).min().unwrap();
    let s = run(&open, 300);
    assert!(s.feasible);
    assert_eq!(s.objectives[0], opt_open, "open path 5 -> .. -> 0");
}

#[test]
fn bpp_minimizes_used_bins() {
    // Four items of weight 3, bins of capacity 6: two items per bin, so the
    // optimum is 2 bins. The objective is the number of non-empty bins (Used).
    let weight = [0, 3, 3, 3, 3]; // by item id 1..4
    let capacity = 6;
    let k = 4; // upper bound on bins
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: k,
        objectives: vec![ObjectiveTier { minimize: true, terms: (0..k).map(used_list).collect(), max_terms: None }],
        constraints: each_con(k, |l| value_reduction(l, ReduceOp::Sum, &weight), Op::Le, capacity),
        globals: vec![],
        schedule: None,
    };
    let sol = run(&model, 300);
    assert!(sol.feasible, "capacity respected");
    assert_eq!(sol.objectives[0], 2, "two bins suffice and are optimal");
    let nonempty = sol.lists.iter().filter(|l| !l.is_empty()).count();
    assert_eq!(nonempty, 2, "exactly two bins used");
    for bin in &sol.lists {
        assert!(bin.iter().map(|&i| weight[i as usize]).sum::<i64>() <= capacity);
    }
}

#[test]
fn lexicographic_fleet_then_distance() {
    // Two well-separated clusters of two customers, capacity 2 per route, up to
    // 4 routes. Tier 0 minimises routes used (-> 2), tier 1 minimises distance.
    let dist = line_matrix(&[0, 10, 11, -10, -11]);
    let demand = [0, 1, 1, 1, 1];
    let k = 4;
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: k,
        objectives: vec![
            ObjectiveTier { minimize: true, terms: (0..k).map(used_list).collect(), max_terms: None },
            ObjectiveTier { minimize: true, terms: (0..k).map(|l| route_cost(l, &dist)).collect(), max_terms: None },
        ],
        constraints: each_con(k, |l| load(l, &demand), Op::Le, 2),
        globals: vec![],
        schedule: None,
    };
    let sol = run(&model, 400);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], 2, "fleet minimised to two routes");
    // With two routes fixed, the best split is the two clusters: each route
    // visits one cluster, cost 2*(dist depot->near) + within-cluster.
    let nonempty = sol.lists.iter().filter(|l| !l.is_empty()).count();
    assert_eq!(nonempty, 2);
}

#[test]
fn qap_matches_brute_force() {
    // One permutation list; minimise sum a[i][j]*b[p[i]][p[j]] over position pairs.
    let a = vec![vec![0, 1, 2, 3], vec![1, 0, 1, 2], vec![2, 1, 0, 1], vec![3, 2, 1, 0]];
    let b = vec![vec![0, 5, 2, 1], vec![5, 0, 3, 2], vec![2, 3, 0, 4], vec![1, 2, 4, 0]];
    let model = CollectionModel {
        items: vec![0, 1, 2, 3],
        lists: 1,
        objectives: min_tier(vec![qap_cost(0, &a, &b)]),
        constraints: vec![],
        globals: vec![],
        schedule: None,
    };
    let mut best = i64::MAX;
    for p in perms(&[0, 1, 2, 3]) {
        let mut c = 0;
        for i in 0..4 {
            for j in 0..4 {
                c += a[i][j] * b[p[i] as usize][p[j] as usize];
            }
        }
        best = best.min(c);
    }
    let sol = run(&model, 400);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], best, "QAP reaches the optimum assignment");
}

#[test]
fn bppc_conflicts_force_extra_bins() {
    // Items 1..4 of weight 1 (capacity is loose), but items 1, 2, 3 mutually
    // conflict, so no two of them may share a bin: the optimum is three bins.
    let weight = [0, 1, 1, 1, 1];
    let capacity = 4;
    let mut conflict = vec![vec![0i64; 5]; 5];
    for &x in &[1usize, 2, 3] {
        for &y in &[1usize, 2, 3] {
            if x != y {
                conflict[x][y] = 1;
            }
        }
    }
    let k = 4;
    let mut constraints = each_con(k, |l| value_reduction(l, ReduceOp::Sum, &weight), Op::Le, capacity);
    for l in 0..k {
        constraints.push(Constraint { reduction: conflict_pairs(l, &conflict), op: Op::Eq, rhs: 0 });
    }
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: k,
        objectives: vec![ObjectiveTier { minimize: true, terms: (0..k).map(used_list).collect(), max_terms: None }],
        constraints,
        globals: vec![],
        schedule: None,
    };
    let sol = run(&model, 500);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], 3, "three mutually conflicting items need three bins");
    for bin in &sol.lists {
        for &x in bin {
            for &y in bin {
                if x != y {
                    assert_eq!(conflict[x as usize][y as usize], 0, "no conflicting pair shares a bin");
                }
            }
        }
    }
}

/// Car-sequencing option penalty: over each window of `q` consecutive cars,
/// count those carrying the option (`option[car]` summed) and penalise the
/// excess over `p`. Uses the `Windows` iterable (inner sum + `Arg(1)` total).
fn option_penalty(list: usize, option: &[i64], q: usize, p: i64) -> Reduction {
    let mut arena = ExprArena::default();
    let car = arena.arg(0);
    let inner = arena.array(Arc::new(option.to_vec()), car);
    let total = arena.arg(1);
    let limit = arena.constant(p);
    let over = arena.sub(total, limit);
    let zero = arena.constant(0);
    let emit = arena.max(zero, over);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Windows { list, size: q, inner }, arena, body: emit, coeff: 1 }
}

#[test]
fn window_objective_keeps_local_search() {
    // A window objective must NOT divert to the exponential domain-exact
    // enumeration. The compiler itself declines this shape, so the orchestrator
    // selects an already-compiled local-search plan.
    use qayd::model::{Model, ModelPackage};
    use qayd::orchestrator::{compile_model_plan, EngineKind, ExecutablePlan, SolveBudget, SolveRequest};
    let option = [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0];
    let model = CollectionModel {
        items: (0..option.len() as i32).collect(),
        lists: 1,
        objectives: min_tier(vec![option_penalty(0, &option, 2, 1)]),
        constraints: vec![],
        globals: vec![],
        schedule: None,
    };
    let package = ModelPackage::new(Model::from_collection(&model));
    let plan = compile_model_plan(&package, &SolveRequest::default(), &SolveBudget::new(None)).unwrap();
    let ExecutablePlan::Single(plan) = plan.description() else {
        panic!("window objective should compile to one engine");
    };
    assert_eq!(plan.engine(), EngineKind::ListLocalSearch);
}

#[test]
fn csp_window_penalty_matches_brute_force() {
    // Five cars, four carry the option; in every window of 2 at most 1 may, so
    // some adjacency penalty is unavoidable. Minimise the total excess.
    let option = [1, 1, 1, 1, 0];
    let q = 2;
    let p = 1;
    let model = CollectionModel {
        items: vec![0, 1, 2, 3, 4],
        lists: 1,
        objectives: min_tier(vec![option_penalty(0, &option, q, p)]),
        constraints: vec![],
        globals: vec![],
        schedule: None,
    };

    let mut best = i64::MAX;
    for perm in perms(&[0, 1, 2, 3, 4]) {
        let mut pen = 0i64;
        for start in 0..=(perm.len() - q) {
            let cnt: i64 = (0..q).map(|k| option[perm[start + k] as usize]).sum();
            pen += (cnt - p).max(0);
        }
        best = best.min(pen);
    }

    let sol = run(&model, 400);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], best, "CSP reaches the minimum option penalty");
}

#[test]
fn top_optional_items_maximize_profit() {
    // Team orienteering: one vehicle, a route-length cap, and an extra "pool"
    // list (index 1) that no reduction references, so items left there are
    // unvisited. Maximise collected profit; far items get dropped to the pool.
    let dist = line_matrix(&[0, 1, 2, 3, 4]); // depot 0; customers 1..4 on a line
    let profit = [0, 10, 10, 10, 10];
    let cap = 6;
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 2, // list 0 = the route, list 1 = the unvisited pool
        objectives: vec![ObjectiveTier { minimize: false, terms: vec![value_reduction(0, ReduceOp::Sum, &profit)], max_terms: None }],
        constraints: vec![Constraint { reduction: edges_cost(0, &dist, 0, 0), op: Op::Le, rhs: cap }],
        globals: vec![],
        schedule: None,
    };

    // Brute force: the best subset whose closed-tour length fits the cap.
    let mut best = 0i64;
    for mask in 0u32..16 {
        let subset: Vec<i32> = (0..4).filter(|c| (mask >> c) & 1 == 1).map(|c| c + 1).collect();
        let d = if subset.is_empty() { 0 } else { perms(&subset).iter().map(|p| path_cost(p, &dist, 0, 0)).min().unwrap() };
        if d <= cap {
            best = best.max(subset.iter().map(|&c| profit[c as usize]).sum::<i64>());
        }
    }

    let sol = run(&model, 400);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], best, "TOP collects the maximum feasible profit");
    assert!(path_cost(&sol.lists[0], &dist, 0, 0) <= cap, "route within the length cap");
    // Every customer is either on the route or in the pool, exactly once.
    let mut served: Vec<i32> = sol.lists.iter().flatten().copied().collect();
    served.sort_unstable();
    assert_eq!(served, vec![1, 2, 3, 4]);
}

#[test]
fn pdptw_same_vehicle_and_precedence() {
    // Two pickup/delivery pairs: pickup 1 -> delivery 2, pickup 3 -> delivery 4.
    // Each pair must be served by the same vehicle (same_list) and the pickup
    // must precede the delivery (a Pairs order check). Minimise total distance.
    let dist = line_matrix(&[0, 3, 4, -3, -4]);
    let k = 2;
    let pairs = [(1i32, 2i32), (3, 4)];
    // is_delivery_of[a][b] = 1 iff a is the delivery whose pickup is b.
    let mut is_del = vec![vec![0i64; 5]; 5];
    for &(p, d) in &pairs {
        is_del[d as usize][p as usize] = 1;
    }
    let mut globals = Vec::new();
    for &(p, d) in &pairs {
        globals.push(GlobalConstraint::SameList { a: p, b: d });
    }
    let mut constraints = Vec::new();
    for l in 0..k {
        constraints.push(Constraint { reduction: precedes_in_list(l, &is_del), op: Op::Eq, rhs: 0 });
    }
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: k,
        objectives: min_tier((0..k).map(|l| route_cost(l, &dist)).collect()),
        constraints,
        globals,
        schedule: None,
    };

    // Brute force: partition into <=2 routes with each pair together and the
    // pickup before its delivery, minimise total closed-tour distance.
    let ord_ok = |route: &[i32]| {
        pairs.iter().all(|&(p, d)| match (route.iter().position(|&x| x == p), route.iter().position(|&x| x == d)) {
            (Some(pi), Some(di)) => pi < di,
            _ => true,
        })
    };
    let best_route = |route: &[i32]| perms(route).iter().filter(|r| ord_ok(r)).map(|r| path_cost(r, &dist, 0, 0)).min();
    let mut best = i64::MAX;
    for code in 0..16u32 {
        let mut r = [Vec::new(), Vec::new()];
        for c in 0..4i32 {
            r[((code >> c) & 1) as usize].push(c + 1);
        }
        let list_of = |x: i32| usize::from(!r[0].contains(&x));
        if pairs.iter().any(|&(p, d)| list_of(p) != list_of(d)) {
            continue;
        }
        if let (Some(a), Some(b)) = (best_route(&r[0]), best_route(&r[1])) {
            best = best.min(a + b);
        }
    }

    let sol = run(&model, 500);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], best, "PDPTW reaches the optimal feasible distance");
    let list_of = |x: i32| sol.lists.iter().position(|l| l.contains(&x)).unwrap();
    for &(p, d) in &pairs {
        assert_eq!(list_of(p), list_of(d), "pair {p}/{d} on the same vehicle");
        let route = &sol.lists[list_of(p)];
        assert!(route.iter().position(|&x| x == p) < route.iter().position(|&x| x == d), "pickup {p} before delivery {d}");
    }
}

#[test]
fn salbp_minimizes_stations_with_precedence() {
    // Five tasks of time 3, station cycle time 6 (two per station), with
    // precedences 1->2, 1->3, 4->5: a task is in a station no later than each of
    // its successors. Minimise the number of used stations.
    let proc = [0, 3, 3, 3, 3, 3];
    let cycle = 6;
    let k = 5;
    let prec = [(1i32, 2i32), (1, 3), (4, 5)];

    let mut globals = Vec::new();
    for &(x, y) in &prec {
        globals.push(GlobalConstraint::ListLe { before: x, after: y });
    }
    let model = CollectionModel {
        items: vec![1, 2, 3, 4, 5],
        lists: k,
        objectives: vec![ObjectiveTier { minimize: true, terms: (0..k).map(used_list).collect(), max_terms: None }],
        constraints: each_con(k, |l| value_reduction(l, ReduceOp::Sum, &proc), Op::Le, cycle),
        globals,
        schedule: None,
    };

    // Brute force over all task-to-station assignments respecting cycle time and
    // precedence; minimise used stations.
    let tasks = [1usize, 2, 3, 4, 5];
    let mut best = usize::MAX;
    for code in 0..k.pow(5) {
        let mut st = [0usize; 6];
        let mut c = code;
        for &t in &tasks {
            st[t] = c % k;
            c /= k;
        }
        let mut load = vec![0i64; k];
        for &t in &tasks {
            load[st[t]] += proc[t];
        }
        if load.iter().any(|&l| l > cycle) {
            continue;
        }
        if prec.iter().any(|&(x, y)| st[x as usize] > st[y as usize]) {
            continue;
        }
        best = best.min((0..k).filter(|&s| load[s] > 0).count());
    }

    let sol = run(&model, 600);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0] as usize, best, "SALBP reaches the minimum station count");
    let station_of = |t: i32| sol.lists.iter().position(|l| l.contains(&t)).unwrap();
    for &(x, y) in &prec {
        assert!(station_of(x) <= station_of(y), "precedence {x} -> {y} respected");
    }
    for st in &sol.lists {
        assert!(st.iter().map(|&t| proc[t as usize]).sum::<i64>() <= cycle, "cycle time respected");
    }
}

#[test]
fn cvrptw_small_matches_brute_force() {
    // Depot 0; customers 1..4 on a line at positions 0,2,4,6,8. Customers 1 and
    // 2 share a tight deadline (5) they cannot both meet on one route, so a
    // feasible solution must split them across the two routes.
    let dist = line_matrix(&[0, 2, 4, 6, 8]);
    let demand = [0, 1, 1, 1, 1];
    let service = [0, 1, 1, 1, 1];
    let earliest = [0, 0, 0, 0, 0];
    let latest = [0, 5, 5, 20, 20];
    let capacity = 2;
    let depot = 0;
    let k = 2;

    let mut constraints = each_con(k, |l| load(l, &demand), Op::Le, capacity);
    for l in 0..k {
        constraints.push(Constraint { reduction: lateness_scan(l, &dist, &earliest, &latest, &service, depot), op: Op::Le, rhs: 0 });
    }
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: k,
        objectives: min_tier((0..k).map(|l| edges_cost(l, &dist, depot, depot)).collect()),
        constraints,
        globals: vec![],
        schedule: None,
    };

    // Brute force: min total closed-tour distance over capacity- and
    // time-window-feasible two-route partitions.
    let best_route = |route: &[i32]| -> Option<i64> {
        if route.iter().map(|&c| demand[c as usize]).sum::<i64>() > capacity {
            return None;
        }
        perms(route)
            .iter()
            .filter(|p| tw_feasible(p, &dist, &earliest, &latest, &service, depot))
            .map(|p| path_cost(p, &dist, depot, depot))
            .min()
    };
    let mut best = i64::MAX;
    for code in 0..16u32 {
        let mut routes = [Vec::new(), Vec::new()];
        for c in 0..4i32 {
            routes[((code >> c) & 1) as usize].push(c + 1);
        }
        if let (Some(a), Some(b)) = (best_route(&routes[0]), best_route(&routes[1])) {
            best = best.min(a + b);
        }
    }

    let sol = run(&model, 500);
    assert!(sol.feasible, "a time-window-feasible plan exists");
    assert_eq!(sol.objectives[0], best, "CVRPTW reaches the optimal feasible distance");
    // Customers 1 and 2 must be on different routes.
    for route in &sol.lists {
        assert!(!(route.contains(&1) && route.contains(&2)), "1 and 2 cannot share a route");
    }
}

/// Aircraft landing cost over a landing-order list: a scan computes each plane's
/// landing time `max(earliest[cur], prev_landing + sep[prev][cur])`, and the
/// body emits the earliness/tardiness penalty around the target time using a
/// ternary (`Lt` + `IfThenElse`).
#[allow(clippy::too_many_arguments)]
fn alp_cost(
    list: usize,
    earliest: &[i64],
    target: &[i64],
    early_cost: &[i64],
    tardy_cost: &[i64],
    sep: &[Vec<i64>],
    boundary: i32,
) -> Reduction {
    let mut arena = ExprArena::default();
    let cur = arena.arg(0);
    let acc = arena.arg(1);
    let prev = arena.arg(2);
    let gap = arena.matrix(Arc::new(sep.to_vec()), prev, cur);
    let cand = arena.add(acc, gap);
    let earl = arena.array(Arc::new(earliest.to_vec()), cur);
    let step = arena.max(earl, cand);
    // emit: landing = Arg1; cost = landing < target ? early : tardy
    let cur2 = arena.arg(0);
    let landing = arena.arg(1);
    let tgt = arena.array(Arc::new(target.to_vec()), cur2);
    let is_early = arena.lt(landing, tgt);
    let ec = arena.array(Arc::new(early_cost.to_vec()), cur2);
    let early_amt = arena.sub(tgt, landing);
    let early = arena.mul(ec, early_amt);
    let tc = arena.array(Arc::new(tardy_cost.to_vec()), cur2);
    let tardy_amt = arena.sub(landing, tgt);
    let tardy = arena.mul(tc, tardy_amt);
    let emit = arena.if_then_else(is_early, early, tardy);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list, init: 0, boundary, step, end: None }, arena, body: emit, coeff: 1 }
}

#[test]
fn alp_matches_brute_force() {
    let n = 4;
    let boundary = n as i32; // separation row/col `n` is the "before first plane" zero row
    let earliest = [0, 0, 0, 0];
    let target = [10, 20, 15, 25];
    let early_cost = [1, 1, 1, 1];
    let tardy_cost = [3, 3, 3, 3];
    // Uniform 5-unit separation between distinct planes; 0 from/to the boundary.
    let mut sep = vec![vec![5i64; n + 1]; n + 1];
    for row in sep.iter_mut() {
        row[n] = 0;
    }
    sep[n] = vec![0; n + 1];

    let model = CollectionModel {
        items: (0..n as i32).collect(),
        lists: 1,
        objectives: min_tier(vec![alp_cost(0, &earliest, &target, &early_cost, &tardy_cost, &sep, boundary)]),
        constraints: vec![],
        globals: vec![],
        schedule: None,
    };

    let mut best = i64::MAX;
    for p in perms(&(0..n as i32).collect::<Vec<_>>()) {
        let mut acc = 0i64;
        let mut prev = n; // boundary
        let mut cost = 0i64;
        for &c in &p {
            let c = c as usize;
            let land = earliest[c].max(acc + sep[prev][c]);
            cost += if land < target[c] { early_cost[c] * (target[c] - land) } else { tardy_cost[c] * (land - target[c]) };
            acc = land;
            prev = c;
        }
        best = best.min(cost);
    }

    let sol = run(&model, 400);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], best, "ALP reaches the minimum earliness/tardiness cost");
}

#[test]
fn scan_accumulator_index_is_rejected() {
    // Indexing a table by the scan accumulator (Arg(1)) has no finite domain and
    // must be rejected at construction, not read as a silent zero.
    let mut arena = ExprArena::default();
    let acc = arena.arg(1);
    let body = arena.array(Arc::new(vec![0, 1, 2]), acc); // array[acc] is illegal
    let step = arena.constant(0);
    let r =
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list: 0, init: 0, boundary: 0, step, end: None }, arena, body, coeff: 1 };
    let model = CollectionModel {
        items: vec![1, 2, 3],
        lists: 1,
        objectives: min_tier(vec![r]),
        constraints: vec![],
        globals: vec![],
        schedule: None,
    };
    assert!(model.validate().is_err(), "indexing by the accumulator is rejected");
}

#[test]
fn pairs_out_of_range_index_is_rejected() {
    // Item 9 used to index a 3x3 matrix is out of range -> rejected.
    let mut arena = ExprArena::default();
    let a = arena.arg(0);
    let b = arena.arg(1);
    let body = arena.matrix(Arc::new(vec![vec![0; 3]; 3]), a, b);
    let r = Reduction { op: ReduceOp::Sum, iterable: Iterable::Pairs(0), arena, body, coeff: 1 };
    let model = CollectionModel {
        items: vec![1, 9],
        lists: 1,
        objectives: min_tier(vec![r]),
        constraints: vec![],
        globals: vec![],
        schedule: None,
    };
    assert!(model.validate().is_err(), "item 9 is out of the 3x3 matrix range");
}

#[test]
fn guarded_out_of_range_branch_is_accepted() {
    // `if item < 3 then matrix[item][item] else 0` over items {1, 9}. At item 9
    // the guard is false, so the out-of-range matrix access in the unselected
    // then-branch is never reached. Validation is lazy (matching solve), so this
    // guarded model must be accepted, not rejected.
    let mut arena = ExprArena::default();
    let item = arena.arg(0);
    let three = arena.constant(3);
    let guard = arena.lt(item, three);
    let unsafe_access = arena.matrix(Arc::new(vec![vec![0; 3]; 3]), item, item);
    let zero = arena.constant(0);
    let body = arena.if_then_else(guard, unsafe_access, zero);
    let r = Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(0), arena, body, coeff: 1 };
    let model = CollectionModel {
        items: vec![1, 9],
        lists: 1,
        objectives: min_tier(vec![r]),
        constraints: vec![],
        globals: vec![],
        schedule: None,
    };
    assert!(model.validate().is_ok(), "a guarded out-of-range branch must validate");
}

#[test]
fn scan_conditional_body_does_not_hide_step_range() {
    // `Scan` also evaluates `step` at solve, so a conditional anywhere in the
    // arena must not switch off the static range check of `step`. Here
    // `step = array[item]` reads out of a length-3 array at item 9; a harmless
    // conditional body must not make the model validate.
    let mut arena = ExprArena::default();
    let item = arena.arg(0);
    let step = arena.array(Arc::new(vec![0i64; 3]), item);
    let cond = arena.constant(1);
    let then_zero = arena.constant(0);
    let else_zero = arena.constant(0);
    let body = arena.if_then_else(cond, then_zero, else_zero);
    let r =
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list: 0, init: 0, boundary: 0, step, end: None }, arena, body, coeff: 1 };
    let model =
        CollectionModel { items: vec![9], lists: 1, objectives: min_tier(vec![r]), constraints: vec![], globals: vec![], schedule: None };
    assert!(model.validate().is_err(), "scan step out of range must be rejected despite a conditional body");
}

#[test]
fn vbp_two_dimensions() {
    // Each bin must respect a capacity in two dimensions at once.
    let w0 = [0, 3, 3, 3, 3];
    let w1 = [0, 4, 4, 4, 4];
    let k = 4;
    let mut constraints = each_con(k, |l| value_reduction(l, ReduceOp::Sum, &w0), Op::Le, 6);
    constraints.extend(each_con(k, |l| value_reduction(l, ReduceOp::Sum, &w1), Op::Le, 8));
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: k,
        objectives: vec![ObjectiveTier { minimize: true, terms: (0..k).map(used_list).collect(), max_terms: None }],
        constraints,
        globals: vec![],
        schedule: None,
    };
    let sol = run(&model, 300);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], 2, "two items per bin in both dimensions");
    for bin in &sol.lists {
        assert!(bin.iter().map(|&i| w0[i as usize]).sum::<i64>() <= 6);
        assert!(bin.iter().map(|&i| w1[i as usize]).sum::<i64>() <= 8);
    }
}

#[test]
fn jssp_interval_optimum() {
    // 2 jobs x 2 machines. Ops 0,1 = job 0 (op0->op1); ops 2,3 = job 1 (op2->op3).
    // Machine 0 runs ops {0,3}, machine 1 runs ops {1,2} (no_overlap each).
    let dur = [3i64, 2, 2, 4];
    let horizon: i64 = dur.iter().sum();
    let intervals: Vec<IntervalVar> = dur.iter().map(|&d| IntervalVar { duration: d, horizon, modes: vec![], optional: false }).collect();
    let sched = Schedule {
        intervals,
        precedences: vec![(0, 1), (2, 3)],
        resources: vec![Resource::NoOverlap(vec![0, 3]), Resource::NoOverlap(vec![1, 2])],
        minimize_makespan: true,
    };
    let model = schedule_model(sched);

    // Brute force the disjunctive optimum: choose each machine's op order, take
    // the longest path; a cyclic choice is infeasible.
    let makespan = |m0_first: bool, m1_first: bool| -> i64 {
        let mut edges = vec![(0usize, 1usize), (2, 3)];
        edges.push(if m0_first { (0, 3) } else { (3, 0) });
        edges.push(if m1_first { (1, 2) } else { (2, 1) });
        let mut start = [0i64; 4];
        for _ in 0..16 {
            for &(a, b) in &edges {
                let need = start[a] + dur[a];
                if need > start[b] {
                    start[b] = need;
                }
            }
        }
        if edges.iter().any(|&(a, b)| start[a] + dur[a] > start[b]) {
            return i64::MAX; // cyclic deadlock
        }
        (0..4).map(|i| start[i] + dur[i]).max().unwrap()
    };
    let best = [(true, true), (true, false), (false, true), (false, false)].iter().map(|&(a, b)| makespan(a, b)).min().unwrap();

    let sol = run(&model, 600);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], best, "JSSP reaches the optimal makespan");
}

#[test]
fn rcpsp_interval_optimum() {
    // 4 tasks, one resource of capacity 2. Precedence 0->2, 1->3.
    let dur = [2i64, 2, 1, 1];
    let demand = [1i64, 2, 1, 2];
    let cap = 2;
    let horizon: i64 = dur.iter().sum();
    let intervals: Vec<IntervalVar> = dur.iter().map(|&d| IntervalVar { duration: d, horizon, modes: vec![], optional: false }).collect();
    let sched = Schedule {
        intervals,
        precedences: vec![(0, 2), (1, 3)],
        resources: vec![Resource::Cumulative { demands: (0..4).map(|i| (i, demand[i])).collect(), capacity: cap }],
        minimize_makespan: true,
    };
    let model = schedule_model(sched);

    // Brute force over integer start times in [0, horizon].
    let feasible_makespan = |s: &[i64; 4]| -> Option<i64> {
        if (s[0] + dur[0] > s[2]) || (s[1] + dur[1] > s[3]) {
            return None;
        }
        // cumulative: no instant over capacity
        for t in 0..horizon {
            let usage: i64 = (0..4).filter(|&i| s[i] <= t && t < s[i] + dur[i]).map(|i| demand[i]).sum();
            if usage > cap {
                return None;
            }
        }
        Some((0..4).map(|i| s[i] + dur[i]).max().unwrap())
    };
    let mut best = i64::MAX;
    for a in 0..=horizon {
        for b in 0..=horizon {
            for c in 0..=horizon {
                for d in 0..=horizon {
                    if let Some(mk) = feasible_makespan(&[a, b, c, d]) {
                        best = best.min(mk);
                    }
                }
            }
        }
    }

    let sol = run(&model, 800);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], best, "RCPSP reaches the optimal makespan");
}

#[test]
fn fjsp_interval_optimum() {
    // Flexible job shop: 2 jobs x 2 ops, 2 machines. Each op may run on either
    // machine with a machine-dependent duration. Op = job*2 + k; job order
    // 0->1 and 2->3; the chosen machine drives the no-overlap.
    let proc = [[2i64, 3], [3, 2], [2, 2], [4, 1]]; // proc[op][machine]
    let horizon: i64 = 12;
    let intervals: Vec<IntervalVar> = proc
        .iter()
        .enumerate()
        .map(|(operation, p)| IntervalVar {
            duration: 0,
            horizon,
            modes: vec![
                Mode { reference: Some(operation * 2), machine: 0, duration: p[0], start_window: (0, horizon - p[0]) },
                Mode { reference: Some(operation * 2 + 1), machine: 1, duration: p[1], start_window: (0, horizon - p[1]) },
            ],
            optional: false,
        })
        .collect();
    let sched =
        Schedule { intervals, precedences: vec![(0, 1), (2, 3)], resources: vec![Resource::MachineNoOverlap], minimize_makespan: true };
    let model = schedule_model(sched);

    // Brute force: machine choice per op (16 combos) x integer starts in [0, H].
    let mut best = i64::MAX;
    for bits in 0..16usize {
        let mc = [bits & 1, (bits >> 1) & 1, (bits >> 2) & 1, (bits >> 3) & 1];
        let dur = [proc[0][mc[0]], proc[1][mc[1]], proc[2][mc[2]], proc[3][mc[3]]];
        for s0 in 0..=horizon {
            for s1 in 0..=horizon {
                if s0 + dur[0] > s1 {
                    continue; // precedence 0 -> 1
                }
                for s2 in 0..=horizon {
                    for s3 in 0..=horizon {
                        if s2 + dur[2] > s3 {
                            continue; // precedence 2 -> 3
                        }
                        let s = [s0, s1, s2, s3];
                        let end = [s0 + dur[0], s1 + dur[1], s2 + dur[2], s3 + dur[3]];
                        if end.iter().zip(&intervals_horizon()).any(|(&e, &h)| e > h) {
                            continue;
                        }
                        // Machine no-overlap: same chosen machine, no overlap.
                        let mut ok = true;
                        'pairs: for i in 0..4 {
                            for j in (i + 1)..4 {
                                if mc[i] == mc[j] && s[i].max(s[j]) < end[i].min(end[j]) {
                                    ok = false;
                                    break 'pairs;
                                }
                            }
                        }
                        if ok {
                            best = best.min(*end.iter().max().unwrap());
                        }
                    }
                }
            }
        }
    }

    let sol = run(&model, 1200);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], best, "FJSP reaches the optimal makespan");
    // The reported machine assignment must be consistent with the makespan.
    assert_eq!(sol.machines.len(), 4);
    assert!(sol.machines.iter().all(|&m| m == 0 || m == 1));
}

fn intervals_horizon() -> [i64; 4] {
    [12, 12, 12, 12]
}

#[test]
fn machine_no_overlap_replay_accepts_touching_blocks_after_grouped_sort() {
    let model = machine_no_overlap_model(&[(1, 2, false), (0, 2, false), (1, 1, false), (0, 2, false)], 8);
    let solution = machine_solution(&[2, 0, 4, 2], &[true; 4], &[1, 0, 1, 0], &[Some(0), Some(1), Some(2), Some(3)], 5);

    assert_eq!(verify_collection_solution(&model, &solution).unwrap(), vec![5]);
}

#[test]
fn machine_no_overlap_replay_rejects_overlap_after_grouped_sort() {
    let model = machine_no_overlap_model(&[(1, 3, false), (0, 2, false), (1, 2, false), (1, 1, false)], 8);
    let solution = machine_solution(&[1, 0, 2, 5], &[true; 4], &[1, 0, 1, 1], &[Some(0), Some(1), Some(2), Some(3)], 6);

    let error = verify_collection_solution(&model, &solution).unwrap_err();
    assert!(error.contains("intervals 0 and 2 overlap"));
}

#[test]
fn machine_no_overlap_replay_ignores_absent_intervals() {
    let model = schedule_model(Schedule {
        intervals: vec![
            IntervalVar {
                duration: 0,
                horizon: 8,
                modes: vec![Mode { reference: Some(0), machine: 0, duration: 3, start_window: (0, 5) }],
                optional: false,
            },
            IntervalVar { duration: 4, horizon: 8, modes: Vec::new(), optional: true },
            IntervalVar {
                duration: 0,
                horizon: 8,
                modes: vec![Mode { reference: Some(2), machine: 0, duration: 2, start_window: (0, 6) }],
                optional: false,
            },
        ],
        precedences: Vec::new(),
        resources: vec![Resource::MachineNoOverlap],
        minimize_makespan: true,
    });
    let solution = machine_solution(&[0, 1, 3], &[true, false, true], &[0, -1, 0], &[Some(0), None, Some(2)], 5);

    assert_eq!(verify_collection_solution(&model, &solution).unwrap(), vec![5]);
}

#[test]
fn machine_no_overlap_replay_rejects_equal_starts_on_one_machine_deterministically() {
    let model = machine_no_overlap_model(&[(2, 2, false), (2, 1, false), (0, 1, false)], 6);
    let solution = machine_solution(&[1, 1, 0], &[true; 3], &[2, 2, 0], &[Some(0), Some(1), Some(2)], 3);

    let error = verify_collection_solution(&model, &solution).unwrap_err();
    assert!(error.contains("intervals 0 and 1 overlap"));
}

#[test]
fn machine_no_overlap_replay_does_not_let_zero_length_peers_mask_later_overlap() {
    let model = machine_no_overlap_model(&[(0, 10, false), (0, 0, false), (0, 1, false)], 12);
    let solution = machine_solution(&[0, 0, 5], &[true; 3], &[0, 0, 0], &[Some(0), Some(1), Some(2)], 10);

    let error = verify_collection_solution(&model, &solution).unwrap_err();
    assert!(error.contains("intervals 0 and 2 overlap"));
}

#[test]
fn machine_no_overlap_replay_matches_the_quadratic_oracle_on_small_cases() {
    for durations in [[0, 0, 1], [0, 1, 0], [2, 0, 1], [2, 1, 3]] {
        let horizon = 6;
        let spec = durations.into_iter().map(|duration| (0usize, duration, false)).collect::<Vec<_>>();
        let model = machine_no_overlap_model(&spec, horizon);

        for s0 in 0..=4 {
            for s1 in 0..=4 {
                for s2 in 0..=4 {
                    let starts = [s0, s1, s2];
                    let makespan = starts.into_iter().zip(durations).map(|(start, duration)| start + duration).max().unwrap();
                    let solution = machine_solution(&starts, &[true; 3], &[0, 0, 0], &[Some(0), Some(1), Some(2)], makespan);
                    let oracle_ok = quadratic_machine_no_overlap_ok(&starts, &durations, horizon);
                    let replay = verify_collection_solution(&model, &solution);

                    assert_eq!(replay.is_ok(), oracle_ok, "durations={durations:?}, starts={starts:?}");
                    if oracle_ok {
                        assert_eq!(replay.unwrap(), vec![makespan], "durations={durations:?}, starts={starts:?}");
                    }
                }
            }
        }
    }
}

/// In-place lexicographic next permutation; false when the sequence is the last.
fn next_permutation(a: &mut [usize]) -> bool {
    if a.len() < 2 {
        return false;
    }
    let mut i = a.len() - 1;
    while i > 0 && a[i - 1] >= a[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = a.len() - 1;
    while a[j] <= a[i - 1] {
        j -= 1;
    }
    a.swap(i - 1, j);
    a[i..].reverse();
    true
}

#[test]
fn msp_select_kth_quantile() {
    // Mining-scheduling flavour: order n blocks for extraction (one permutation
    // list). A block placed at 1-indexed position p realises value g - rate*p
    // (later extraction is discounted). The objective is a risk quantile: the
    // k-th smallest realised value (value-at-risk), maximised so the order lifts
    // the worst outcomes.
    let g = [10i64, 3, 8, 5, 12, 1];
    let n = g.len();
    let rate = 2i64;
    let k = 2usize; // 0-indexed: the 3rd-smallest realised value
    let items: Vec<i32> = (0..n as i32).collect();

    let mut arena = ExprArena::default();
    // step(cur, acc, prev) -> acc + 1  (acc is the 1-indexed position)
    let a1 = arena.arg(1);
    let one = arena.constant(1);
    let step = arena.add(a1, one);
    // emit(cur, acc, prev) -> g[cur] - rate * acc
    let cur = arena.arg(0);
    let gcur = arena.array(Arc::new(g.to_vec()), cur);
    let acc = arena.arg(1);
    let rate_c = arena.constant(rate);
    let racc = arena.mul(rate_c, acc);
    let emit = arena.sub(gcur, racc);
    let reduction = Reduction {
        op: ReduceOp::SelectKth(k),
        iterable: Iterable::Scan { list: 0, init: 0, boundary: 0, step, end: None },
        arena,
        body: emit,
        coeff: 1,
    };

    let model = CollectionModel {
        items,
        lists: 1,
        objectives: vec![ObjectiveTier { minimize: false, terms: vec![reduction], max_terms: None }],
        constraints: vec![],
        globals: vec![],
        schedule: None,
    };

    // Oracle: max over all orderings of the k-th smallest realised value.
    let mut order: Vec<usize> = (0..n).collect();
    let mut best = i64::MIN;
    loop {
        let mut realised: Vec<i64> = order.iter().enumerate().map(|(idx, &it)| g[it] - rate * (idx as i64 + 1)).collect();
        realised.sort_unstable();
        best = best.max(realised[k]);
        if !next_permutation(&mut order) {
            break;
        }
    }

    let sol = run(&model, 600);
    assert!(sol.feasible);
    assert_eq!(sol.objectives[0], best, "MSP reaches the optimal value-at-risk quantile");
}

/// Exhaustive cross-check that incremental candidate scoring equals a full
/// recompute, over every single-list edit (remove/insert/replace/move/reverse),
/// for routing-shaped models (closed-tour edge cost + capacity + used + count)
/// on random asymmetric and symmetric matrices. Drives the engine's
/// `audit_incremental` hook.
#[test]
fn incremental_scoring_matches_full_recompute() {
    // Small deterministic LCG so the test is reproducible without rand.
    let mut seed = 0x1234_5678u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as i64
    };

    for trial in 0..40 {
        let n = 3 + (trial % 5); // universe size 3..7
        let items: Vec<i32> = (0..n as i32).collect();
        for symmetric in [false, true] {
            let mut dist: Vec<Vec<i64>> = (0..n).map(|_| (0..n).map(|_| next().rem_euclid(20)).collect()).collect();
            if symmetric {
                for i in 0..n {
                    let (before, tail) = dist.split_at_mut(i);
                    let row = &mut tail[0];
                    for (j, prev_row) in before.iter().enumerate() {
                        row[j] = prev_row[i];
                    }
                }
            }
            let demand: Vec<i64> = (0..n).map(|_| next().rem_euclid(9) + 1).collect();
            let cap = 15;
            let lists = 2 + (trial % 2); // 2 or 3 lists

            // Objective tier 0: total edge cost over each list's closed tour (depot 0).
            // plus fleet (used) and a count term, all incremental-supported.
            let mut obj: Vec<Reduction> = Vec::new();
            for l in 0..lists {
                obj.push(edges_cost(l, &dist, 0, 0));
                obj.push(used_list(l));
                obj.push(count_items(l));
            }
            // Constraint: each list's demand sum within capacity (Sum over Items).
            let cons = each_con(lists, |l| load(l, &demand), Op::Le, cap);

            let model = CollectionModel {
                items: items.clone(),
                lists,
                objectives: vec![ObjectiveTier { minimize: true, terms: obj, max_terms: None }],
                constraints: cons,
                globals: vec![],
                schedule: None,
            };

            // Partition the items across the lists (round-robin-ish, varied).
            let mut contents: Vec<Vec<i32>> = vec![Vec::new(); lists];
            for (i, &it) in items.iter().enumerate() {
                contents[(i + (trial % lists)) % lists].push(it);
            }
            let checked = qayd::engines::ls::lists::audit_incremental(&model, &contents);
            assert!(checked > 0, "audit should check at least one edit");
        }
    }
}
