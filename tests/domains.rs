use qayd::constraints::interval::{cumulative, exactly_one_mode, interval_precedence, makespan_bound, no_overlap};
use qayd::constraints::list::{item_precedence, list_cardinality, list_item_sum, list_len, partition, same_list};
use qayd::constraints::scheduling::cumulative as cumulative_integer;
use qayd::{first_domain_solution, solve_domains, solve_search, IntervalId, IntervalPresence, SearchControl, Solver, Store, VarId};

#[test]
fn domain_intervals_are_integer_backed() {
    // List/interval ids are their own id spaces, but both are now backed by
    // integer variables (the learning engine's source of truth): a list allocates
    // one membership variable per item plus a length variable, a mandatory
    // interval one start variable, an optional interval also a presence variable.
    let mut store = Store::new();
    let list = store.new_list(vec![10, 20]); // ListId 0: members VarId(0),(1) + length VarId(2)
    let mandatory = store.new_interval(0, 10, 3); // IntervalId 0: start = VarId(3)
    let optional = store.new_optional_interval(0, 10, 3); // IntervalId 1: start = VarId(4), presence = VarId(5)
    let x = store.new_var_range(1, 5); // VarId(6)

    assert_eq!(list.index(), 0);
    assert_eq!(mandatory.index(), 0);
    assert_eq!(optional.index(), 1);
    assert_eq!(x, VarId(6));
    assert_eq!(store.num_lists(), 1);
    assert_eq!(store.num_intervals(), 2);
    assert_eq!(store.num_vars(), 7); // 3 (list: 2 members + length) + 1 (mandatory) + 2 (optional) + 1 (x)
}

#[test]
fn integer_only_models_keep_variable_ids() {
    // A model that creates no domains allocates variable ids from 0,
    // unaffected by the integer-backing of intervals.
    let mut store = Store::new();
    assert_eq!(store.new_var_range(0, 9), VarId(0));
    assert_eq!(store.new_var_set(&[1, 2]), VarId(1));
    assert_eq!(store.num_vars(), 2);
}

#[test]
fn domain_list_membership_is_trailed() {
    let mut store = Store::new();
    let list = store.new_list(vec![10, 20, 30]);

    store.push_level();
    assert_eq!(store.require_list_item(list, 10), Ok(true));
    assert_eq!(store.forbid_list_item(list, 20), Ok(true));
    assert!(store.list_required(list, 10));
    assert!(!store.list_possible(list, 20));
    assert!(store.list_possible(list, 30));
    assert_eq!(store.list_required_count(list), 1);
    assert_eq!(store.list_possible_count(list), 2);
    // Length stays its full range: membership and length are coupled only by the
    // ListCardinality propagator, which is not posted here.
    assert_eq!((store.list_len_min(list), store.list_len_max(list)), (0, 3));

    store.pop_level();
    assert!(!store.list_required(list, 10));
    assert!(store.list_possible(list, 20));
    assert_eq!(store.list_required_count(list), 0);
    assert_eq!(store.list_possible_count(list), 3);
    assert_eq!((store.list_len_min(list), store.list_len_max(list)), (0, 3));
}

#[test]
fn domain_list_rejects_inconsistent_membership_and_length() {
    // Store-level direct conflicts (membership variable already fixed the other way).
    let mut store = Store::new();
    let list = store.new_list(vec![1, 2]);
    assert_eq!(store.require_list_item(list, 1), Ok(true));
    assert!(store.forbid_list_item(list, 1).is_err()); // item 1 already required
    assert!(store.require_list_item(list, 99).is_err()); // item not in the universe

    // Cardinality conflicts are the ListCardinality propagator's job, not the Store's.
    let mut solver = Solver::new();
    let list = solver.store.new_list(vec![1, 2]);
    list_cardinality(&mut solver, list);
    solver.store.set_list_len_max(list, 1).unwrap();
    solver.store.require_list_item(list, 1).unwrap();
    solver.store.require_list_item(list, 2).unwrap(); // Store allows it; coupling not yet run
    assert!(solver.propagate().is_err(), "two required items exceed length_max 1");

    let mut solver = Solver::new();
    let list2 = solver.store.new_list(vec![3, 4]);
    list_cardinality(&mut solver, list2);
    solver.store.set_list_len_min(list2, 2).unwrap();
    solver.store.forbid_list_item(list2, 3).unwrap();
    assert!(solver.propagate().is_err(), "forbidding an item drops below length_min 2");
}

#[test]
fn domain_interval_bounds_are_trailed() {
    let mut store = Store::new();
    let interval = store.new_interval(2, 12, 5);

    assert_eq!(store.interval_presence(interval), IntervalPresence::Present);
    assert_eq!(store.interval_duration(interval), 5);
    assert_eq!((store.interval_start_min(interval), store.interval_start_max(interval)), (2, 12));
    assert_eq!((store.interval_end_min(interval), store.interval_end_max(interval)), (7, 17));

    store.push_level();
    assert_eq!(store.set_interval_start_min(interval, 6), Ok(true));
    assert_eq!(store.set_interval_start_max(interval, 9), Ok(true));
    assert_eq!((store.interval_start_min(interval), store.interval_start_max(interval)), (6, 9));
    assert_eq!((store.interval_end_min(interval), store.interval_end_max(interval)), (11, 14));
    assert!(store.set_interval_start_min(interval, 10).is_err());
    store.pop_level();

    assert_eq!((store.interval_start_min(interval), store.interval_start_max(interval)), (2, 12));
    assert_eq!((store.interval_end_min(interval), store.interval_end_max(interval)), (7, 17));
}

#[test]
fn optional_interval_presence_is_trailed() {
    let mut store = Store::new();
    let interval = store.new_optional_interval(0, 4, 2);

    assert_eq!(store.interval_presence(interval), IntervalPresence::Optional);
    store.push_level();
    assert_eq!(store.require_interval_presence(interval), Ok(true));
    assert_eq!(store.interval_presence(interval), IntervalPresence::Present);
    assert!(store.forbid_interval_presence(interval).is_err());
    store.pop_level();

    assert_eq!(store.interval_presence(interval), IntervalPresence::Optional);
    assert_eq!(store.forbid_interval_presence(interval), Ok(true));
    assert_eq!(store.interval_presence(interval), IntervalPresence::Absent);
    assert!(store.require_interval_presence(interval).is_err());
}

#[test]
fn domain_list_cardinality_closes_membership() {
    // length pinned to 3 over a 3-item universe requires every item.
    let mut solver = Solver::new();
    let list = solver.store.new_list(vec![1, 2, 3]);
    list_cardinality(&mut solver, list);
    solver.store.set_list_len_min(list, 3).unwrap();
    solver.propagate().unwrap();
    assert!(solver.store.list_required(list, 1));
    assert!(solver.store.list_required(list, 2));
    assert!(solver.store.list_required(list, 3));

    // length pinned to 0 forbids every item.
    let list = solver.store.new_list(vec![4, 5, 6]);
    list_cardinality(&mut solver, list);
    solver.store.set_list_len_max(list, 0).unwrap();
    solver.propagate().unwrap();
    assert!(!solver.store.list_possible(list, 4));
    assert!(!solver.store.list_possible(list, 5));
    assert!(!solver.store.list_possible(list, 6));
}

#[test]
fn domain_partition_forbids_item_from_other_lists() {
    let mut solver = Solver::new();
    let a = solver.store.new_list(vec![1, 2]);
    let b = solver.store.new_list(vec![1, 2]);
    partition(&mut solver, &[a, b], &[1, 2]);

    assert_eq!(solver.store.require_list_item(a, 1), Ok(true));
    solver.propagate().unwrap();

    assert!(solver.store.list_required(a, 1));
    assert!(!solver.store.list_possible(b, 1));
    assert!(solver.store.list_possible(b, 2));
}

#[test]
fn domain_partition_requires_sole_possible_owner() {
    let mut solver = Solver::new();
    let a = solver.store.new_list(vec![7]);
    let b = solver.store.new_list(vec![7]);
    partition(&mut solver, &[a, b], &[7]);

    assert_eq!(solver.store.forbid_list_item(a, 7), Ok(true));
    solver.propagate().unwrap();

    assert!(solver.store.list_required(b, 7));
    assert!(!solver.store.list_possible(a, 7));
}

#[test]
fn domain_partition_reports_impossible_item() {
    let mut solver = Solver::new();
    let a = solver.store.new_list(vec![9]);
    let b = solver.store.new_list(vec![9]);
    partition(&mut solver, &[a, b], &[9]);

    assert_eq!(solver.store.forbid_list_item(a, 9), Ok(true));
    assert_eq!(solver.store.forbid_list_item(b, 9), Ok(true));

    assert!(solver.propagate().is_err());
}

#[test]
fn domain_partition_reports_two_required_owners() {
    let mut solver = Solver::new();
    let a = solver.store.new_list(vec![11]);
    let b = solver.store.new_list(vec![11]);
    partition(&mut solver, &[a, b], &[11]);

    assert_eq!(solver.store.require_list_item(a, 11), Ok(true));
    assert_eq!(solver.store.require_list_item(b, 11), Ok(true));

    assert!(solver.propagate().is_err());
}

#[test]
fn domain_list_events_wake_partition() {
    let mut solver = Solver::new();
    let a = solver.store.new_list(vec![13]);
    let b = solver.store.new_list(vec![13]);
    let prop = partition(&mut solver, &[a, b], &[13]);

    solver.propagate().unwrap();
    assert!(solver.fd_at_fixpoint());

    assert_eq!(solver.store.require_list_item(a, 13), Ok(true));
    assert_eq!(solver.peek_prop(), Some(prop));
    solver.propagate().unwrap();

    assert!(!solver.store.list_possible(b, 13));
}

#[test]
fn domain_same_list_propagates_known_owner() {
    let mut solver = Solver::new();
    let a = solver.store.new_list(vec![1, 2]);
    let b = solver.store.new_list(vec![1, 2]);
    partition(&mut solver, &[a, b], &[1, 2]);
    same_list(&mut solver, &[a, b], 1, 2);

    assert_eq!(solver.store.require_list_item(a, 1), Ok(true));
    solver.propagate().unwrap();

    assert!(solver.store.list_required(a, 2));
    assert!(!solver.store.list_possible(b, 1));
    assert!(!solver.store.list_possible(b, 2));
}

#[test]
fn domain_same_list_prunes_impossible_owner() {
    let mut solver = Solver::new();
    let a = solver.store.new_list(vec![1, 2]);
    let b = solver.store.new_list(vec![1, 2]);
    partition(&mut solver, &[a, b], &[1, 2]);
    same_list(&mut solver, &[a, b], 1, 2);

    assert_eq!(solver.store.forbid_list_item(a, 1), Ok(true));
    solver.propagate().unwrap();

    assert!(!solver.store.list_possible(a, 2));
    assert!(solver.store.list_required(b, 1));
    assert!(solver.store.list_required(b, 2));
}

#[test]
fn domain_item_precedence_prunes_list_indices() {
    let mut solver = Solver::new();
    let a = solver.store.new_list(vec![1, 2]);
    let b = solver.store.new_list(vec![1, 2]);
    let c = solver.store.new_list(vec![1, 2]);
    let lists = [a, b, c];
    partition(&mut solver, &lists, &[1, 2]);
    item_precedence(&mut solver, &lists, 1, 2);

    assert_eq!(solver.store.require_list_item(b, 2), Ok(true));
    solver.propagate().unwrap();

    assert!(!solver.store.list_possible(c, 1));
    assert!(solver.store.list_possible(a, 1));
    assert!(solver.store.list_possible(b, 1));
}

#[test]
fn domain_item_precedence_rejects_reversed_required_owners() {
    let mut solver = Solver::new();
    let a = solver.store.new_list(vec![1, 2]);
    let b = solver.store.new_list(vec![1, 2]);
    let c = solver.store.new_list(vec![1, 2]);
    let lists = [a, b, c];
    partition(&mut solver, &lists, &[1, 2]);
    item_precedence(&mut solver, &lists, 1, 2);

    assert_eq!(solver.store.require_list_item(c, 1), Ok(true));
    assert_eq!(solver.store.require_list_item(a, 2), Ok(true));

    assert!(solver.propagate().is_err());
}

#[test]
fn domain_list_len_bounds_close_membership() {
    let mut solver = Solver::new();
    let list = solver.store.new_list(vec![1, 2]);
    list_cardinality(&mut solver, list);
    list_len(&mut solver, list, 1, 1);

    solver.propagate().unwrap();
    assert_eq!((solver.store.list_len_min(list), solver.store.list_len_max(list)), (1, 1));

    assert_eq!(solver.store.require_list_item(list, 1), Ok(true));
    solver.propagate().unwrap();

    assert!(solver.store.list_required(list, 1));
    assert!(!solver.store.list_possible(list, 2));
}

#[test]
fn domain_item_sum_upper_bound_prunes_optional_items() {
    let mut solver = Solver::new();
    let list = solver.store.new_list(vec![1, 2]);
    list_item_sum(&mut solver, list, vec![(1, 5), (2, 7)], i64::MIN / 4, 6);

    assert_eq!(solver.store.require_list_item(list, 1), Ok(true));
    solver.propagate().unwrap();

    assert!(solver.store.list_required(list, 1));
    assert!(!solver.store.list_possible(list, 2));
}

#[test]
fn domain_item_sum_lower_bound_requires_items() {
    let mut solver = Solver::new();
    let list = solver.store.new_list(vec![1, 2]);
    list_item_sum(&mut solver, list, vec![(1, 5), (2, 7)], 8, i64::MAX / 4);

    solver.propagate().unwrap();

    assert!(solver.store.list_required(list, 1));
    assert!(solver.store.list_required(list, 2));
}

#[test]
fn domain_search_respects_item_sum_capacity() {
    let mut solver = Solver::new();
    let a = solver.store.new_list(vec![1, 2, 3]);
    let b = solver.store.new_list(vec![1, 2, 3]);
    let weights = vec![(1, 6), (2, 6), (3, 6)];
    partition(&mut solver, &[a, b], &[1, 2, 3]);
    list_item_sum(&mut solver, a, weights.clone(), i64::MIN / 4, 10);
    list_item_sum(&mut solver, b, weights, i64::MIN / 4, 10);

    let stats = solve_domains(&mut solver, |_, _| SearchControl::Continue);

    assert_eq!(stats.solutions, 0);
}

#[test]
fn domain_search_respects_same_list() {
    let mut solver = Solver::new();
    let a = solver.store.new_list(vec![1, 2]);
    let b = solver.store.new_list(vec![1, 2]);
    partition(&mut solver, &[a, b], &[1, 2]);
    same_list(&mut solver, &[a, b], 1, 2);

    let stats = solve_domains(&mut solver, |_, _| SearchControl::Continue);

    assert_eq!(stats.solutions, 2);
}

#[test]
fn domain_interval_precedence_prunes_present_bounds() {
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 20, 5);
    let b = solver.store.new_interval(0, 12, 3);
    interval_precedence(&mut solver, a, b);

    solver.propagate().unwrap();

    assert_eq!(solver.store.interval_start_max(a), 7);
    assert_eq!(solver.store.interval_start_min(b), 5);
}

#[test]
fn domain_interval_precedence_rejects_present_infeasibility() {
    let mut solver = Solver::new();
    let a = solver.store.new_interval(10, 20, 5);
    let b = solver.store.new_interval(0, 12, 3);
    interval_precedence(&mut solver, a, b);

    assert!(solver.propagate().is_err());
}

#[test]
fn domain_interval_precedence_ignores_absent_interval() {
    let mut solver = Solver::new();
    let a = solver.store.new_optional_interval(10, 20, 5);
    let b = solver.store.new_interval(0, 12, 3);
    interval_precedence(&mut solver, a, b);

    assert_eq!(solver.store.forbid_interval_presence(a), Ok(true));
    solver.propagate().unwrap();

    assert_eq!(solver.store.interval_presence(a), IntervalPresence::Absent);
    assert_eq!((solver.store.interval_start_min(b), solver.store.interval_start_max(b)), (0, 12));
}

#[test]
fn domain_interval_precedence_forbids_optional_side_when_impossible() {
    let mut solver = Solver::new();
    let a = solver.store.new_interval(10, 20, 5);
    let b = solver.store.new_optional_interval(0, 12, 3);
    interval_precedence(&mut solver, a, b);

    solver.propagate().unwrap();

    assert_eq!(solver.store.interval_presence(b), IntervalPresence::Absent);
}

#[test]
fn domain_interval_precedence_keeps_both_optional_when_only_pair_is_impossible() {
    let mut solver = Solver::new();
    let a = solver.store.new_optional_interval(10, 20, 5);
    let b = solver.store.new_optional_interval(0, 12, 3);
    interval_precedence(&mut solver, a, b);

    solver.propagate().unwrap();

    assert_eq!(solver.store.interval_presence(a), IntervalPresence::Optional);
    assert_eq!(solver.store.interval_presence(b), IntervalPresence::Optional);
}

#[test]
fn domain_interval_events_wake_precedence() {
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 20, 5);
    let b = solver.store.new_interval(0, 20, 3);
    let prop = interval_precedence(&mut solver, a, b);

    solver.propagate().unwrap();
    assert!(solver.fd_at_fixpoint());

    assert_eq!(solver.store.set_interval_start_max(b, 12), Ok(true));
    assert_eq!(solver.peek_prop(), Some(prop));
    solver.propagate().unwrap();

    assert_eq!(solver.store.interval_start_max(a), 7);
}

#[test]
fn domain_search_counts_partition_assignments() {
    let mut solver = Solver::new();
    let a = solver.store.new_list(vec![21]);
    let b = solver.store.new_list(vec![21]);
    partition(&mut solver, &[a, b], &[21]);

    let stats = solve_domains(&mut solver, |_, _| SearchControl::Continue);

    assert_eq!(stats.solutions, 2);
}

#[test]
fn domain_search_finds_balanced_partition() {
    let mut solver = Solver::new();
    let a = solver.store.new_list(vec![1, 2]);
    let b = solver.store.new_list(vec![1, 2]);
    list_cardinality(&mut solver, a);
    list_cardinality(&mut solver, b);
    assert_eq!(solver.store.set_list_len_max(a, 1), Ok(true));
    assert_eq!(solver.store.set_list_len_max(b, 1), Ok(true));
    partition(&mut solver, &[a, b], &[1, 2]);

    let solution = first_domain_solution(&mut solver).unwrap();

    assert_eq!(solution.lists, vec![vec![1], vec![2]]);
}

#[test]
fn domain_search_enumerates_interval_starts() {
    let mut solver = Solver::new();
    solver.store.new_interval(0, 2, 1);

    let mut starts = Vec::new();
    let stats = solve_domains(&mut solver, |_, solution| {
        starts.push(solution.interval_starts[0]);
        SearchControl::Continue
    });

    assert_eq!(stats.solutions, 3);
    assert_eq!(starts, vec![Some(0), Some(1), Some(2)]);
}

#[test]
fn domain_search_branches_optional_interval_presence() {
    let mut solver = Solver::new();
    solver.store.new_optional_interval(0, 1, 1);

    let mut starts = Vec::new();
    let stats = solve_domains(&mut solver, |_, solution| {
        starts.push(solution.interval_starts[0]);
        SearchControl::Continue
    });

    assert_eq!(stats.solutions, 3);
    assert_eq!(starts, vec![Some(0), Some(1), None]);
}

#[test]
fn domain_search_respects_interval_precedence() {
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 5, 3);
    let b = solver.store.new_interval(0, 5, 2);
    interval_precedence(&mut solver, a, b);

    let solution = first_domain_solution(&mut solver).unwrap();
    let a_start = solution.interval_starts[a.index()].unwrap();
    let b_start = solution.interval_starts[b.index()].unwrap();

    assert!(a_start + 3 <= b_start);
}

#[test]
fn domain_no_overlap_forces_only_feasible_order() {
    // a is pinned early (start 0, dur 5 -> end 5); b cannot start before 5, so
    // a-before-b is the only feasible order and b.start_min is lifted to 5.
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 0, 5);
    let b = solver.store.new_interval(0, 10, 5);
    no_overlap(&mut solver, &[a, b]);
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_start_min(b), 5);
    // Idempotent: a second run changes nothing.
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_start_min(b), 5);
}

#[test]
fn domain_no_overlap_rejects_unschedulable_present_pair() {
    // Two present intervals both pinned to [0,0] with duration 10 cannot be
    // ordered either way -> inconsistent.
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 0, 10);
    let b = solver.store.new_interval(0, 0, 10);
    no_overlap(&mut solver, &[a, b]);
    assert!(solver.propagate().is_err());
}

#[test]
fn domain_no_overlap_forbids_unschedulable_optional() {
    // Present a blocks the whole window; optional b cannot fit either order, so
    // b is forced absent instead of failing.
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 0, 10);
    let b = solver.store.new_optional_interval(0, 0, 10);
    no_overlap(&mut solver, &[a, b]);
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_presence(b), IntervalPresence::Absent);
}

#[test]
fn domain_no_overlap_propagation_is_trailed() {
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 0, 5);
    let b = solver.store.new_interval(0, 10, 5);
    no_overlap(&mut solver, &[a, b]);
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_start_min(b), 5);

    solver.store.push_level();
    // Pin b late; a-after-b becomes infeasible but a-before-b still holds.
    solver.store.set_interval_start_min(b, 8).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_start_min(b), 8);
    solver.store.pop_level();

    // Back to the pre-decision propagated state.
    assert_eq!(solver.store.interval_start_min(b), 5);
}

#[test]
fn domain_cumulative_detects_overload() {
    // Two interruptible tasks both pinned to [0,3) each demanding 2 of a
    // capacity-3 resource -> 4 > 3 over [0,3) -> infeasible.
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 0, 3);
    let b = solver.store.new_interval(0, 0, 3);
    cumulative(&mut solver, &[a, b], &[2, 2], 3);
    assert!(solver.propagate().is_err());
}

#[test]
fn domain_cumulative_pushes_start_past_mandatory_part() {
    // a is pinned at [0,2) using 2/3; b (demand 2) cannot run during a's
    // mandatory part, so b.start is pushed to 2.
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 0, 2);
    let b = solver.store.new_interval(0, 4, 2);
    cumulative(&mut solver, &[a, b], &[2, 2], 3);
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_start_min(b), 2);
    // Idempotent.
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_start_min(b), 2);
}

#[test]
fn domain_cumulative_forbids_unfittable_optional() {
    // a fills the resource over [0,2); optional b cannot fit anywhere -> absent.
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 0, 2);
    let b = solver.store.new_optional_interval(0, 0, 2);
    cumulative(&mut solver, &[a, b], &[2, 2], 3);
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_presence(b), IntervalPresence::Absent);
}

#[test]
fn domain_cumulative_does_not_filter_an_undecided_optional_start() {
    // The compulsory part of `a` excludes starts 0 and 1 for `b` if `b` is
    // present.  While presence is undecided that inference is conditional: an
    // absent interval's backing start remains unconstrained by cumulative.
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 0, 2);
    let b = solver.store.new_optional_interval(0, 4, 2);
    cumulative(&mut solver, &[a, b], &[2, 2], 3);

    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_presence(b), IntervalPresence::Optional);
    assert_eq!((solver.store.interval_start_min(b), solver.store.interval_start_max(b)), (0, 4));

    solver.store.push_level();
    solver.store.forbid_interval_presence(b).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_presence(b), IntervalPresence::Absent);
    assert_eq!((solver.store.interval_start_min(b), solver.store.interval_start_max(b)), (0, 4));
    solver.store.pop_level();

    solver.store.require_interval_presence(b).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_start_min(b), 2);
}

#[test]
fn domain_cumulative_keeps_a_task_ending_after_i32_max_feasible() {
    // Endpoints are mathematical i128 values inside cumulative. Saturating
    // `i32::MAX + duration` to `i32::MAX` creates a zero-width energetic
    // window and used to report this single feasible task as UNSAT.
    let mut solver = Solver::new();
    let task = solver.store.new_interval(i32::MAX, i32::MAX, 1);
    cumulative(&mut solver, &[task], &[1], 1);

    solver.propagate().expect("a single task at i32::MAX is feasible");
    assert_eq!(solver.store.interval_start_min(task), i32::MAX);
}

#[test]
fn fixed_and_interval_cumulative_match_an_exhaustive_oracle() {
    use std::collections::BTreeSet;

    for capacity in 1..=3i32 {
        for duration_mask in 0..8u32 {
            for demand_mask in 0..8u32 {
                let durations = (0..3).map(|task| 1 + ((duration_mask >> task) & 1) as i32).collect::<Vec<_>>();
                let demands = (0..3).map(|task| 1 + ((demand_mask >> task) & 1) as i32).collect::<Vec<_>>();

                let mut interval_solver = Solver::new();
                let intervals = durations.iter().map(|&duration| interval_solver.store.new_interval(0, 2, duration)).collect::<Vec<_>>();
                cumulative(&mut interval_solver, &intervals, &demands, capacity);
                let mut interval_solutions = BTreeSet::new();
                solve_domains(&mut interval_solver, |_, solution| {
                    interval_solutions.insert(
                        intervals
                            .iter()
                            .map(|interval| solution.interval_starts[interval.index()].expect("mandatory interval"))
                            .collect::<Vec<_>>(),
                    );
                    SearchControl::Continue
                });

                let mut integer_solver = Solver::new();
                let starts = (0..3).map(|_| integer_solver.new_var_range(0, 2)).collect::<Vec<_>>();
                cumulative_integer(
                    &mut integer_solver,
                    &starts,
                    &durations.iter().map(|&value| i64::from(value)).collect::<Vec<_>>(),
                    &demands.iter().map(|&value| i64::from(value)).collect::<Vec<_>>(),
                    i64::from(capacity),
                );
                let mut integer_solutions = BTreeSet::new();
                solve_search(&mut integer_solver, &starts, |solver| {
                    integer_solutions.insert(starts.iter().map(|&start| solver.store.value(start)).collect::<Vec<_>>());
                    SearchControl::Continue
                });

                let mut oracle = BTreeSet::new();
                for first in 0..=2 {
                    for second in 0..=2 {
                        for third in 0..=2 {
                            let assignment = [first, second, third];
                            let horizon = assignment.iter().enumerate().map(|(task, &start)| start + durations[task]).max().unwrap();
                            let feasible = (0..horizon).all(|time| {
                                (0..3)
                                    .filter(|&task| assignment[task] <= time && time < assignment[task] + durations[task])
                                    .map(|task| demands[task])
                                    .sum::<i32>()
                                    <= capacity
                            });
                            if feasible {
                                oracle.insert(assignment.to_vec());
                            }
                        }
                    }
                }

                assert_eq!(interval_solutions, oracle, "interval adapter: cap={capacity}, dur={durations:?}, demand={demands:?}");
                assert_eq!(integer_solutions, oracle, "integer adapter: cap={capacity}, dur={durations:?}, demand={demands:?}");
            }
        }
    }
}

#[test]
fn fixed_and_interval_cumulative_agree_on_zero_usage_tasks() {
    use std::collections::BTreeSet;

    let durations = [0, 2, 2];
    let demands = [3, 0, 2];
    let expected = (-1..=1).flat_map(|first| (-1..=1).map(move |second| vec![first, second, 0])).collect::<BTreeSet<_>>();

    let mut interval_solver = Solver::new();
    let intervals = [
        interval_solver.store.new_interval(-1, 1, durations[0]),
        interval_solver.store.new_interval(-1, 1, durations[1]),
        interval_solver.store.new_interval(0, 0, durations[2]),
    ];
    cumulative(&mut interval_solver, &intervals, &demands, 2);
    let mut interval_solutions = BTreeSet::new();
    solve_domains(&mut interval_solver, |_, solution| {
        interval_solutions.insert(
            intervals.iter().map(|interval| solution.interval_starts[interval.index()].expect("mandatory interval")).collect::<Vec<_>>(),
        );
        SearchControl::Continue
    });

    let mut integer_solver = Solver::new();
    let starts = [integer_solver.new_var_range(-1, 1), integer_solver.new_var_range(-1, 1), integer_solver.new_var_range(0, 0)];
    cumulative_integer(&mut integer_solver, &starts, &[0, 2, 2], &[3, 0, 2], 2);
    let mut integer_solutions = BTreeSet::new();
    solve_search(&mut integer_solver, &starts, |solver| {
        integer_solutions.insert(starts.iter().map(|&start| solver.store.value(start)).collect::<Vec<_>>());
        SearchControl::Continue
    });

    assert_eq!(interval_solutions, expected);
    assert_eq!(integer_solutions, expected);
}

#[test]
fn fixed_and_interval_cumulative_agree_on_negative_touching_windows() {
    use std::collections::BTreeSet;

    // The first task may start at -2 or -1. Only -2 ends exactly where the
    // second starts, so half-open intervals make that touching schedule valid.
    let expected = BTreeSet::from([vec![-2, 0]]);

    let mut interval_solver = Solver::new();
    let first = interval_solver.store.new_interval(-2, -1, 2);
    let second = interval_solver.store.new_interval(0, 0, 2);
    cumulative(&mut interval_solver, &[first, second], &[2, 2], 2);
    let mut interval_solutions = BTreeSet::new();
    solve_domains(&mut interval_solver, |_, solution| {
        interval_solutions.insert(vec![
            solution.interval_starts[first.index()].expect("mandatory interval"),
            solution.interval_starts[second.index()].expect("mandatory interval"),
        ]);
        SearchControl::Continue
    });

    let mut integer_solver = Solver::new();
    let first = integer_solver.new_var_range(-2, -1);
    let second = integer_solver.new_var_range(0, 0);
    cumulative_integer(&mut integer_solver, &[first, second], &[2, 2], &[2, 2], 2);
    let mut integer_solutions = BTreeSet::new();
    solve_search(&mut integer_solver, &[first, second], |solver| {
        integer_solutions.insert(vec![solver.store.value(first), solver.store.value(second)]);
        SearchControl::Continue
    });

    assert_eq!(interval_solutions, expected);
    assert_eq!(integer_solutions, expected);
}

#[test]
fn domain_cumulative_propagation_is_trailed() {
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 0, 2);
    let b = solver.store.new_interval(0, 4, 2);
    cumulative(&mut solver, &[a, b], &[2, 2], 3);
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_start_min(b), 2);

    solver.store.push_level();
    solver.store.set_interval_start_min(b, 3).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_start_min(b), 3);
    solver.store.pop_level();

    assert_eq!(solver.store.interval_start_min(b), 2);
}

#[test]
fn cumulative_adapters_honor_a_prearmed_stop_without_a_false_conflict() {
    let mut interval_solver = Solver::new();
    let interval_a = interval_solver.store.new_interval(0, 0, 2);
    let interval_b = interval_solver.store.new_interval(0, 0, 2);
    cumulative(&mut interval_solver, &[interval_a, interval_b], &[1, 1], 1);
    interval_solver.propagate_until(|| true).expect("a prearmed stop is not an interval cumulative conflict");
    assert!(interval_solver.fd_at_fixpoint());
    interval_solver.enqueue_all();
    assert!(interval_solver.propagate().is_err(), "the real overload remains observable on a complete retry");

    let mut integer_solver = Solver::new();
    let integer_a = integer_solver.new_var_range(0, 0);
    let integer_b = integer_solver.new_var_range(0, 0);
    cumulative_integer(&mut integer_solver, &[integer_a, integer_b], &[2, 2], &[1, 1], 1);
    integer_solver.propagate_until(|| true).expect("a prearmed stop is not an integer cumulative conflict");
    assert!(integer_solver.fd_at_fixpoint());
    integer_solver.enqueue_all();
    assert!(integer_solver.propagate().is_err(), "the real overload remains observable on a complete retry");
}

#[test]
fn interval_cumulative_interruption_preserves_trail_and_retry_state() {
    use std::cell::Cell;

    let mut solver = Solver::new();
    let blocker = solver.store.new_interval(0, 0, 10);
    let target = solver.store.new_interval(0, 20, 10);
    let mut intervals = vec![blocker, target];
    intervals.extend((0..128).map(|index| solver.store.new_interval(100 + 2 * index, 100 + 2 * index, 1)));
    let mut demands = vec![2, 2];
    demands.extend([1; 128]);
    cumulative(&mut solver, &intervals, &demands, 3);

    solver.store.push_level();
    solver.store.set_interval_start_min(target, 1).unwrap();
    let polls = Cell::new(0usize);
    solver
        .propagate_until(|| {
            let next = polls.get() + 1;
            polls.set(next);
            next >= 8
        })
        .expect("cancellation inside interval energetic reasoning is not an inconsistency");
    assert!((8..=9).contains(&polls.get()), "the stop must be observed inside the bounded energetic poll interval");
    assert_eq!(solver.store.interval_start_min(target), 1, "an incomplete energetic result must not be applied");
    assert!(solver.fd_at_fixpoint());

    solver.store.pop_level();
    assert_eq!(solver.store.interval_start_min(target), 0);
    solver.enqueue_all();
    solver.propagate().expect("the interrupted propagator must remain reusable after rollback");
    assert_eq!(solver.store.interval_start_min(target), 10);
}

#[test]
fn integer_cumulative_interruption_preserves_trail_and_retry_state() {
    use std::cell::Cell;

    let mut solver = Solver::new();
    let blocker = solver.new_var_range(0, 0);
    let target = solver.new_var_range(0, 20);
    let mut starts = vec![blocker, target];
    starts.extend((0..128).map(|index| solver.new_var_range(100 + 2 * index, 100 + 2 * index)));
    let mut durations = vec![10, 10];
    durations.extend([1; 128]);
    let mut heights = vec![2, 2];
    heights.extend([1; 128]);
    cumulative_integer(&mut solver, &starts, &durations, &heights, 3);

    solver.store.push_level();
    solver.store.remove_below(target, 1).unwrap();
    let polls = Cell::new(0usize);
    solver
        .propagate_until(|| {
            let next = polls.get() + 1;
            polls.set(next);
            next >= 8
        })
        .expect("cancellation inside integer energetic reasoning is not an inconsistency");
    assert!((8..=9).contains(&polls.get()), "the stop must be observed inside the bounded energetic poll interval");
    assert_eq!(solver.store.min(target), 1, "an incomplete energetic result must not be applied");
    assert!(solver.fd_at_fixpoint());

    solver.store.pop_level();
    assert_eq!(solver.store.min(target), 0);
    solver.enqueue_all();
    solver.propagate().expect("the interrupted propagator must remain reusable after rollback");
    assert_eq!(solver.store.min(target), 10);
}

#[test]
fn domain_exactly_one_mode_requires_last_candidate() {
    let mut solver = Solver::new();
    let m: Vec<_> = (0..3).map(|_| solver.store.new_optional_interval(0, 5, 2)).collect();
    exactly_one_mode(&mut solver, &m);
    solver.store.forbid_interval_presence(m[0]).unwrap();
    solver.store.forbid_interval_presence(m[1]).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_presence(m[2]), IntervalPresence::Present);
}

#[test]
fn domain_exactly_one_mode_forbids_others_when_one_present() {
    let mut solver = Solver::new();
    let m: Vec<_> = (0..3).map(|_| solver.store.new_optional_interval(0, 5, 2)).collect();
    exactly_one_mode(&mut solver, &m);
    solver.store.require_interval_presence(m[0]).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_presence(m[1]), IntervalPresence::Absent);
    assert_eq!(solver.store.interval_presence(m[2]), IntervalPresence::Absent);
}

#[test]
fn domain_exactly_one_mode_fails_when_none_possible() {
    let mut solver = Solver::new();
    let m: Vec<_> = (0..3).map(|_| solver.store.new_optional_interval(0, 5, 2)).collect();
    exactly_one_mode(&mut solver, &m);
    for &interval in &m {
        solver.store.forbid_interval_presence(interval).unwrap();
    }
    assert!(solver.propagate().is_err());
}

#[test]
fn domain_exactly_one_mode_fails_when_two_present() {
    let mut solver = Solver::new();
    let m: Vec<_> = (0..3).map(|_| solver.store.new_optional_interval(0, 5, 2)).collect();
    exactly_one_mode(&mut solver, &m);
    solver.store.require_interval_presence(m[0]).unwrap();
    solver.store.require_interval_presence(m[1]).unwrap();
    assert!(solver.propagate().is_err());
}

#[test]
fn domain_fjsp_reaches_optimum() {
    // 2 jobs x 2 ops, 2 machines. op = job*2 + k; job order 0->1, 2->3.
    // proc[op][machine]. Each op becomes one optional mode-interval per machine;
    // exactly_one_mode picks one, per-machine no_overlap over the mode-intervals,
    // precedence posted over every mode pair, makespan over all modes.
    let proc = [[2i32, 3], [3, 2], [2, 2], [4, 1]];
    let horizon = 12i32;
    let mut solver = Solver::new();

    let mut mode = Vec::new(); // mode[op] = Vec<(machine, IntervalId, duration)>
    for row in &proc {
        let mut ms = Vec::new();
        for (m, &dur) in row.iter().enumerate() {
            ms.push((m, solver.store.new_optional_interval(0, horizon - dur, dur), dur));
        }
        let ids: Vec<_> = ms.iter().map(|&(_, id, _)| id).collect();
        exactly_one_mode(&mut solver, &ids);
        mode.push(ms);
    }
    for machine in 0..2 {
        let group: Vec<_> = mode.iter().flatten().filter(|&&(m, _, _)| m == machine).map(|&(_, id, _)| id).collect();
        no_overlap(&mut solver, &group);
    }
    for &(a, b) in &[(0usize, 1usize), (2, 3)] {
        for &(_, ia, _) in &mode[a] {
            for &(_, ib, _) in &mode[b] {
                interval_precedence(&mut solver, ia, ib);
            }
        }
    }
    let all: Vec<_> = mode.iter().flatten().map(|&(_, id, _)| id).collect();
    let all_dur: Vec<i32> = mode.iter().flatten().map(|&(_, _, d)| d).collect();
    let ub = std::sync::Arc::new(std::sync::atomic::AtomicI32::new(i32::MAX));
    makespan_bound(&mut solver, &all, &all_dur, std::sync::Arc::clone(&ub));

    let id_dur: Vec<(usize, i32)> = mode.iter().flatten().map(|&(_, id, d)| (id.index(), d)).collect();
    let cb_ub = std::sync::Arc::clone(&ub);
    let mut best = i32::MAX;
    solve_domains(&mut solver, |_, sol| {
        let makespan = id_dur.iter().filter_map(|&(idx, d)| sol.interval_starts[idx].map(|s| s + d)).max().unwrap_or(0);
        if makespan < best {
            best = makespan;
            cb_ub.store(makespan - 1, std::sync::atomic::Ordering::Relaxed);
        }
        SearchControl::Continue
    });

    // Brute-force optimum: machine choice per op x integer starts.
    let mut bf = i32::MAX;
    for bits in 0..16usize {
        let mc = [bits & 1, (bits >> 1) & 1, (bits >> 2) & 1, (bits >> 3) & 1];
        let dur = [proc[0][mc[0]], proc[1][mc[1]], proc[2][mc[2]], proc[3][mc[3]]];
        for s0 in 0..=horizon {
            for s1 in 0..=horizon {
                if s0 + dur[0] > s1 {
                    continue;
                }
                for s2 in 0..=horizon {
                    for s3 in 0..=horizon {
                        if s2 + dur[2] > s3 {
                            continue;
                        }
                        let s = [s0, s1, s2, s3];
                        let end = [s0 + dur[0], s1 + dur[1], s2 + dur[2], s3 + dur[3]];
                        if end.iter().any(|&e| e > horizon) {
                            continue;
                        }
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
                            bf = bf.min(*end.iter().max().unwrap());
                        }
                    }
                }
            }
        }
    }
    assert_eq!(best, bf, "FJSP optimum matches brute force");
}

#[test]
fn domain_no_overlap_enumerates_same_solution_set() {
    // The disjunctive order decisions must not change the solution set: a
    // no_overlap over three intervals must enumerate exactly the brute-force set
    // of non-overlapping schedules, with no missing (complete) and no duplicate
    // (clean partition) solutions.
    let durations = [2i32, 1, 2];
    let horizon = 5i32;
    let mut solver = Solver::new();
    let ivs: Vec<_> = durations.iter().map(|&d| solver.store.new_interval(0, horizon - d, d)).collect();
    no_overlap(&mut solver, &ivs);

    let mut count = 0usize;
    let mut enumerated = std::collections::BTreeSet::new();
    solve_domains(&mut solver, |_, sol| {
        let starts: Vec<i32> = sol.interval_starts.iter().map(|s| s.unwrap()).collect();
        enumerated.insert(starts);
        count += 1;
        SearchControl::Continue
    });

    let mut brute = std::collections::BTreeSet::new();
    for s0 in 0..=horizon - durations[0] {
        for s1 in 0..=horizon - durations[1] {
            for s2 in 0..=horizon - durations[2] {
                let s = [s0, s1, s2];
                let e = [s0 + durations[0], s1 + durations[1], s2 + durations[2]];
                let mut ok = true;
                'pairs: for i in 0..3 {
                    for j in (i + 1)..3 {
                        if s[i].max(s[j]) < e[i].min(e[j]) {
                            ok = false;
                            break 'pairs;
                        }
                    }
                }
                if ok {
                    brute.insert(vec![s0, s1, s2]);
                }
            }
        }
    }

    assert_eq!(enumerated, brute, "disjunctive branching enumerates exactly the no-overlap solution set");
    assert_eq!(count, brute.len(), "no duplicate or missing solutions");
}

#[test]
fn domain_no_overlap_zero_duration_no_duplicates() {
    // A zero-duration interval never strictly overlaps, so it imposes no order.
    // Enumeration must still be duplicate-free and match brute force (regression
    // for auxiliary order branching duplicating solutions on zero durations).
    let durations = [2i32, 0, 2];
    let horizon = 4i32;
    let mut solver = Solver::new();
    let ivs: Vec<_> = durations.iter().map(|&d| solver.store.new_interval(0, horizon - d, d)).collect();
    no_overlap(&mut solver, &ivs);

    let mut count = 0usize;
    let mut enumerated = std::collections::BTreeSet::new();
    solve_domains(&mut solver, |_, sol| {
        let starts: Vec<i32> = sol.interval_starts.iter().map(|s| s.unwrap()).collect();
        enumerated.insert(starts);
        count += 1;
        SearchControl::Continue
    });

    let mut brute = std::collections::BTreeSet::new();
    for s0 in 0..=horizon - durations[0] {
        for s1 in 0..=horizon - durations[1] {
            for s2 in 0..=horizon - durations[2] {
                let s = [s0, s1, s2];
                let e = [s0 + durations[0], s1 + durations[1], s2 + durations[2]];
                let mut ok = true;
                'pairs: for i in 0..3 {
                    for j in (i + 1)..3 {
                        if s[i].max(s[j]) < e[i].min(e[j]) {
                            ok = false;
                            break 'pairs;
                        }
                    }
                }
                if ok {
                    brute.insert(vec![s0, s1, s2]);
                }
            }
        }
    }

    assert_eq!(enumerated, brute, "zero-duration enumeration matches brute force");
    assert_eq!(count, brute.len(), "no duplicate solutions from zero-duration pairs");
}

#[test]
fn domain_interval_precedence_runs_under_cdcl() {
    // A precedence chain solved by the integer CDCL engine over the intervals'
    // start variables (IntervalPrecedence is now an explained integer
    // propagator). The makespan optimum must equal the critical path and agree
    // with the chronological domain search.
    let durations = [2i32, 3, 4];
    let horizon = 20i32;

    // CDCL backend: minimise a makespan var >= each interval's end.
    let cdcl = {
        let mut solver = Solver::new();
        let ivs: Vec<_> = durations.iter().map(|&d| solver.store.new_interval(0, horizon - d, d)).collect();
        interval_precedence(&mut solver, ivs[0], ivs[1]);
        interval_precedence(&mut solver, ivs[1], ivs[2]);
        let makespan = solver.store.new_var_range(0, horizon);
        let mut search_vars: Vec<_> = ivs.iter().map(|&iv| solver.store.interval_start_var(iv)).collect();
        for (i, &iv) in ivs.iter().enumerate() {
            let start = solver.store.interval_start_var(iv);
            qayd::constraints::intension::intension(
                &mut solver,
                qayd::expr::ge(
                    qayd::expr::var(makespan),
                    qayd::expr::add(vec![qayd::expr::var(start), qayd::expr::int(i64::from(durations[i]))]),
                ),
            );
        }
        search_vars.push(makespan);
        qayd::search::minimize(&mut solver, &search_vars, makespan).expect("feasible").1
    };

    // Structured chronological search with makespan branch-and-bound.
    let chrono = {
        let mut solver = Solver::new();
        let ivs: Vec<_> = durations.iter().map(|&d| solver.store.new_interval(0, horizon - d, d)).collect();
        interval_precedence(&mut solver, ivs[0], ivs[1]);
        interval_precedence(&mut solver, ivs[1], ivs[2]);
        let ub = std::sync::Arc::new(std::sync::atomic::AtomicI32::new(i32::MAX));
        makespan_bound(&mut solver, &ivs, &durations, std::sync::Arc::clone(&ub));
        let id_dur: Vec<(usize, i32)> = ivs.iter().enumerate().map(|(i, &iv)| (iv.index(), durations[i])).collect();
        let cb = std::sync::Arc::clone(&ub);
        let mut best = i32::MAX;
        solve_domains(&mut solver, |_, sol| {
            let mk = id_dur.iter().filter_map(|&(idx, d)| sol.interval_starts[idx].map(|s| s + d)).max().unwrap_or(0);
            if mk < best {
                best = mk;
                cb.store(mk - 1, std::sync::atomic::Ordering::Relaxed);
            }
            SearchControl::Continue
        });
        best
    };

    assert_eq!(cdcl, 9, "chain critical path = 2 + 3 + 4");
    assert_eq!(cdcl, chrono, "CDCL backend agrees with chronological domain search");
}

#[test]
fn domain_optional_interval_forbidden_under_cdcl() {
    // `a` (mandatory) ends at 5; `a` precedes optional `b`, but `b` cannot start
    // that late, so the precedence forbids `b` (exercising
    // forbid_interval_presence_because and the presence premises). CDCL must find
    // `b` absent and makespan 5.
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 0, 5); // fixed start 0, end 5
    let b = solver.store.new_optional_interval(0, 1, 3); // start_max 1, cannot be >= 5
    interval_precedence(&mut solver, a, b);
    let a_start = solver.store.interval_start_var(a);
    let b_presence = solver.store.interval_presence_var(b).expect("optional interval has a presence variable");
    let makespan = solver.store.new_var_range(0, 20);
    qayd::constraints::intension::intension(
        &mut solver,
        qayd::expr::ge(qayd::expr::var(makespan), qayd::expr::add(vec![qayd::expr::var(a_start), qayd::expr::int(5)])),
    );
    let search_vars = vec![a_start, b_presence, makespan];
    let (assignment, value) = qayd::search::minimize(&mut solver, &search_vars, makespan).expect("feasible");
    assert_eq!(value, 5, "makespan equals a's end");
    assert_eq!(assignment[1], 0, "optional b is forbidden by the precedence");
}

#[test]
fn domain_no_overlap_runs_under_cdcl() {
    // Two mandatory intervals on a unary resource cannot overlap, so the makespan
    // is at least the sum of durations. CDCL searches the start variables AND the
    // disjunctive order variable (the order decision is now first-class), and must
    // match the chronological domain optimum. no_overlap uses sound
    // scope-snapshot reasons here (tight reasons come later).
    let durations = [3i32, 3];
    let horizon = 8i32;

    let cdcl = {
        let mut solver = Solver::new();
        let ivs: Vec<_> = durations.iter().map(|&d| solver.store.new_interval(0, horizon - d, d)).collect();
        no_overlap(&mut solver, &ivs);
        let makespan = solver.store.new_var_range(0, horizon);
        let mut search_vars: Vec<_> = ivs.iter().map(|&iv| solver.store.interval_start_var(iv)).collect();
        search_vars.push(solver.store.disjunctive_order_var(0)); // order decision is in the search
        for (i, &iv) in ivs.iter().enumerate() {
            let start = solver.store.interval_start_var(iv);
            qayd::constraints::intension::intension(
                &mut solver,
                qayd::expr::ge(
                    qayd::expr::var(makespan),
                    qayd::expr::add(vec![qayd::expr::var(start), qayd::expr::int(i64::from(durations[i]))]),
                ),
            );
        }
        search_vars.push(makespan);
        qayd::search::minimize(&mut solver, &search_vars, makespan).expect("feasible").1
    };

    let chrono = {
        let mut solver = Solver::new();
        let ivs: Vec<_> = durations.iter().map(|&d| solver.store.new_interval(0, horizon - d, d)).collect();
        no_overlap(&mut solver, &ivs);
        let ub = std::sync::Arc::new(std::sync::atomic::AtomicI32::new(i32::MAX));
        makespan_bound(&mut solver, &ivs, &durations, std::sync::Arc::clone(&ub));
        let id_dur: Vec<(usize, i32)> = ivs.iter().enumerate().map(|(i, &iv)| (iv.index(), durations[i])).collect();
        let cb = std::sync::Arc::clone(&ub);
        let mut best = i32::MAX;
        solve_domains(&mut solver, |_, sol| {
            let mk = id_dur.iter().filter_map(|&(idx, d)| sol.interval_starts[idx].map(|s| s + d)).max().unwrap_or(0);
            if mk < best {
                best = mk;
                cb.store(mk - 1, std::sync::atomic::Ordering::Relaxed);
            }
            SearchControl::Continue
        });
        best
    };

    assert_eq!(cdcl, 6, "two non-overlapping 3-unit intervals => makespan 6");
    assert_eq!(cdcl, chrono, "CDCL with the disjunctive order matches chronological search");
}

#[test]
fn domain_no_overlap_cdcl_enumerates_same_solution_set() {
    // Enumerate every schedule under the CDCL engine (which feeds no_overlap's
    // tight explanations into conflict analysis) and assert the solution set
    // equals the chronological domain search. A wrong explanation would
    // over-prune and silently drop feasible schedules.
    let durations = [2i32, 1, 2];
    let horizon = 6i32;

    let cdcl_set: std::collections::BTreeSet<Vec<i32>> = {
        let mut solver = Solver::new();
        let ivs: Vec<_> = durations.iter().map(|&d| solver.store.new_interval(0, horizon - d, d)).collect();
        no_overlap(&mut solver, &ivs);
        let start_vars: Vec<_> = ivs.iter().map(|&iv| solver.store.interval_start_var(iv)).collect();
        let mut sols = std::collections::BTreeSet::new();
        qayd::search::solve(&mut solver, &start_vars, |s: &Solver| {
            sols.insert(start_vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
            SearchControl::Continue
        });
        sols
    };

    let chrono_set: std::collections::BTreeSet<Vec<i32>> = {
        let mut solver = Solver::new();
        let ivs: Vec<_> = durations.iter().map(|&d| solver.store.new_interval(0, horizon - d, d)).collect();
        no_overlap(&mut solver, &ivs);
        let ids: Vec<usize> = ivs.iter().map(|iv| iv.index()).collect();
        let mut sols = std::collections::BTreeSet::new();
        solve_domains(&mut solver, |_, sol| {
            sols.insert(ids.iter().map(|&idx| sol.interval_starts[idx].expect("mandatory interval is present")).collect::<Vec<_>>());
            SearchControl::Continue
        });
        sols
    };

    assert!(!cdcl_set.is_empty(), "schedule has feasible solutions");
    assert_eq!(cdcl_set, chrono_set, "CDCL (using tight explanations) enumerates the same schedules as chronological search");
}

#[test]
fn domain_no_overlap_cdcl_optional_same_solution_set() {
    // Oracle with an optional interval: the forbid explanation must keep, not
    // drop, the schedules where the optional is present. Absent-optional start
    // values are canonicalised so the comparison key is
    // (m0 start, m1 start, optional start or -1 when absent).
    let mand = [2i32, 2];
    let opt_dur = 1i32;
    let horizon = 5i32;
    let key = |s0: i32, s1: i32, present: bool, ostart: i32| -> (i32, i32, i32) { (s0, s1, if present { ostart } else { -1 }) };

    let cdcl_set: std::collections::BTreeSet<(i32, i32, i32)> = {
        let mut solver = Solver::new();
        let m0 = solver.store.new_interval(0, horizon - mand[0], mand[0]);
        let m1 = solver.store.new_interval(0, horizon - mand[1], mand[1]);
        let opt = solver.store.new_optional_interval(0, horizon - opt_dur, opt_dur);
        no_overlap(&mut solver, &[m0, m1, opt]);
        let s0 = solver.store.interval_start_var(m0);
        let s1 = solver.store.interval_start_var(m1);
        let os = solver.store.interval_start_var(opt);
        let op = solver.store.interval_presence_var(opt).expect("optional has a presence var");
        let search_vars = vec![s0, s1, op, os];
        let mut sols = std::collections::BTreeSet::new();
        qayd::search::solve(&mut solver, &search_vars, |st: &Solver| {
            let present = st.store.value(op) == 1;
            sols.insert(key(st.store.value(s0), st.store.value(s1), present, st.store.value(os)));
            SearchControl::Continue
        });
        sols
    };

    let chrono_set: std::collections::BTreeSet<(i32, i32, i32)> = {
        let mut solver = Solver::new();
        let m0 = solver.store.new_interval(0, horizon - mand[0], mand[0]);
        let m1 = solver.store.new_interval(0, horizon - mand[1], mand[1]);
        let opt = solver.store.new_optional_interval(0, horizon - opt_dur, opt_dur);
        no_overlap(&mut solver, &[m0, m1, opt]);
        let (i0, i1, io) = (m0.index(), m1.index(), opt.index());
        let mut sols = std::collections::BTreeSet::new();
        solve_domains(&mut solver, |_, sol| {
            let present = sol.interval_starts[io].is_some();
            sols.insert(key(
                sol.interval_starts[i0].unwrap(),
                sol.interval_starts[i1].unwrap(),
                present,
                sol.interval_starts[io].unwrap_or(-1),
            ));
            SearchControl::Continue
        });
        sols
    };

    assert!(!cdcl_set.is_empty(), "schedule has feasible solutions");
    assert!(cdcl_set.iter().any(|&(_, _, o)| o == -1), "some schedules omit the optional");
    assert!(cdcl_set.iter().any(|&(_, _, o)| o != -1), "some schedules include the optional");
    assert_eq!(cdcl_set, chrono_set, "CDCL keeps every optional-present and optional-absent schedule");
}

/// Build a fixed 2-job x 2-machine job-shop in the domain kernel: each job is
/// a chain of operations (interval_precedence), each machine a unary resource
/// (no_overlap). Returns the operation intervals and their durations.
fn build_jssp(solver: &mut Solver, horizon: i32) -> (Vec<IntervalId>, Vec<i32>) {
    // jobs[j] = [(machine, duration), ...] in processing order.
    let jobs: [&[(usize, i32)]; 2] = [&[(0, 2), (1, 3)], &[(1, 2), (0, 1)]];
    let n_machines = 2;
    let mut ivs = Vec::new();
    let mut durs = Vec::new();
    let mut machine_ops: Vec<Vec<IntervalId>> = vec![Vec::new(); n_machines];
    for job in jobs {
        let mut chain = Vec::new();
        for &(m, d) in job {
            let iv = solver.store.new_interval(0, horizon - d, d);
            ivs.push(iv);
            durs.push(d);
            machine_ops[m].push(iv);
            chain.push(iv);
        }
        for w in chain.windows(2) {
            interval_precedence(solver, w[0], w[1]);
        }
    }
    for ops in &machine_ops {
        if ops.len() > 1 {
            no_overlap(solver, ops);
        }
    }
    (ivs, durs)
}

#[test]
fn domain_jssp_cdcl_matches_chronological_optimum() {
    // End-to-end scheduling under CDCL: branch ONLY the disjunctive order
    // variables (and makespan); the operation starts are fixed by propagation
    // along the critical path. The optimum must equal the chronological search.
    let horizon = 8i32;

    let cdcl = {
        let mut solver = Solver::new();
        let (ivs, durs) = build_jssp(&mut solver, horizon);
        let makespan = solver.store.new_var_range(0, horizon);
        for (iv, &d) in ivs.iter().zip(&durs) {
            let start = solver.store.interval_start_var(*iv);
            qayd::constraints::intension::intension(
                &mut solver,
                qayd::expr::ge(qayd::expr::var(makespan), qayd::expr::add(vec![qayd::expr::var(start), qayd::expr::int(i64::from(d))])),
            );
        }
        // schedule_search_vars: order variables first, then makespan. No starts.
        let mut search_vars: Vec<_> = (0..solver.store.disjunctive_pair_count()).map(|i| solver.store.disjunctive_order_var(i)).collect();
        search_vars.push(makespan);
        qayd::search::minimize(&mut solver, &search_vars, makespan).expect("feasible").1
    };

    let chrono = {
        let mut solver = Solver::new();
        let (ivs, durs) = build_jssp(&mut solver, horizon);
        let ub = std::sync::Arc::new(std::sync::atomic::AtomicI32::new(i32::MAX));
        makespan_bound(&mut solver, &ivs, &durs, std::sync::Arc::clone(&ub));
        let id_dur: Vec<(usize, i32)> = ivs.iter().zip(&durs).map(|(iv, &d)| (iv.index(), d)).collect();
        let cb = std::sync::Arc::clone(&ub);
        let mut best = i32::MAX;
        solve_domains(&mut solver, |_, sol| {
            let mk = id_dur.iter().filter_map(|&(idx, d)| sol.interval_starts[idx].map(|s| s + d)).max().unwrap_or(0);
            if mk < best {
                best = mk;
                cb.store(mk - 1, std::sync::atomic::Ordering::Relaxed);
            }
            SearchControl::Continue
        });
        best
    };

    assert!(cdcl > 0, "job-shop has a positive makespan");
    assert_eq!(cdcl, chrono, "CDCL scheduling backend (order-variable search) matches chronological optimum");
}

/// Build the fixed 2-job x 2-op flexible job-shop from the chrono FJSP test: each
/// operation has one optional mode-interval per machine (exactly_one_mode picks
/// one), each machine is a unary resource, precedences over every mode pair.
/// Returns (mode interval, duration) for every mode.
fn build_fjsp(solver: &mut Solver, horizon: i32) -> Vec<(IntervalId, i32)> {
    let proc = [[2i32, 3], [3, 2], [2, 2], [4, 1]];
    let mut mode: Vec<Vec<(usize, IntervalId, i32)>> = Vec::new();
    for row in &proc {
        let mut ms = Vec::new();
        for (m, &dur) in row.iter().enumerate() {
            ms.push((m, solver.store.new_optional_interval(0, horizon - dur, dur), dur));
        }
        let ids: Vec<_> = ms.iter().map(|&(_, id, _)| id).collect();
        exactly_one_mode(solver, &ids);
        mode.push(ms);
    }
    for machine in 0..2 {
        let group: Vec<_> = mode.iter().flatten().filter(|&&(m, _, _)| m == machine).map(|&(_, id, _)| id).collect();
        no_overlap(solver, &group);
    }
    for &(a, b) in &[(0usize, 1usize), (2, 3)] {
        for &(_, ia, _) in &mode[a] {
            for &(_, ib, _) in &mode[b] {
                interval_precedence(solver, ia, ib);
            }
        }
    }
    mode.iter().flatten().map(|&(_, id, d)| (id, d)).collect()
}

#[test]
fn domain_fjsp_cdcl_matches_chronological_optimum() {
    // FJSP under CDCL: branch the mode presences AND the disjunctive order
    // variables; starts follow by propagation. The makespan only counts present
    // modes, so it is a reified bound: present => makespan >= start + duration.
    let horizon = 12i32;

    let cdcl = {
        let mut solver = Solver::new();
        let modes = build_fjsp(&mut solver, horizon);
        let makespan = solver.store.new_var_range(0, horizon);
        for &(iv, d) in &modes {
            let start = solver.store.interval_start_var(iv);
            let present = solver.store.interval_presence_var(iv).expect("mode is optional");
            qayd::constraints::intension::intension(
                &mut solver,
                qayd::expr::imp(
                    qayd::expr::eq(qayd::expr::var(present), qayd::expr::int(1)),
                    qayd::expr::ge(qayd::expr::var(makespan), qayd::expr::add(vec![qayd::expr::var(start), qayd::expr::int(i64::from(d))])),
                ),
            );
        }
        // schedule_search_vars: order variables, then mode presences, then makespan.
        let mut search_vars: Vec<_> = (0..solver.store.disjunctive_pair_count()).map(|i| solver.store.disjunctive_order_var(i)).collect();
        search_vars.extend(modes.iter().map(|&(iv, _)| solver.store.interval_presence_var(iv).expect("mode is optional")));
        search_vars.push(makespan);
        qayd::search::minimize(&mut solver, &search_vars, makespan).expect("feasible").1
    };

    let chrono = {
        let mut solver = Solver::new();
        let modes = build_fjsp(&mut solver, horizon);
        let all: Vec<IntervalId> = modes.iter().map(|&(id, _)| id).collect();
        let all_dur: Vec<i32> = modes.iter().map(|&(_, d)| d).collect();
        let ub = std::sync::Arc::new(std::sync::atomic::AtomicI32::new(i32::MAX));
        makespan_bound(&mut solver, &all, &all_dur, std::sync::Arc::clone(&ub));
        let id_dur: Vec<(usize, i32)> = modes.iter().map(|&(id, d)| (id.index(), d)).collect();
        let cb = std::sync::Arc::clone(&ub);
        let mut best = i32::MAX;
        solve_domains(&mut solver, |_, sol| {
            let mk = id_dur.iter().filter_map(|&(idx, d)| sol.interval_starts[idx].map(|s| s + d)).max().unwrap_or(0);
            if mk < best {
                best = mk;
                cb.store(mk - 1, std::sync::atomic::Ordering::Relaxed);
            }
            SearchControl::Continue
        });
        best
    };

    assert!(cdcl > 0, "flexible job-shop has a positive makespan");
    assert_eq!(cdcl, chrono, "CDCL FJSP backend (presence + order search) matches chronological optimum");
}

#[test]
fn domain_fjsp_fresh_replay_reproduces_optimum() {
    // The root state of a solver AFTER optimize is unspecified for reuse: the
    // B&B may leave a strict `makespan < optimum` bound (or not, depending on
    // propagation order), so incumbent replay is only guaranteed on a fresh
    // solver rebuilt from the same model -- identical ids.
    let horizon = 12i32;
    let mut solver = Solver::new();
    let modes = build_fjsp(&mut solver, horizon);
    let makespan = solver.store.new_var_range(0, horizon);
    let presence_vars: Vec<_> = modes.iter().map(|&(iv, _)| solver.store.interval_presence_var(iv).expect("optional")).collect();
    for &(iv, d) in &modes {
        let start = solver.store.interval_start_var(iv);
        let present = solver.store.interval_presence_var(iv).expect("optional");
        qayd::constraints::intension::intension(
            &mut solver,
            qayd::expr::imp(
                qayd::expr::eq(qayd::expr::var(present), qayd::expr::int(1)),
                qayd::expr::ge(qayd::expr::var(makespan), qayd::expr::add(vec![qayd::expr::var(start), qayd::expr::int(i64::from(d))])),
            ),
        );
    }
    let n_modes = modes.len();
    let n_orders = solver.store.disjunctive_pair_count();
    let mut search_vars: Vec<_> = presence_vars.clone();
    search_vars.extend((0..n_orders).map(|i| solver.store.disjunctive_order_var(i)));
    search_vars.push(makespan);
    let (assignment, value) = qayd::search::minimize(&mut solver, &search_vars, makespan).expect("feasible");

    // Fresh solver replays the incumbent and reproduces the optimum makespan.
    let mut fresh = Solver::new();
    let fresh_modes = build_fjsp(&mut fresh, horizon);
    for (i, &(iv, _)) in fresh_modes.iter().enumerate() {
        let p = fresh.store.interval_presence_var(iv).expect("optional");
        fresh.store.fix(p, assignment[i]).expect("fix presence on fresh solver");
    }
    for k in 0..n_orders {
        let order_var = fresh.store.disjunctive_order_var(k);
        fresh.store.fix(order_var, assignment[n_modes + k]).expect("fix order on fresh solver");
    }
    fresh.propagate().expect("fresh-solver replay propagates");
    let mk = fresh_modes
        .iter()
        .enumerate()
        .filter(|&(i, _)| assignment[i] == 1)
        .map(|(_, &(iv, d))| fresh.store.interval_start_min(iv) + d)
        .max()
        .unwrap_or(0);

    assert_eq!(mk, value, "fresh-solver replay reproduces the optimum makespan");
}

#[test]
fn domain_exactly_one_mode_cdcl_same_solution_set() {
    // Enumerate which mode is present under CDCL (which feeds exactly_one_mode's
    // tight explanations into conflict analysis) and compare to the chronological
    // domain search. A wrong explanation would over-prune and drop a valid choice.
    let k = 3usize;
    let horizon = 5i32;
    let presence_set = |cdcl: bool| -> std::collections::BTreeSet<Vec<bool>> {
        let mut solver = Solver::new();
        let modes: Vec<_> = (0..k).map(|_| solver.store.new_optional_interval(0, horizon - 2, 2)).collect();
        exactly_one_mode(&mut solver, &modes);
        let mut set = std::collections::BTreeSet::new();
        if cdcl {
            let presence: Vec<_> = modes.iter().map(|&m| solver.store.interval_presence_var(m).expect("optional mode")).collect();
            qayd::search::solve(&mut solver, &presence, |s: &Solver| {
                set.insert(presence.iter().map(|&v| s.store.value(v) == 1).collect::<Vec<_>>());
                SearchControl::Continue
            });
        } else {
            let ids: Vec<usize> = modes.iter().map(|m| m.index()).collect();
            solve_domains(&mut solver, |_, dom| {
                set.insert(ids.iter().map(|&i| dom.interval_starts[i].is_some()).collect::<Vec<_>>());
                SearchControl::Continue
            });
        }
        set
    };
    let cdcl = presence_set(true);
    let chrono = presence_set(false);
    assert_eq!(cdcl.len(), k, "exactly K single-mode choices");
    assert!(cdcl.iter().all(|v| v.iter().filter(|&&p| p).count() == 1), "exactly one mode present per solution");
    assert_eq!(cdcl, chrono, "CDCL (tight explanations) enumerates the same mode choices as chronological");
}

#[test]
fn domain_cumulative_cdcl_same_solution_set() {
    // Capacity 1 with unit demands forbids overlap, so the cumulative overload
    // explanation drives the conflicts. Enumerate the schedules under CDCL (using
    // that explanation) and compare to chronological: a too-strong reason would
    // over-prune and drop a feasible schedule.
    let horizon = 4i32;
    let starts_set = |cdcl: bool| -> std::collections::BTreeSet<Vec<i32>> {
        let mut solver = Solver::new();
        let a = solver.store.new_interval(0, horizon - 2, 2);
        let b = solver.store.new_interval(0, horizon - 2, 2);
        cumulative(&mut solver, &[a, b], &[1, 1], 1);
        let mut set = std::collections::BTreeSet::new();
        if cdcl {
            let sv = vec![solver.store.interval_start_var(a), solver.store.interval_start_var(b)];
            qayd::search::solve(&mut solver, &sv, |s: &Solver| {
                set.insert(sv.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
                SearchControl::Continue
            });
        } else {
            let ids = [a.index(), b.index()];
            solve_domains(&mut solver, |_, dom| {
                set.insert(ids.iter().map(|&i| dom.interval_starts[i].expect("present")).collect::<Vec<_>>());
                SearchControl::Continue
            });
        }
        set
    };
    let cdcl = starts_set(true);
    let chrono = starts_set(false);
    assert!(!cdcl.is_empty(), "feasible schedules exist");
    assert_eq!(cdcl, chrono, "CDCL (cumulative overload explanation) enumerates the same schedules as chronological");
}

#[test]
fn domain_cumulative_cdcl_mixed_demands_same_solution_set() {
    // Capacity 2 with demands [1,1,2]: the heavy interval cannot share an instant
    // with either light one, but the two light ones may overlap. This exercises
    // both the overload conflicts and the start-min pushes under CDCL.
    let horizon = 5i32;
    let starts_set = |cdcl: bool| -> std::collections::BTreeSet<Vec<i32>> {
        let mut solver = Solver::new();
        let ivs: Vec<_> = (0..3).map(|_| solver.store.new_interval(0, horizon - 2, 2)).collect();
        cumulative(&mut solver, &ivs, &[1, 1, 2], 2);
        let mut set = std::collections::BTreeSet::new();
        if cdcl {
            let sv: Vec<_> = ivs.iter().map(|&iv| solver.store.interval_start_var(iv)).collect();
            qayd::search::solve(&mut solver, &sv, |s: &Solver| {
                set.insert(sv.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
                SearchControl::Continue
            });
        } else {
            let ids: Vec<usize> = ivs.iter().map(|iv| iv.index()).collect();
            solve_domains(&mut solver, |_, dom| {
                set.insert(ids.iter().map(|&i| dom.interval_starts[i].expect("present")).collect::<Vec<_>>());
                SearchControl::Continue
            });
        }
        set
    };
    let cdcl = starts_set(true);
    let chrono = starts_set(false);
    assert!(!cdcl.is_empty(), "feasible schedules exist");
    assert_eq!(cdcl, chrono, "CDCL (overload + push explanations) enumerates the same schedules as chronological");
}

#[test]
fn domain_cumulative_energetic_scope_cdcl_same_solution_set() {
    // There is no compulsory part at the root: each duration-2 task starts in
    // 0..=3. Energetic conflicts appear only after search narrows windows. The
    // kernel deliberately leaves those conflicts on the propagator-scope
    // explanation until an exact Omega witness is threaded through the Store.
    let schedules = |cdcl: bool| -> std::collections::BTreeSet<Vec<i32>> {
        let mut solver = Solver::new();
        let intervals = (0..4).map(|_| solver.store.new_interval(0, 3, 2)).collect::<Vec<_>>();
        cumulative(&mut solver, &intervals, &[1; 4], 2);
        let mut schedules = std::collections::BTreeSet::new();
        if cdcl {
            let starts = intervals.iter().map(|&interval| solver.store.interval_start_var(interval)).collect::<Vec<_>>();
            solve_search(&mut solver, &starts, |solver| {
                schedules.insert(starts.iter().map(|&start| solver.store.value(start)).collect());
                SearchControl::Continue
            });
        } else {
            solve_domains(&mut solver, |_, solution| {
                schedules.insert(
                    intervals.iter().map(|interval| solution.interval_starts[interval.index()].expect("mandatory interval")).collect(),
                );
                SearchControl::Continue
            });
        }
        schedules
    };

    let cdcl = schedules(true);
    let chronological = schedules(false);
    assert!(!cdcl.is_empty());
    assert_eq!(cdcl, chronological, "energetic scope explanations must preserve every feasible schedule");
}

#[test]
fn domain_detectable_precedences_pushes_past_pairwise() {
    // a, b decided before c on a unary resource. Pairwise no_overlap pushes
    // c.start only to max(end_min(a), end_min(b)) = 2; detectable precedences
    // pushes it to the earliest the whole {a, b} set can finish (ECT = 4).
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 8, 2);
    let b = solver.store.new_interval(0, 8, 2);
    let c = solver.store.new_interval(0, 8, 2);
    no_overlap(&mut solver, &[a, b, c]);
    for k in 0..solver.store.disjunctive_pair_count() {
        let pair = solver.store.disjunctive_pair(k);
        if pair == (a, c) || pair == (b, c) {
            solver.store.set_disjunctive_order(k, 1).unwrap(); // first (a/b) before second (c)
        }
    }
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_start_min(c), 4, "detectable precedences pushes c past the pairwise bound (2) to the set ECT (4)");
}

#[test]
fn domain_detectable_precedences_cdcl_same_solution_set() {
    // Three equal intervals on a unary resource: tight enough that the global
    // detectable-precedence ECT bound fires during search. Enumerate under CDCL
    // and compare to chronological -- the over-pruning guard for its explanation.
    let horizon = 6i32;
    let starts_set = |cdcl: bool| -> std::collections::BTreeSet<Vec<i32>> {
        let mut solver = Solver::new();
        let ivs: Vec<_> = (0..3).map(|_| solver.store.new_interval(0, horizon - 2, 2)).collect();
        no_overlap(&mut solver, &ivs);
        let mut set = std::collections::BTreeSet::new();
        if cdcl {
            let sv: Vec<_> = ivs.iter().map(|&iv| solver.store.interval_start_var(iv)).collect();
            qayd::search::solve(&mut solver, &sv, |s: &Solver| {
                set.insert(sv.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
                SearchControl::Continue
            });
        } else {
            let ids: Vec<usize> = ivs.iter().map(|iv| iv.index()).collect();
            solve_domains(&mut solver, |_, dom| {
                set.insert(ids.iter().map(|&i| dom.interval_starts[i].expect("present")).collect::<Vec<_>>());
                SearchControl::Continue
            });
        }
        set
    };
    let cdcl = starts_set(true);
    let chrono = starts_set(false);
    assert!(!cdcl.is_empty(), "feasible schedules exist");
    assert_eq!(cdcl, chrono, "CDCL enumerates the same schedules as chronological (detectable-precedence guard)");
}

#[test]
fn domain_detectable_precedences_bounds_forced_pushes_past_pairwise() {
    // a, b are forced before c by BOUNDS (each cannot start after c's earliest
    // end), with no decided order. Their set ECT (2) pushes c past pairwise (1).
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 1, 1); // start [0,1], dur 1
    let b = solver.store.new_interval(0, 1, 1);
    let c = solver.store.new_interval(0, 8, 2); // start [0,8], dur 2
    no_overlap(&mut solver, &[a, b, c]);
    solver.propagate().unwrap();
    assert_eq!(solver.store.interval_start_min(c), 2, "bounds-forced detectable precedences push c to ECT (2), past pairwise (1)");
}

#[test]
fn domain_partition_requires_last_list() {
    // Item 10 forbidden from two of three lists: partition must require it in the
    // only remaining list (proves the propagation fires).
    let universe = vec![10, 20, 30];
    let mut solver = Solver::new();
    let lists: Vec<_> = (0..3).map(|_| solver.store.new_list(universe.clone())).collect();
    partition(&mut solver, &lists, &universe);
    solver.store.forbid_list_item(lists[0], 10).unwrap();
    solver.store.forbid_list_item(lists[1], 10).unwrap();
    solver.propagate().unwrap();
    assert!(solver.store.list_required(lists[2], 10), "partition requires item 10 in the only remaining list");
}

#[test]
fn domain_partition_cdcl_same_solution_set() {
    // Three lists partition three items. Enumerate the membership variables under
    // CDCL (reasoning over the partition explanations) and under chronological
    // domain search; the sets must match -- the over-pruning guard for partition.
    let universe = vec![10, 20, 30];
    let members_set = |cdcl: bool| -> std::collections::BTreeSet<Vec<i32>> {
        let mut solver = Solver::new();
        let lists: Vec<_> = (0..3).map(|_| solver.store.new_list(universe.clone())).collect();
        partition(&mut solver, &lists, &universe);
        let mut member_vars = Vec::new();
        let mut keys = Vec::new();
        for (li, &list) in lists.iter().enumerate() {
            for &item in &universe {
                member_vars.push(solver.store.list_member_var(list, item).expect("member var"));
                keys.push((li, item));
            }
        }
        let mut set = std::collections::BTreeSet::new();
        if cdcl {
            qayd::search::solve(&mut solver, &member_vars, |s: &Solver| {
                set.insert(member_vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
                SearchControl::Continue
            });
        } else {
            solve_domains(&mut solver, |_, dom| {
                let assignment = keys.iter().map(|&(li, item)| i32::from(dom.lists[li].contains(&item))).collect::<Vec<_>>();
                set.insert(assignment);
                SearchControl::Continue
            });
        }
        set
    };
    let cdcl = members_set(true);
    let chrono = members_set(false);
    assert_eq!(cdcl.len(), 27, "three items, each in exactly one of three lists");
    assert_eq!(cdcl, chrono, "CDCL enumerates the same partitions as chronological (partition over-pruning guard)");
}

#[test]
fn domain_same_list_cdcl_same_solution_set() {
    // Partition three items across three lists, with same_list(10, 20). Enumerate
    // memberships under CDCL and chronological search: the sets must match, and in
    // every solution 10 and 20 share a list.
    let universe = vec![10, 20, 30];
    let solve_set = |cdcl: bool| -> std::collections::BTreeSet<Vec<i32>> {
        let mut solver = Solver::new();
        let lists: Vec<_> = (0..3).map(|_| solver.store.new_list(universe.clone())).collect();
        partition(&mut solver, &lists, &universe);
        same_list(&mut solver, &lists, 10, 20);
        let mut member_vars = Vec::new();
        let mut keys = Vec::new();
        for (li, &list) in lists.iter().enumerate() {
            for &item in &universe {
                member_vars.push(solver.store.list_member_var(list, item).expect("member var"));
                keys.push((li, item));
            }
        }
        let mut set = std::collections::BTreeSet::new();
        if cdcl {
            qayd::search::solve(&mut solver, &member_vars, |s: &Solver| {
                set.insert(member_vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
                SearchControl::Continue
            });
        } else {
            solve_domains(&mut solver, |_, dom| {
                set.insert(keys.iter().map(|&(li, item)| i32::from(dom.lists[li].contains(&item))).collect::<Vec<_>>());
                SearchControl::Continue
            });
        }
        set
    };
    let cdcl = solve_set(true);
    let chrono = solve_set(false);
    assert_eq!(cdcl.len(), 9, "list of the 10/20 pair (3) times list of 30 (3)");
    assert_eq!(cdcl, chrono, "CDCL enumerates the same assignments as chronological (same_list over-pruning guard)");
    for sol in &cdcl {
        for li in 0..3 {
            assert_eq!(sol[li * 3], sol[li * 3 + 1], "items 10 and 20 always share a list");
        }
    }
}

#[test]
fn domain_list_len_cardinality_cdcl_same_solution_set() {
    // Partition three items across three lists, each list coupled to its length
    // variable, with list 0 pinned to exactly one item. Enumerate memberships
    // under CDCL (reasoning over the ListCardinality explanations) and under
    // chronological domain search: the sets must match -- the over-pruning guard
    // for the length/membership coupling.
    let universe = vec![10, 20, 30];
    let solve_set = |cdcl: bool| -> std::collections::BTreeSet<Vec<i32>> {
        let mut solver = Solver::new();
        let lists: Vec<_> = (0..3).map(|_| solver.store.new_list(universe.clone())).collect();
        for &list in &lists {
            list_cardinality(&mut solver, list);
        }
        partition(&mut solver, &lists, &universe);
        list_len(&mut solver, lists[0], 1, 1);
        let mut member_vars = Vec::new();
        let mut keys = Vec::new();
        for (li, &list) in lists.iter().enumerate() {
            for &item in &universe {
                member_vars.push(solver.store.list_member_var(list, item).expect("member var"));
                keys.push((li, item));
            }
        }
        let mut set = std::collections::BTreeSet::new();
        if cdcl {
            qayd::search::solve(&mut solver, &member_vars, |s: &Solver| {
                set.insert(member_vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
                SearchControl::Continue
            });
        } else {
            solve_domains(&mut solver, |_, dom| {
                set.insert(keys.iter().map(|&(li, item)| i32::from(dom.lists[li].contains(&item))).collect::<Vec<_>>());
                SearchControl::Continue
            });
        }
        set
    };
    let cdcl = solve_set(true);
    let chrono = solve_set(false);
    assert_eq!(cdcl.len(), 12, "one of three items in list 0, the other two split across lists 1 and 2");
    assert_eq!(cdcl, chrono, "CDCL enumerates the same assignments as chronological (cardinality over-pruning guard)");
}

#[test]
fn domain_item_sum_mixed_weights_cdcl_same_solution_set() {
    // One list, weights 5*m1 + 7*m2 - 3*m3 - 2*m4 bounded to [0, 4]. Enumerate the
    // membership variables under CDCL (reasoning over the ListItemSum explanations,
    // both positive and negative weight cases) and under chronological domain
    // search: the sets must match -- the over-pruning guard for item_sum.
    let universe = vec![1, 2, 3, 4];
    let weights = vec![(1i32, 5i64), (2, 7), (3, -3), (4, -2)];
    let solve_set = |cdcl: bool| -> std::collections::BTreeSet<Vec<i32>> {
        let mut solver = Solver::new();
        let list = solver.store.new_list(universe.clone());
        list_item_sum(&mut solver, list, weights.clone(), 0, 4);
        let member_vars: Vec<_> = universe.iter().map(|&item| solver.store.list_member_var(list, item).expect("member var")).collect();
        let mut set = std::collections::BTreeSet::new();
        if cdcl {
            qayd::search::solve(&mut solver, &member_vars, |s: &Solver| {
                set.insert(member_vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
                SearchControl::Continue
            });
        } else {
            solve_domains(&mut solver, |_, dom| {
                set.insert(universe.iter().map(|&item| i32::from(dom.lists[0].contains(&item))).collect::<Vec<_>>());
                SearchControl::Continue
            });
        }
        set
    };
    let cdcl = solve_set(true);
    let chrono = solve_set(false);
    assert_eq!(cdcl.len(), 6, "subsets whose weighted sum lands in [0, 4]");
    assert_eq!(cdcl, chrono, "CDCL enumerates the same subsets as chronological (item_sum over-pruning guard)");
}

#[test]
fn domain_item_precedence_pushes_after_forward() {
    // 10 forced into the last list: precedence forces 20 there too (no earlier
    // list can hold it). Proves the propagation fires.
    let universe = vec![10, 20];
    let mut solver = Solver::new();
    let lists: Vec<_> = (0..3).map(|_| solver.store.new_list(universe.clone())).collect();
    partition(&mut solver, &lists, &universe);
    item_precedence(&mut solver, &lists, 10, 20);
    solver.store.require_list_item(lists[2], 10).unwrap();
    solver.propagate().unwrap();
    assert!(!solver.store.list_possible(lists[0], 20));
    assert!(!solver.store.list_possible(lists[1], 20));
    assert!(solver.store.list_required(lists[2], 20));
}

#[test]
fn domain_item_precedence_cdcl_same_solution_set() {
    // Partition three items across three ordered lists, with item_precedence(10, 20)
    // (list index of 10 <= list index of 20). Enumerate memberships under CDCL
    // (reasoning over the precedence explanations) and chronological domain search:
    // the sets must match -- the over-pruning guard for item_precedence.
    let universe = vec![10, 20, 30];
    let solve_set = |cdcl: bool| -> std::collections::BTreeSet<Vec<i32>> {
        let mut solver = Solver::new();
        let lists: Vec<_> = (0..3).map(|_| solver.store.new_list(universe.clone())).collect();
        partition(&mut solver, &lists, &universe);
        item_precedence(&mut solver, &lists, 10, 20);
        let mut member_vars = Vec::new();
        let mut keys = Vec::new();
        for (li, &list) in lists.iter().enumerate() {
            for &item in &universe {
                member_vars.push(solver.store.list_member_var(list, item).expect("member var"));
                keys.push((li, item));
            }
        }
        let mut set = std::collections::BTreeSet::new();
        if cdcl {
            qayd::search::solve(&mut solver, &member_vars, |s: &Solver| {
                set.insert(member_vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
                SearchControl::Continue
            });
        } else {
            solve_domains(&mut solver, |_, dom| {
                set.insert(keys.iter().map(|&(li, item)| i32::from(dom.lists[li].contains(&item))).collect::<Vec<_>>());
                SearchControl::Continue
            });
        }
        set
    };
    let cdcl = solve_set(true);
    let chrono = solve_set(false);
    assert_eq!(cdcl.len(), 18, "six ordered (list10 <= list20) pairs times three lists for item 30");
    assert_eq!(cdcl, chrono, "CDCL enumerates the same assignments as chronological (item_precedence over-pruning guard)");
}
