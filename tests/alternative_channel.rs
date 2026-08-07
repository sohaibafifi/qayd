//! `alternative_channel` (native alternative-master bounds channel): the propagator's
//! root fixpoint must equal the projection of the feasible set (a brute-force
//! fixpoint), and its full solution set must match a brute-force enumeration
//! (exercising backtracking). Also pins the capability-pruning rule: a mode
//! whose start window cannot reach the shared start becomes absent at the root.

use std::collections::BTreeSet;

use qayd::constraints::interval::{alternative_channel, exactly_one_mode};
use qayd::{solve_search, IntervalId, IntervalPresence, SearchControl, Solver, VarId};

/// One alternative-master instance: the shared start's domain and each mode's start
/// window `[a, b]`. Duration is irrelevant to the channel, so it is fixed at 1.
struct Instance {
    s_dom: (i32, i32),
    windows: Vec<(i32, i32)>,
}

/// Build a solver holding the shared start `S`, one optional interval per mode,
/// and the two co-posted propagators. Returns `S`, the mode intervals, and their
/// (start, presence) backing vars.
fn build(inst: &Instance) -> (Solver, VarId, Vec<IntervalId>, Vec<VarId>, Vec<VarId>) {
    let mut s = Solver::new();
    let shared = s.new_var_range(inst.s_dom.0, inst.s_dom.1);
    let mut modes = Vec::new();
    let mut starts = Vec::new();
    let mut presences = Vec::new();
    for &(a, b) in &inst.windows {
        let iv = s.store.new_optional_interval(a, b, 1);
        starts.push(s.store.interval_start_var(iv));
        presences.push(s.store.interval_presence_var(iv).expect("optional interval has a presence var"));
        modes.push(iv);
    }
    exactly_one_mode(&mut s, &modes);
    alternative_channel(&mut s, shared, &modes);
    (s, shared, modes, starts, presences)
}

/// Every feasible full assignment `(S, [s_m], [p_m])`: exactly one mode present,
/// the present mode's start equal to `S`, each start in its window.
fn brute_tuples(inst: &Instance) -> BTreeSet<(i32, Vec<i32>, Vec<i32>)> {
    let k = inst.windows.len();
    let mut out = BTreeSet::new();
    let mut starts = vec![0i32; k];
    // Enumerate the cross product of the mode start windows.
    fn rec(m: usize, inst: &Instance, starts: &mut Vec<i32>, out: &mut BTreeSet<(i32, Vec<i32>, Vec<i32>)>) {
        let k = inst.windows.len();
        if m == k {
            for chosen in 0..k {
                let s_val = starts[chosen];
                if s_val < inst.s_dom.0 || s_val > inst.s_dom.1 {
                    continue; // S must lie in its own domain
                }
                let presences: Vec<i32> = (0..k).map(|j| i32::from(j == chosen)).collect();
                out.insert((s_val, starts.clone(), presences));
            }
            return;
        }
        let (a, b) = inst.windows[m];
        for v in a..=b {
            starts[m] = v;
            rec(m + 1, inst, starts, out);
        }
    }
    rec(0, inst, &mut starts, &mut out);
    out
}

/// Enumerate the solver's full solution set over `[S, s_0.., p_0..]`.
fn solver_tuples(inst: &Instance) -> BTreeSet<(i32, Vec<i32>, Vec<i32>)> {
    let (mut s, shared, _modes, starts, presences) = build(inst);
    let mut vars = vec![shared];
    vars.extend(starts.iter().copied());
    vars.extend(presences.iter().copied());
    let k = starts.len();
    let mut out = BTreeSet::new();
    solve_search(&mut s, &vars, |st| {
        let s_val = st.store.value(shared);
        let s_vec: Vec<i32> = starts.iter().map(|&v| st.store.value(v)).collect();
        let p_vec: Vec<i32> = presences.iter().map(|&v| st.store.value(v)).collect();
        debug_assert_eq!(s_vec.len(), k);
        out.insert((s_val, s_vec, p_vec));
        SearchControl::Continue
    });
    out
}

#[test]
fn solution_set_matches_brute_force_with_backtracking() {
    let instances = [
        // Two overlapping modes: both choosable, S free where either can host it.
        Instance { s_dom: (0, 10), windows: vec![(0, 2), (5, 8)] },
        // A mode whose window sits partly outside S's domain.
        Instance { s_dom: (3, 7), windows: vec![(0, 10), (5, 6)] },
        // Three modes, one fully disjoint from S's domain (never choosable).
        Instance { s_dom: (0, 4), windows: vec![(0, 1), (2, 3), (8, 9)] },
    ];
    for inst in &instances {
        assert_eq!(solver_tuples(inst), brute_tuples(inst), "solver solutions must equal brute force");
    }
}

/// Projection of the feasible tuple set: tightest sound bounds/presence (the
/// brute-force fixpoint the propagator is compared against).
struct Projection {
    s_bounds: (i32, i32),
    start_bounds: Vec<(i32, i32)>,
    presence: Vec<IntervalPresence>,
}

fn project(inst: &Instance) -> Projection {
    let tuples = brute_tuples(inst);
    assert!(!tuples.is_empty(), "test instances are feasible");
    let k = inst.windows.len();
    let s_bounds = (tuples.iter().map(|(s, _, _)| *s).min().unwrap(), tuples.iter().map(|(s, _, _)| *s).max().unwrap());
    let mut start_bounds = Vec::with_capacity(k);
    let mut presence = Vec::with_capacity(k);
    for m in 0..k {
        start_bounds.push((tuples.iter().map(|(_, sv, _)| sv[m]).min().unwrap(), tuples.iter().map(|(_, sv, _)| sv[m]).max().unwrap()));
        let can_one = tuples.iter().any(|(_, _, pv)| pv[m] == 1);
        let can_zero = tuples.iter().any(|(_, _, pv)| pv[m] == 0);
        presence.push(match (can_one, can_zero) {
            (true, false) => IntervalPresence::Present,
            (false, true) => IntervalPresence::Absent,
            _ => IntervalPresence::Optional,
        });
    }
    Projection { s_bounds, start_bounds, presence }
}

#[test]
fn root_bounds_match_projection() {
    let instances = [
        Instance { s_dom: (0, 10), windows: vec![(0, 2), (5, 8)] },
        Instance { s_dom: (3, 7), windows: vec![(0, 10), (5, 6)] },
        Instance { s_dom: (0, 4), windows: vec![(0, 1), (2, 3), (8, 9)] },
        // Sole reachable mode: exactly-one forces it present, channel ties s == S.
        Instance { s_dom: (0, 4), windows: vec![(1, 3), (8, 9)] },
    ];
    for inst in &instances {
        let want = project(inst);
        let (mut s, shared, modes, _starts, _presences) = build(inst);
        s.propagate().expect("root propagation is consistent");
        assert_eq!((s.store.min(shared), s.store.max(shared)), want.s_bounds, "S bounds");
        for (m, &iv) in modes.iter().enumerate() {
            assert_eq!((s.store.interval_start_min(iv), s.store.interval_start_max(iv)), want.start_bounds[m], "mode {m} start bounds",);
            assert_eq!(s.store.interval_presence(iv), want.presence[m], "mode {m} presence");
        }
    }
}

#[test]
fn disjoint_mode_is_forbidden_at_root() {
    // Mode 2's window [8, 9] cannot reach any S in [0, 4]: it must go absent at
    // the root. Two reachable modes remain, so neither is yet forced present.
    let inst = Instance { s_dom: (0, 4), windows: vec![(0, 3), (1, 2), (8, 9)] };
    let (mut s, _shared, modes, _starts, _presences) = build(&inst);
    s.propagate().expect("root propagation is consistent");
    assert_eq!(s.store.interval_presence(modes[2]), IntervalPresence::Absent, "disjoint mode forced absent");
    assert_eq!(s.store.interval_presence(modes[0]), IntervalPresence::Optional, "reachable mode stays optional");
    assert_eq!(s.store.interval_presence(modes[1]), IntervalPresence::Optional, "reachable mode stays optional");
}
