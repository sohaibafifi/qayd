use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::ids::VarId;
use crate::lcg::lit::{AtomKind, AtomTable, Lit, LitOrConst};
use crate::lcg::semantic::{self, SemanticMinimizationStats};
use crate::lcg::trail::{Cdcl, Conflict, Reason};
use crate::lcg::view::Tri;
use crate::Solver;

fn literal(value: LitOrConst) -> Lit {
    match value {
        LitOrConst::Lit(literal) => literal,
        other => panic!("expected a nonconstant literal, got {other:?}"),
    }
}

fn holds(atoms: &AtomTable, clause: &[Lit], assignment: &[i32]) -> bool {
    clause.iter().any(|literal| {
        let atom_truth = match atoms.decode(literal.atom()) {
            AtomKind::Ge { var, k } => assignment[var.index()] >= k,
            AtomKind::Eq { var, v } => assignment[var.index()] == v,
        };
        atom_truth == literal.is_positive()
    })
}

fn check_equivalence(atoms: &AtomTable, original: &[Lit], assignments: &[Vec<i32>]) -> Vec<Lit> {
    let mut clause = original.to_vec();
    let allocated = atoms.num_atoms();
    let (removed, interrupted) = semantic::minimize(atoms, &mut clause, &AtomicBool::new(false));
    assert!(!interrupted);
    assert_eq!(clause.len() + removed, original.len());
    assert_eq!(clause.first(), original.first(), "asserting literal changed");
    assert_eq!(allocated, atoms.num_atoms(), "minimizer materialized an atom");
    let mut position = 0;
    for &survivor in &clause {
        position += original[position..].iter().position(|&candidate| candidate == survivor).unwrap() + 1;
    }
    for assignment in assignments {
        assert_eq!(
            holds(atoms, &clause, assignment),
            holds(atoms, original, assignment),
            "clause {original:?} became {clause:?} at {assignment:?}"
        );
    }
    clause
}

#[test]
fn semantic_minimization_matches_exhaustive_two_variable_clause_truth_tables() {
    let atoms = AtomTable::build(2, |_| (-1, 1));
    let literals = (0..atoms.num_atoms() as u32).flat_map(|atom| [Lit::positive(atom), Lit::negative(atom)]).collect::<Vec<_>>();
    let assignments = (-1..=1).flat_map(|x| (-1..=1).map(move |y| vec![x, y])).collect::<Vec<_>>();
    // Includes repeated literals, tautologies, every polarity, and unrelated
    // variables. Exhaustion does not assume that a clause is falsified now.
    for &first in &literals {
        for &second in &literals {
            check_equivalence(&atoms, &[first, second], &assignments);
            for &third in &literals {
                check_equivalence(&atoms, &[first, second, third], &assignments);
            }
        }
    }
}

#[test]
fn semantic_minimization_matches_exhaustive_four_literal_hole_combinations() {
    let atoms = AtomTable::build(1, |_| (0, 3));
    let literals = (0..atoms.num_atoms() as u32).flat_map(|atom| [Lit::positive(atom), Lit::negative(atom)]).collect::<Vec<_>>();
    let assignments = (0..=3).map(|value| vec![value]).collect::<Vec<_>>();
    for &first in &literals {
        for &second in &literals {
            for &third in &literals {
                for &fourth in &literals {
                    check_equivalence(&atoms, &[first, second, third, fourth], &assignments);
                }
            }
        }
    }
}

#[test]
fn semantic_minimization_combines_bounds_and_distinct_holes() {
    let atoms = AtomTable::build(2, |var| if var == VarId(0) { (0, 8) } else { (0, 1) });
    let x = VarId(0);
    let guard = literal(atoms.ge(VarId(1), 1));
    let ge_three = literal(atoms.ge(x, 3));
    let ge_five = literal(atoms.ge(x, 5));
    let eq_three = literal(atoms.eq(x, 3));
    let eq_four = literal(atoms.eq(x, 4));
    let assignments = (0..=8).flat_map(|x| (0..=1).map(move |y| vec![x, y])).collect::<Vec<_>>();
    let clause = check_equivalence(&atoms, &[guard, ge_three, ge_five, eq_three, eq_four], &assignments);
    assert_eq!(clause, [guard, ge_five, eq_three, eq_four]);

    // Repeated exclusions cannot pretend to cover a second value.
    let original = [guard, ge_three, ge_five, eq_three, eq_three];
    let clause = check_equivalence(&atoms, &original, &assignments);
    assert!(clause.contains(&ge_three));
}

#[test]
fn semantic_minimization_handles_sparse_sign_and_wide_domains_without_new_atoms() {
    for values in [vec![-1, 1], vec![-100_000, -3, 2, 90_000], (0..=400).step_by(2).collect()] {
        let support: Arc<[i32]> = Arc::from(values.clone());
        let atoms = AtomTable::build_active_sparse(
            1,
            |_| true,
            |_| values == [-1, 1],
            |_| Some(Arc::clone(&support)),
            |_| (*values.first().unwrap(), *values.last().unwrap()),
        );
        let x = VarId(0);
        let boundary = values[1];
        let first = literal(atoms.ge(x, boundary));
        let last = literal(atoms.eq(x, *values.last().unwrap()));
        let assignments = values.iter().map(|&value| vec![value]).collect::<Vec<_>>();
        let clause = check_equivalence(&atoms, &[first, last], &assignments);
        assert_eq!(clause, [first]);
    }

    let atoms = AtomTable::build(1, |_| (i32::MIN, i32::MAX));
    assert!(atoms.is_lazy(VarId(0)));
    let lo = literal(atoms.ge(VarId(0), i32::MIN + 1)).negate();
    let hi = literal(atoms.ge(VarId(0), i32::MAX));
    let endpoints = [vec![i32::MIN], vec![i32::MIN + 1], vec![0], vec![i32::MAX - 1], vec![i32::MAX]];
    assert_eq!(check_equivalence(&atoms, &[lo, hi], &endpoints), [lo, hi]);
}

#[test]
fn semantic_minimization_uses_widened_arithmetic_at_both_integer_extremes() {
    let atoms = AtomTable::build(2, |var| if var == VarId(0) { (i32::MIN, i32::MAX) } else { (0, 1) });
    let x = VarId(0);
    let guard = literal(atoms.ge(VarId(1), 1));
    let high = [guard, literal(atoms.ge(x, i32::MAX - 1)), literal(atoms.ge(x, i32::MAX)), literal(atoms.eq(x, i32::MAX - 1))];
    let low = [
        guard,
        literal(atoms.ge(x, i32::MIN + 2)).negate(),
        literal(atoms.ge(x, i32::MIN + 1)).negate(),
        literal(atoms.eq(x, i32::MIN + 1)),
    ];
    let assignments = [i32::MIN, i32::MIN + 1, i32::MIN + 2, -1, 0, i32::MAX - 2, i32::MAX - 1, i32::MAX]
        .into_iter()
        .flat_map(|x| (0..=1).map(move |y| vec![x, y]))
        .collect::<Vec<_>>();
    assert_eq!(check_equivalence(&atoms, &high, &assignments), [high[0], high[2], high[3]]);
    assert_eq!(check_equivalence(&atoms, &low, &assignments), [low[0], low[2], low[3]]);
}

#[test]
fn semantic_minimization_is_opt_in_and_ignores_live_decision_bounds() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 5);
    let y = solver.new_var_range(0, 1);
    let mut cdcl = Cdcl::new(&mut solver, &[x, y]);
    assert!(cdcl.init());
    let guard = literal(cdcl.atoms.ge(y, 1));
    let ge_one = literal(cdcl.atoms.ge(x, 1));
    let ge_three = literal(cdcl.atoms.ge(x, 3));
    let original = [guard, ge_three.negate(), ge_one.negate()];
    let mut clause = original.to_vec();
    cdcl.semantic_minimize_learnt(&mut clause);
    assert_eq!(clause, original);
    assert_eq!(cdcl.semantic_minimization_stats(), SemanticMinimizationStats::default());

    cdcl.set_semantic_minimization(true);
    cdcl.semantic_minimize_learnt(&mut clause);
    assert_eq!(clause, [guard, ge_three.negate()]);
    assert_eq!(cdcl.semantic_minimization_stats().removed_literals, 1);

    cdcl.decide(ge_three).unwrap();
    let mut independent = vec![guard, ge_one.negate()];
    cdcl.semantic_minimize_learnt(&mut independent);
    assert_eq!(independent, [guard, ge_one.negate()], "a current-domain false literal is not a root fact");
    cdcl.backjump_to(0);
    assert_eq!(cdcl.value(ge_one), Tri::Unknown);
}

#[test]
fn semantic_minimization_respects_proof_logging_in_either_configuration_order() {
    for enable_first in [false, true] {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 5);
        let mut proof = |_: &[Lit]| {};
        let mut cdcl = Cdcl::new(&mut solver, &[x]);
        let original = [literal(cdcl.atoms.ge(x, 1)), literal(cdcl.atoms.ge(x, 3))];
        if enable_first {
            cdcl.set_semantic_minimization(true);
        }
        cdcl.set_proof_logger(&mut proof);
        if !enable_first {
            cdcl.set_semantic_minimization(true);
        }
        let mut clause = original.to_vec();
        cdcl.semantic_minimize_learnt(&mut clause);
        assert_eq!(clause, original);
        assert_eq!(cdcl.semantic_minimization_stats().skipped_proof_logging, 1);
        assert_eq!(cdcl.semantic_minimization_stats().removed_literals, 0);
    }
}

#[test]
fn semantic_minimization_respects_cancellation_and_clause_work_limit() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 5);
    let stop = AtomicBool::new(true);
    let mut cdcl = Cdcl::new(&mut solver, &[x]);
    cdcl.set_semantic_minimization(true);
    cdcl.set_stop(&stop);
    let first = literal(cdcl.atoms.ge(x, 1));
    let second = literal(cdcl.atoms.ge(x, 3));
    let mut clause = vec![first, second];
    cdcl.semantic_minimize_learnt(&mut clause);
    assert_eq!(clause, [first, second]);
    assert_eq!(cdcl.semantic_minimization_stats().interrupted, 1);

    let original = vec![first; semantic::MAX_CLAUSE_LEN + 1];
    let mut clause = original.clone();
    cdcl.semantic_minimize_learnt(&mut clause);
    assert_eq!(clause, original);
    assert_eq!(cdcl.semantic_minimization_stats().skipped_oversized, 1);
}

#[test]
fn semantic_minimization_shortens_real_conflict_and_preserves_backjump_watches() {
    for enabled in [false, true] {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 5);
        let y = solver.new_var_range(0, 1);
        let mut cdcl = Cdcl::new(&mut solver, &[x, y]);
        cdcl.set_semantic_minimization(enabled);
        assert!(cdcl.init());
        let ge_one = literal(cdcl.atoms.ge(x, 1));
        let ge_three = literal(cdcl.atoms.ge(x, 3));
        let guard = literal(cdcl.atoms.ge(y, 1));
        cdcl.decide(ge_one).unwrap();
        cdcl.decide(ge_three).unwrap();
        cdcl.decide(guard).unwrap();
        // Add a genuine disjunctive constraint after decisions, making it
        // immediately conflicting. Both x bounds have independent decisions,
        // so implication-graph minimization cannot remove the weaker one.
        let conflict = cdcl.add_clause(vec![ge_one.negate(), ge_three.negate(), guard.negate()], false, 0);
        assert!(cdcl.resolve_conflict(Conflict::Clause(conflict)));
        assert_eq!(cdcl.decision_level(), 2);
        assert_eq!(cdcl.value(guard), Tri::False);
        assert!(cdcl.propagate_and_learn());
        assert_eq!(cdcl.semantic_minimization_stats().removed_literals, u64::from(enabled));
        let Reason::Clause(learned) = cdcl.reason_of(guard.atom()) else {
            panic!("asserting literal lost its clause reason");
        };
        assert_eq!(cdcl.clauses.get(*learned).lits.len(), if enabled { 2 } else { 3 });

        cdcl.backjump_to(0);
        cdcl.decide(guard).unwrap();
        assert!(cdcl.propagate_and_learn());
        cdcl.decide(ge_one).unwrap();
        assert!(cdcl.propagate_and_learn());
        assert_eq!(cdcl.value(ge_three), Tri::False, "watched learned clause failed after backjump");
    }
}
