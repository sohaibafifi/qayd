use std::sync::atomic::AtomicBool;

use crate::engines::ls::integer::{
    audit_compile_warm_start, audit_warm_start_preflight, compile_interruptible, has_additive_penalty_objective,
};
use crate::model::{CompiledCp, Constraint, IntExpr, IntGlobalConstraint, IntVarRef, Model, ModelPackage, Objective};
use crate::orchestrator::{
    compile_cp_plan, solve_cp_plan, solve_model, solve_model_silent, EngineKind, EventCallback, EventControl, IgnoreEvents, SearchPhase,
    SearchPolicy, SolveBudget, SolveEvent, SolveLimits, SolveRequest, SolveStatus, ValueSelector, VariableSelector,
};

fn equal(left: IntVarRef, right: IntVarRef) -> IntExpr {
    IntExpr::Eq(Box::new(left.into()), Box::new(right.into()))
}

fn proximity(left: IntVarRef, right: IntVarRef, distance: i64, reverse: bool) -> IntExpr {
    let penalty = IntExpr::Sub(
        Box::new(distance.into()),
        Box::new(IntExpr::Abs(Box::new(IntExpr::Sub(Box::new(left.into()), Box::new(right.into()))))),
    );
    IntExpr::Max(if reverse { vec![penalty, 0.into()] } else { vec![0.into(), penalty] })
}

fn base(preference: Option<Vec<i32>>) -> (Model, [IntVarRef; 3]) {
    let mut model = Model::new();
    let left = model.int_range(0, 2);
    let right = model.int_range(0, 2);
    let cost = model.int_range(0, 2);
    if let Some(array) = preference {
        model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::ElementConst { array, index: left, value: cost }));
    }
    (model, [left, right, cost])
}

fn objective(mut model: Model, terms: Vec<IntExpr>, minimize: bool) -> Model {
    model.add_objective(Objective::IntExpr { minimize, expr: IntExpr::Add(terms) });
    model
}

fn fixture() -> Model {
    let (mut model, [left, right, cost]) = base(Some(vec![2, 0, 1]));
    model.add_constraint(Constraint::Intension(IntExpr::Ne(Box::new(left.into()), Box::new(right.into()))));
    objective(
        model,
        vec![
            IntExpr::Mul(vec![3.into(), equal(left, right)]),
            IntExpr::Mul(vec![2.into(), proximity(left, right, 2, false)]),
            cost.into(),
            4.into(),
        ],
        true,
    )
}

#[test]
fn additive_penalty_admission_recognizes_nested_weighted_and_reordered_terms() {
    for reverse in [false, true] {
        let (model, [left, right, cost]) = base(Some(vec![2, 0, 1]));
        let model = objective(
            model,
            vec![
                IntExpr::Add(vec![equal(right, left), IntExpr::Mul(vec![2.into(), proximity(left, right, 2, reverse), 3.into()])]),
                IntExpr::Mul(vec![cost.into(), 2.into()]),
                7.into(),
            ],
            true,
        );
        assert!(has_additive_penalty_objective(&model, &AtomicBool::new(false)));
    }
}

#[test]
fn additive_penalty_admission_rejects_unqualified_and_nonpositive_shapes() {
    let (base, [left, right, cost]) = base(Some(vec![2, 0, 1]));
    let rejected = [
        vec![cost.into(), 1.into()],
        vec![left.into(), equal(left, right)],
        vec![equal(left, left)],
        vec![IntExpr::Eq(Box::new(left.into()), Box::new(1.into()))],
        vec![IntExpr::Mul(vec![(-1).into(), equal(left, right)])],
        vec![IntExpr::Mul(vec![0.into(), equal(left, right)])],
        vec![IntExpr::Mul(vec![i64::MAX.into(), 2.into(), equal(left, right)])],
        vec![IntExpr::Mul(vec![equal(left, right), cost.into()])],
        vec![proximity(left, right, 0, false)],
        vec![proximity(left, left, 2, false)],
        vec![equal(left, right), (-1).into()],
        vec![IntExpr::Abs(Box::new(IntExpr::Sub(Box::new(left.into()), Box::new(right.into()))))],
    ];
    for (index, terms) in rejected.into_iter().enumerate() {
        let model = objective(base.clone(), terms, true);
        assert!(!has_additive_penalty_objective(&model, &AtomicBool::new(false)), "accepted rejected shape {index}");
    }
    assert!(!has_additive_penalty_objective(&objective(base.clone(), vec![equal(left, right)], false), &AtomicBool::new(false)));
    let mut multiple = objective(base, vec![equal(left, right)], true);
    multiple.add_objective(Objective::IntExpr { minimize: true, expr: cost.into() });
    assert!(!has_additive_penalty_objective(&multiple, &AtomicBool::new(false)));
}

#[test]
fn additive_penalty_preferences_require_one_unconditional_nonnegative_definition() {
    for preference in [None, Some(vec![-1, 0, 1])] {
        let (model, [left, right, cost]) = base(preference);
        let model = objective(model, vec![equal(left, right), cost.into()], true);
        assert!(!has_additive_penalty_objective(&model, &AtomicBool::new(false)));
    }
    let (mut model, [left, right, cost]) = base(Some(vec![2, 0, 1]));
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::ElementConst { array: vec![2, 0, 1], index: left, value: cost }));
    assert!(!has_additive_penalty_objective(&objective(model, vec![equal(left, right), cost.into()], true), &AtomicBool::new(false)));

    let (mut model, [left, right, cost]) = base(None);
    let selector = model.bool_var();
    model.add_constraint(Constraint::Selected {
        selector,
        constraint: Box::new(Constraint::IntegerGlobal(IntGlobalConstraint::ElementConst {
            array: vec![2, 0, 1],
            index: left,
            value: cost,
        })),
    });
    assert!(!has_additive_penalty_objective(&objective(model, vec![equal(left, right), cost.into()], true), &AtomicBool::new(false)));
}

#[test]
fn additive_penalty_warm_start_counts_symbolic_objective_memory_and_honors_limits() {
    let model = fixture();
    let stop = AtomicBool::new(false);
    let compiled = CompiledCp::compile_interruptible(&model, &stop).unwrap().unwrap();
    let (plan_bytes, _, _) = audit_warm_start_preflight(&model, &stop).unwrap();
    let (admitted, peak_bytes) = audit_compile_warm_start(&model, &compiled, u64::MAX, &stop);
    assert!(admitted);
    let local = compile_interruptible(&model, &compiled, &stop, peak_bytes).unwrap().unwrap();
    assert!(local.additive_penalty_scout.is_some(), "the memory gate must include the compiled feasibility scout");
    assert_eq!(peak_bytes, plan_bytes * 2);
    assert_eq!(audit_compile_warm_start(&model, &compiled, peak_bytes, &stop), (true, peak_bytes));
    assert_eq!(audit_compile_warm_start(&model, &compiled, peak_bytes - 1, &stop), (false, 0));
    assert_eq!(audit_compile_warm_start(&model, &compiled, 0, &stop), (false, 0));
    assert_eq!(audit_compile_warm_start(&model, &compiled, u64::MAX, &AtomicBool::new(true)), (false, 0));
    assert!(!has_additive_penalty_objective(&model, &AtomicBool::new(true)));

    let (base, [left, right, _]) = base(None);
    let small = objective(base.clone(), vec![equal(left, right)], true);
    let large = objective(base, vec![equal(left, right); 100], true);
    let small = audit_warm_start_preflight(&small, &stop).unwrap();
    let large = audit_warm_start_preflight(&large, &stop).unwrap();
    assert_eq!(large.0 - small.0, 99 * 3 * 96, "ordinary symbolic objective nodes must contribute to the memory estimate");
    assert_eq!(large.2 - small.2, 99 * 3);
}

fn warm_started(result: &crate::orchestrator::SolveResult) -> bool {
    result
        .reports()
        .iter()
        .any(|report| report.metadata.iter().any(|(key, value)| key == "ls_role" && value == "additive_penalty_warm_start"))
}

fn scout_started(result: &crate::orchestrator::SolveResult) -> bool {
    result.reports().iter().any(|report| {
        report.engine == Some(EngineKind::IntegerExact)
            && report.metadata.iter().any(|(key, value)| key == "warm_start_method" && value == "objective_inputs_cp_scout")
    })
}

#[test]
fn additive_penalty_warm_start_is_replayed_and_exact_search_keeps_proof_authority() {
    let result = solve_model_silent(&ModelPackage::new(fixture()), &SolveRequest::default()).unwrap();
    // The feasible pair (2, 0) costs one preference unit plus the constant four.
    // Every pair at distance one costs at least two proximity units.
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [5]);
    assert!(warm_started(&result));
    assert!(scout_started(&result));
    assert!(result.proof().is_some());
    assert!(result.reports().iter().any(|report| report.engine == Some(EngineKind::IntegerExact)));
}

#[test]
fn additive_penalty_warm_start_respects_explicit_search_policies_and_zero_budget() {
    let request = SolveRequest {
        search_policy: SearchPolicy::new(vec![SearchPhase::new(vec![0, 1], VariableSelector::InputOrder, ValueSelector::Min)]),
        ..SolveRequest::default()
    };
    let result = solve_model_silent(&ModelPackage::new(fixture()), &request).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [5]);
    assert!(!warm_started(&result));
    let request =
        SolveRequest { limits: SolveLimits { time: Some(std::time::Duration::ZERO), ..SolveLimits::default() }, ..SolveRequest::default() };
    let result = solve_model_silent(&ModelPackage::new(fixture()), &request).unwrap();
    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(result.primal().is_none());
    assert!(!warm_started(&result));
}

#[test]
fn stopping_on_penalty_warm_start_retains_a_verified_incumbent_without_proof() {
    let mut candidate_source = None;
    let mut progress_source = None;
    let result = {
        let mut sink = EventCallback(|event| {
            Ok(match event {
                SolveEvent::Candidate(candidate) => {
                    candidate_source = Some(candidate.source());
                    EventControl::Continue
                }
                SolveEvent::Progress { engine, .. } => {
                    progress_source = Some(engine);
                    EventControl::Stop
                }
                _ => EventControl::Continue,
            })
        });
        solve_model(&ModelPackage::new(fixture()), &SolveRequest::default(), &mut sink).unwrap()
    };
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert!(result.primal().is_some());
    assert!(result.proof().is_none());
    assert!(warm_started(&result));
    assert!(scout_started(&result));
    assert_eq!(candidate_source, Some(EngineKind::IntegerExact));
    assert_eq!(progress_source, Some(EngineKind::IntegerExact));
}

#[test]
fn penalty_scout_scopes_objective_inputs_and_completes_other_decisions() {
    let mut model = fixture();
    let room = model.int_range(0, 2);
    model.add_constraint(Constraint::Intension(IntExpr::Ne(Box::new(room.into()), Box::new(IntVarRef(0).into()))));
    let stop = AtomicBool::new(false);
    let compiled = CompiledCp::compile_interruptible(&model, &stop).unwrap().unwrap();
    let local = compile_interruptible(&model, &compiled, &stop, u64::MAX).unwrap().unwrap();
    let phase = local.additive_penalty_scout.unwrap();
    assert_eq!(phase.scope, compiled.int_variables()[..2]);
    assert_eq!(phase.semantic_salts, [0, 1]);
    assert_eq!(phase.variable, VariableSelector::DomWdeg);
    assert_eq!(phase.value, ValueSelector::RandomSeeded);
    let mut sink =
        EventCallback(|event| Ok(if matches!(event, SolveEvent::Candidate(_)) { EventControl::Stop } else { EventControl::Continue }));
    let result = solve_model(&ModelPackage::new(model), &SolveRequest::default(), &mut sink).unwrap();
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert!(result.proof().is_none());
    assert!(scout_started(&result));
    let values = &result.primal().unwrap().assignment().integers;
    assert!(values.iter().all(Option::is_some), "the scout must complete decisions outside its objective scope");
    assert_ne!(values[room.0], values[0]);
    assert_ne!(values[0], values[1]);
    assert_eq!(values[2], Some([2, 0, 1][values[0].unwrap() as usize]));
}

#[test]
fn penalty_scout_does_not_start_after_cancellation_of_a_prepared_plan() {
    let model = fixture();
    let request = SolveRequest::default();
    let budget = SolveBudget::new(None);
    let plan = compile_cp_plan(&model, &request, &budget).unwrap();
    budget.cancel();
    let result = solve_cp_plan(&model, &plan, &request, &budget, &mut IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(result.primal().is_none());
    assert!(result.proof().is_none());
    assert!(!warm_started(&result));
}

#[test]
fn parsed_penalty_scout_follows_shifted_preference_inputs() {
    for start in [-2, 1] {
        let package = parse_penalty_fixture(&format!(
            r#"<instance format="XCSP3" type="COP">
              <variables><var id="a">{start}..{end}</var><var id="b">{start}..{end}</var>
                <var id="preference_index">{start}..{end}</var><var id="cost">0..2</var></variables>
              <constraints><element><list startIndex="{start}">2 0 1</list><index>preference_index</index><value>cost</value></element></constraints>
              <objectives><minimize type="sum"><list>eq(a,b) cost</list></minimize></objectives>
            </instance>"#,
            end = start + 2
        ));
        let stop = AtomicBool::new(false);
        let compiled = CompiledCp::compile_interruptible(&package.model, &stop).unwrap().unwrap();
        let plan = compile_interruptible(&package.model, &compiled, &stop, u64::MAX).unwrap().unwrap();
        let index = package
            .model
            .constraints()
            .iter()
            .find_map(|constraint| match constraint {
                Constraint::IntegerGlobal(IntGlobalConstraint::ElementConst { index, .. }) => Some(*index),
                _ => None,
            })
            .unwrap();
        // The frontend posts index0 - original = -start. The normalized
        // linear producer derives the original index from index0, so the
        // canonical scout decision is index0 rather than the source variable.
        assert!(plan.spec.is_derived(compiled.int_variables()[2]));
        assert!(plan.spec.is_derived(compiled.int_variables()[3]));
        assert!(!plan.spec.is_derived(compiled.int_variables()[index.0]));
        let phase = plan.additive_penalty_scout.unwrap();
        assert_eq!(phase.scope, [compiled.int_variables()[0], compiled.int_variables()[1], compiled.int_variables()[index.0]]);
        assert_eq!(phase.semantic_salts, [0, 1, index.0 as u64]);

        let request = SolveRequest::default();
        let budget = SolveBudget::new(None);
        let plan = compile_cp_plan(&package.model, &request, &budget).unwrap();
        let mut sink =
            EventCallback(|event| Ok(if matches!(event, SolveEvent::Candidate(_)) { EventControl::Stop } else { EventControl::Continue }));
        let result = solve_cp_plan(&package.model, &plan, &request, &budget, &mut sink).unwrap();
        assert_eq!(result.status(), SolveStatus::Satisfiable);
        assert!(result.proof().is_none());
        assert!(scout_started(&result));
        let candidate = result.primal().unwrap();
        let values = &candidate.assignment().integers;
        let original = values[2].unwrap();
        let shifted = values[index.0].unwrap();
        let cost = values[3].unwrap();
        assert_eq!(shifted, original - start);
        assert_eq!(cost, [2, 0, 1][shifted as usize]);
        assert_eq!(candidate.objectives(), [i64::from(values[0] == values[1]) + cost]);
    }
}

use crate::frontends::xcsp::callback as diagnostic_xcsp_callback;

fn parse_penalty_fixture(xml: &str) -> ModelPackage {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "qayd-penalty-element-{}-{}.xml",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::write(&path, xml).unwrap();
    let mut frontend = diagnostic_xcsp_callback::Model::new();
    let parsed = xcsp3_rust_parser::xcsp_runner::XcspRunner::run(path.to_str().unwrap(), &mut frontend);
    std::fs::remove_file(path).unwrap();
    parsed.unwrap();
    assert!(frontend.error.is_none(), "{:?}", frontend.error);
    frontend.into_package()
}

#[test]
fn parsed_literal_elements_preserve_offsets_without_interning_array_entries() {
    for start in [-2, 0, 1] {
        for literal_value in [false, true] {
            let variable = if literal_value { "" } else { "<var id=\"cost\">0..9</var>" };
            let value = if literal_value { "8" } else { "cost" };
            let xml = format!(
                "<instance format=\"XCSP3\" type=\"CSP\"><variables><var id=\"index\">{start}..{end}</var>{variable}</variables>
                 <constraints><element><list startIndex=\"{start}\">5 8 3</list><index>index</index><value>{value}</value></element></constraints></instance>",
                end = start + 2
            );
            let package = parse_penalty_fixture(&xml);
            let model = &package.model;
            // Two semantic variables: index plus declared cost, or index plus
            // the required literal value. Only an offset adds another index.
            assert_eq!(model.int_vars().len(), 2 + usize::from(start != 0));
            let elements = model
                .constraints()
                .iter()
                .filter_map(|constraint| match constraint {
                    Constraint::IntegerGlobal(IntGlobalConstraint::ElementConst { array, index, value }) => Some((array, index, value)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(elements.len(), 1);
            assert_eq!(elements[0].0, &[5, 8, 3]);
            assert!(!model
                .constraints()
                .iter()
                .any(|constraint| matches!(constraint, Constraint::IntegerGlobal(IntGlobalConstraint::Element { .. }))));
            let request = SolveRequest {
                mode: crate::orchestrator::SolveMode::LocalSearch,
                limits: SolveLimits { iterations: Some(100), ..SolveLimits::default() },
                ..SolveRequest::default()
            };
            for request in [SolveRequest::default(), request] {
                let result = solve_model_silent(&package, &request).unwrap();
                assert_eq!(result.status(), SolveStatus::Satisfiable);
                let assignment = &result.primal().unwrap().assignment().integers;
                let index = assignment[0].unwrap();
                let cost = assignment[elements[0].2 .0].unwrap();
                assert_eq!(cost, [5, 8, 3][usize::try_from(index - start).unwrap()]);
                if literal_value {
                    assert_eq!(index, start + 1);
                    assert_eq!(cost, 8);
                }
            }
        }
    }
}

#[test]
fn parsed_literal_preference_costs_qualify_for_penalty_warm_start() {
    let package = parse_penalty_fixture(
        r#"<instance format="XCSP3" type="COP">
          <variables><var id="a">0..2</var><var id="b">0..2</var><var id="cost">0..2</var></variables>
          <constraints><element><list>2 0 1</list><index>a</index><value>cost</value></element></constraints>
          <objectives><minimize type="sum"><list>eq(a,b) max(0,sub(2,dist(a,b))) cost</list><coeffs>3 2 1</coeffs></minimize></objectives>
        </instance>"#,
    );
    assert_eq!(package.model.int_vars().len(), 3);
    assert!(has_additive_penalty_objective(&package.model, &AtomicBool::new(false)));
    let compiled = CompiledCp::compile_interruptible(&package.model, &AtomicBool::new(false)).unwrap().unwrap();
    assert!(audit_compile_warm_start(&package.model, &compiled, u64::MAX, &AtomicBool::new(false)).0);
}

#[test]
#[ignore = "local examination corpus diagnostic; parsing and compilation only"]
fn examination_actual_semantic_penalty_admission_diagnostic() {
    let path = std::env::var("QAYD_PENALTY_DIAGNOSTIC_XML")
        .expect("set QAYD_PENALTY_DIAGNOSTIC_XML to the decompressed ExaminationTimetabling1-max-D1-1-16 XML");
    let mut frontend = diagnostic_xcsp_callback::Model::new();
    xcsp3_rust_parser::xcsp_runner::XcspRunner::run(&path, &mut frontend).unwrap();
    assert!(frontend.error.is_none(), "{:?}", frontend.error);
    let package = frontend.into_package();
    let model = &package.model;
    let stop = AtomicBool::new(false);
    let mut elements = 0usize;
    let mut constant_elements = 0usize;
    let mut fixed_nonnegative_elements = 0usize;
    for constraint in model.constraints() {
        match constraint {
            Constraint::IntegerGlobal(IntGlobalConstraint::Element { array, .. }) => {
                elements += 1;
                if array.iter().all(|variable| {
                    let domain = &model.int_vars()[variable.0];
                    match domain {
                        crate::model::IntDomain::Bool => false,
                        crate::model::IntDomain::Range { lo, hi } => lo == hi && *lo >= 0,
                        crate::model::IntDomain::Set(values) => {
                            values.first().is_some_and(|first| *first >= 0 && values.iter().all(|value| value == first))
                        }
                    }
                }) {
                    fixed_nonnegative_elements += 1;
                }
            }
            Constraint::IntegerGlobal(IntGlobalConstraint::ElementConst { .. }) => constant_elements += 1,
            _ => {}
        }
    }
    eprintln!(
        "parsed vars={} constraints={} elements={} fixed_nonnegative_elements={} constant_elements={} additive_admitted={}",
        model.int_vars().len(),
        model.constraints().len(),
        elements,
        fixed_nonnegative_elements,
        constant_elements,
        has_additive_penalty_objective(model, &stop)
    );
    eprintln!("original preflight={:?}", audit_warm_start_preflight(model, &stop));
    if let [Objective::IntExpr { minimize, expr: IntExpr::Add(terms) }] = model.objectives() {
        eprintln!("objective minimize={minimize} terms={} first={:?} last={:?}", terms.len(), terms.first(), terms.last());
    }
    let crate::model::IndependentDecomposition::Components(components) = model.independent_family_components_interruptible(&stop) else {
        panic!("expected the original objective and fixed-room components");
    };
    for (index, component) in components.iter().enumerate() {
        let compiled = CompiledCp::compile_interruptible(&component.model, &stop).unwrap().unwrap();
        eprintln!(
            "first physical domain size={} bounds={}..{}",
            compiled.problem().solver.store.size(compiled.int_variables()[0]),
            compiled.problem().solver.store.min(compiled.int_variables()[0]),
            compiled.problem().solver.store.max(compiled.int_variables()[0])
        );
        if !component.model.objectives().is_empty() {
            let plan = crate::engines::ls::integer::compile_interruptible(&component.model, &compiled, &stop, u64::MAX).unwrap().unwrap();
            let phase = plan.additive_penalty_scout.as_ref().expect("the original objective must admit its dependency scout");
            eprintln!("scout scope variables={} semantic_ids={:?}", phase.scope.len(), phase.semantic_salts);
            // Independent XML inspection gives 215 direct objective periods
            // and 254 preference indices. Their union excludes exactly these
            // three periods, which occur only in hard constraints.
            let expected = (0..261usize).filter(|period| ![76, 120, 255].contains(period)).collect::<Vec<_>>();
            assert_eq!(expected.len(), 258);
            assert_eq!(phase.scope, expected.iter().map(|&period| compiled.int_variables()[period]).collect::<Vec<_>>());
            assert_eq!(phase.semantic_salts, expected.iter().map(|&period| period as u64).collect::<Vec<_>>());
            for period in [76, 120, 255] {
                assert!(!plan.spec.is_derived(compiled.int_variables()[period]));
            }
            let domains = crate::engines::ls::cop::audit_local_domains(compiled.problem(), plan.spec, &stop).unwrap();
            let period = &domains[compiled.int_variables()[0].index()];
            eprintln!("first local period domain size={} values={period:?}", period.len());
            assert_eq!(period.len(), 28);
            assert!(period.iter().all(|value| !(6..=9).contains(value) && !(20..=23).contains(value) && !(34..=37).contains(value)));
        }

        eprintln!(
            "component={index} vars={} constraints={} tiers={:?} additive={} preflight={:?} compiled_warm={:?}",
            component.model.int_vars().len(),
            component.model.constraints().len(),
            component.objective_tiers,
            has_additive_penalty_objective(&component.model, &stop),
            audit_warm_start_preflight(&component.model, &stop),
            audit_compile_warm_start(&component.model, &compiled, u64::MAX, &stop),
        );
    }
    assert_eq!(elements, 0);
    assert_eq!(constant_elements, 254);
    assert!(has_additive_penalty_objective(model, &stop));
}
