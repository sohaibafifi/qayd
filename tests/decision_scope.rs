use std::sync::atomic::AtomicBool;

use qayd::frontends::xcsp::{self, RunOptions};
use qayd::model::{Constraint, IntExpr, Model, ModelPackage, Objective, Relation};
use qayd::orchestrator::{solve_model_silent, CpControls, SolveError, SolveMode, SolveRequest, SolveStatus};

fn coupled_binary_model() -> ModelPackage {
    let mut model = Model::new();
    let primary = model.bool_var();
    let hidden = model.bool_var();
    model.add_constraint(Constraint::Linear { terms: vec![(1, primary), (1, hidden)], relation: Relation::Ge, rhs: 0 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(primary) });
    ModelPackage::new(model)
}

fn metadata_value<'a>(result: &'a qayd::orchestrator::SolveResult, key: &str) -> Option<&'a str> {
    result.reports().iter().flat_map(|report| &report.metadata).find_map(|(name, value)| (name == key).then_some(value.as_str()))
}

fn run_xcsp_with_semantic_branching(xml: &str) -> String {
    let mut output = Vec::new();
    xcsp::run_to_with_options(
        xml,
        false,
        &AtomicBool::new(false),
        &mut output,
        RunOptions { semantic_branching: true, ..RunOptions::default() },
    )
    .unwrap();
    String::from_utf8(output).unwrap()
}

#[test]
fn explicit_scope_keeps_the_full_semantic_assignment() {
    let request = SolveRequest { mode: SolveMode::Exact, primary_branch_scope: Some(vec![0]), ..SolveRequest::default() };

    let result = solve_model_silent(&coupled_binary_model(), &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    let integers = &result.primal().unwrap().assignment().integers;
    assert_eq!(integers.len(), 2);
    assert!(integers.iter().all(Option::is_some));
}

#[test]
fn legacy_and_completion_only_requests_both_return_complete_assignments() {
    for scope in [None, Some(Vec::new())] {
        let request = SolveRequest { mode: SolveMode::Exact, primary_branch_scope: scope, ..SolveRequest::default() };
        let result = solve_model_silent(&coupled_binary_model(), &request).unwrap();
        assert_eq!(result.status(), SolveStatus::Optimal);
        assert!(result.primal().unwrap().assignment().integers.iter().all(Option::is_some));
    }
}

#[test]
fn scoped_single_worker_csp_keeps_the_default_chronological_search() {
    let mut model = Model::new();
    model.bool_var();
    model.bool_var();
    let request = SolveRequest { mode: SolveMode::Exact, primary_branch_scope: Some(vec![0]), ..SolveRequest::default() };

    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(metadata_value(&result, "csp_search"), Some("chronological-dfs"));
    assert!(result.primal().unwrap().assignment().integers.iter().all(Option::is_some));
}

#[test]
fn invalid_primary_scope_references_are_rejected() {
    let duplicate = SolveRequest { primary_branch_scope: Some(vec![0, 0]), ..SolveRequest::default() };
    assert!(matches!(
        solve_model_silent(&coupled_binary_model(), &duplicate),
        Err(SolveError::InvalidRequest(message)) if message.contains("appears twice in primary_branch_scope")
    ));

    let out_of_range = SolveRequest { primary_branch_scope: Some(vec![2]), ..SolveRequest::default() };
    assert!(matches!(
        solve_model_silent(&coupled_binary_model(), &out_of_range),
        Err(SolveError::InvalidRequest(message)) if message.contains("unknown integer variable 2")
    ));

    let branch_outside = SolveRequest { primary_branch_scope: Some(vec![0]), branch_order: vec![1], ..SolveRequest::default() };
    assert!(matches!(
        solve_model_silent(&coupled_binary_model(), &branch_outside),
        Err(SolveError::InvalidRequest(message)) if message.contains("outside primary_branch_scope")
    ));
}

#[test]
fn decomposition_preserves_an_explicit_empty_component_scope() {
    let mut model = Model::new();
    model.bool_var();
    model.bool_var();
    let request = SolveRequest { mode: SolveMode::Exact, primary_branch_scope: Some(vec![0]), ..SolveRequest::default() };

    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert!(result.primal().unwrap().assignment().integers.iter().all(Option::is_some));
}

#[test]
fn no_learn_csp_keeps_chronological_dfs_with_an_explicit_scope() {
    let mut model = Model::new();
    let primary = model.bool_var();
    let completion = model.bool_var();
    model.add_constraint(Constraint::Linear { terms: vec![(1, primary), (1, completion)], relation: Relation::Eq, rhs: 1 });
    let request = SolveRequest {
        mode: SolveMode::Exact,
        cp: CpControls { no_learn_csp: true, ..CpControls::default() },
        primary_branch_scope: Some(vec![0]),
        ..SolveRequest::default()
    };

    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(metadata_value(&result, "csp_search"), Some("chronological-dfs"));
    assert_eq!(result.aggregate_search_stats().learned_lits, 0);
    assert!(result.primal().unwrap().assignment().integers.iter().all(Option::is_some));
}

#[test]
fn split_and_lns_roles_do_not_promote_completion_variables() {
    let split_request = SolveRequest {
        mode: SolveMode::Exact,
        threads: 2,
        cp: CpControls { split: true, ..CpControls::default() },
        primary_branch_scope: Some(vec![0]),
        ..SolveRequest::default()
    };
    let split = solve_model_silent(&coupled_binary_model(), &split_request).unwrap();
    assert_eq!(metadata_value(&split, "split_jobs"), Some("1"));

    let lns_request = SolveRequest {
        mode: SolveMode::Exact,
        threads: 2,
        cp: CpControls { lns: 1, ..CpControls::default() },
        primary_branch_scope: Some(Vec::new()),
        ..SolveRequest::default()
    };
    let lns = solve_model_silent(&coupled_binary_model(), &lns_request).unwrap();
    assert_eq!(metadata_value(&lns, "lns_attempts"), Some("0"));
    assert!(lns.primal().unwrap().assignment().integers.iter().all(Option::is_some));
}

#[test]
fn xcsp_existential_index_witness_is_completed_after_declared_variables() {
    let xml = r#"
<instance format="XCSP3" type="COP">
  <variables>
    <array id="x" size="[2]">0..1</array>
  </variables>
  <constraints>
    <element>
      <list>x[]</list>
      <value>0</value>
    </element>
  </constraints>
  <objectives>
    <minimize type="sum">x[]</minimize>
  </objectives>
</instance>
"#;

    let output = run_xcsp_with_semantic_branching(xml);

    assert!(output.contains("s OPTIMUM FOUND"), "{output}");
    assert!(output.contains("v 0 0"), "{output}");
}

#[test]
fn xcsp_objective_auxiliary_is_completed_outside_the_declared_scope() {
    let xml = r#"
<instance format="XCSP3" type="COP">
  <variables>
    <var id="x">0..2</var>
  </variables>
  <constraints></constraints>
  <objectives>
    <minimize>add(x,1)</minimize>
  </objectives>
</instance>
"#;

    let output = run_xcsp_with_semantic_branching(xml);

    assert!(output.contains("s OPTIMUM FOUND"), "{output}");
    assert!(output.contains("o 1"), "{output}");
    assert!(output.contains("v 0"), "{output}");
}
