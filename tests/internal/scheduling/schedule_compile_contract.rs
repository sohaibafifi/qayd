use qayd::model::{CompiledCollection, Model, ModelPackage};
use qayd::orchestrator::{compile_model_plan, SolveBudget, SolveError, SolveMode, SolveRequest};

#[test]
fn exact_schedule_rejects_oversized_duration_during_compilation() {
    let mut model = Model::new();
    let duration = i64::from(i32::MAX) + 1;
    model.interval(0, 0, duration);

    model.validate().expect("the interval is semantically valid");
    CompiledCollection::compile(&model).expect("the collection IR preserves the semantic i64 duration");

    let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };
    let error = match compile_model_plan(&ModelPackage::new(model), &request, &SolveBudget::new(None)) {
        Ok(_) => panic!("an exact schedule with an unrepresentable duration was accepted"),
        Err(error) => error,
    };

    let diagnostic = match error {
        SolveError::Compile(diagnostic) => diagnostic,
        other => panic!("numeric lowering must fail as a compile error, got {other}"),
    };
    assert!(diagnostic.contains("interval duration"), "missing duration context: {diagnostic}");
    assert!(diagnostic.contains("i32"), "missing representation context: {diagnostic}");
}
