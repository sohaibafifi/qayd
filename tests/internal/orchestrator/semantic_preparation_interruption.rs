use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use qayd::model::{
    Constraint, IndependentDecomposition, IntVarRef, Model, ModelMetadata, ModelObject, ModelPackage, Relation, SourceRange,
};
use qayd::orchestrator::{compile_model_plan, ExecutablePlan, SolveBudget, SolveError, SolveRequest};

#[test]
fn prearmed_stop_wins_at_every_semantic_preparation_entry() {
    let stop = AtomicBool::new(true);
    let mut package = ModelPackage::new(Model::new());
    package.metadata.names.insert(ModelObject::IntVar(IntVarRef(0)), "unknown".to_string());

    assert!(!package.model.validate_interruptible(&stop).expect("prearmed cancellation is not a validation error"));
    assert!(!package.validate_interruptible(&stop).expect("prearmed cancellation is not a metadata error"));
    assert!(matches!(package.model.independent_family_components_interruptible(&stop), IndependentDecomposition::Interrupted));

    let objects = BTreeMap::new();
    assert!(package.project_interruptible(Model::new(), &objects, &stop).is_none());

    let budget = SolveBudget::with_stop(None, Arc::new(AtomicBool::new(true)));
    assert!(matches!(compile_model_plan(&package, &SolveRequest::default(), &budget), Err(SolveError::Interrupted(_))));
}

#[test]
fn cancellation_inside_one_large_invalid_domain_is_not_reported_as_invalid() {
    let mut model = Model::new();
    model.int_set(vec![7; 1_000_000]);
    let stop = AtomicBool::new(false);

    std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(1));
            stop.store(true, Ordering::Release);
        });
        assert!(!model.validate_interruptible(&stop).expect("mid-validation cancellation must supersede accumulated diagnostics"));
    });
}

#[test]
fn independent_component_discovery_stops_during_large_arena_preparation() {
    let mut model = Model::new();
    for _ in 0..150_000 {
        model.bool_var();
    }
    let stop = AtomicBool::new(false);

    std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(1));
            stop.store(true, Ordering::Release);
        });
        assert!(matches!(model.independent_family_components_interruptible(&stop), IndependentDecomposition::Interrupted));
    });
}

#[test]
fn metadata_projection_stops_inside_one_large_projected_value() {
    let mut model = Model::new();
    let variable = model.bool_var();
    let projected_model = model.clone();
    let object = ModelObject::IntVar(variable);
    let mut package = ModelPackage::new(model);
    package.metadata.aliases.insert(object, vec!["alias".to_string(); 500_000]);
    let objects = BTreeMap::from([(object, object)]);
    let stop = AtomicBool::new(false);
    let ready = Barrier::new(2);

    std::thread::scope(|scope| {
        scope.spawn(|| {
            ready.wait();
            std::thread::sleep(Duration::from_millis(1));
            stop.store(true, Ordering::Release);
        });
        ready.wait();
        assert!(package.project_interruptible(projected_model, &objects, &stop).is_none());
    });
}

#[test]
fn completed_projection_preserves_metadata_semantics() {
    let mut model = Model::new();
    let variable = model.bool_var();
    let object = ModelObject::IntVar(variable);
    let mut package = ModelPackage::new(model.clone());
    package.metadata = ModelMetadata {
        names: BTreeMap::from([(object, "x".to_string())]),
        aliases: BTreeMap::from([(object, vec!["alias".to_string()])]),
        sources: BTreeMap::from([(object, vec![SourceRange { source: "instance".to_string(), start: 2, end: 3 }])]),
        shapes: BTreeMap::from([(object, vec![(1, 4)])]),
        frontend_ids: BTreeMap::from([(("xcsp".to_string(), "x".to_string()), object)]),
        outputs: vec![object],
        annotations: BTreeMap::from([("format".to_string(), "plain".to_string())]),
        object_annotations: BTreeMap::from([(object, BTreeMap::from([("kind".to_string(), "decision".to_string())]))]),
    };
    let objects = BTreeMap::from([(object, object)]);

    let projected = package.project_interruptible(model, &objects, &AtomicBool::new(false)).expect("an unarmed projection completes");

    assert_eq!(projected.metadata, package.metadata);
}

#[test]
fn completed_family_decomposition_preserves_component_metadata() {
    let mut model = Model::new();
    let integer = model.int_range(0, 1);
    let list = model.list(vec![1]);
    model.add_constraint(Constraint::Linear { terms: vec![(1, integer)], relation: Relation::Eq, rhs: 1 });
    model.add_constraint(Constraint::ListPartition { lists: vec![list], items: vec![1] });
    let integer_object = ModelObject::IntVar(integer);
    let list_object = ModelObject::ListVar(list);
    let mut package = ModelPackage::new(model);
    package.metadata.names.insert(integer_object, "x".to_string());
    package.metadata.names.insert(list_object, "route".to_string());

    let plan =
        compile_model_plan(&package, &SolveRequest::default(), &SolveBudget::new(None)).expect("an uncoupled mixed model decomposes");
    assert!(matches!(plan.description(), ExecutablePlan::Decomposed { components, .. } if components.len() == 2));
    let metadata = plan.audit_component_metadata();
    assert_eq!(metadata.len(), 2);
    assert!(metadata.iter().any(|item| item.names.get(&integer_object).is_some_and(|name| name == "x")));
    assert!(metadata.iter().any(|item| item.names.get(&list_object).is_some_and(|name| name == "route")));
}
