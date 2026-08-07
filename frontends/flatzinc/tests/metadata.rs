#![allow(dead_code)]

#[path = "../src/model.rs"]
mod model;
#[path = "../src/parse.rs"]
mod parse;
#[path = "../src/text.rs"]
mod text;

use qayd::model::{ModelObject, SourceRange};

fn annotation_values<'a>(annotations: impl Iterator<Item = &'a String>) -> Vec<&'a str> {
    annotations.map(String::as_str).collect()
}

#[test]
fn package_retains_item_annotations_and_source_ranges() {
    let source = "var 0..5: x :: output_var :: is_defined_var;\n\
                  constraint int_ge(x, 1) :: defines_var(x);\n\
                  solve :: int_search([x], first_fail, indomain_min, complete) minimize x;\n";

    let parsed = parse::parse(source).expect("annotated FlatZinc should parse");
    let (package, _) = parsed.into_package();
    package.validate().expect("retained frontend metadata should be valid");

    let values = annotation_values(package.metadata.annotations.values());
    assert!(values.contains(&"output_var"));
    assert!(values.contains(&"is_defined_var"));
    assert!(values.contains(&"defines_var(x)"));
    assert!(values.contains(&"int_search([x], first_fail, indomain_min, complete)"));

    let variable =
        *package.metadata.frontend_ids.get(&("flatzinc".to_string(), "x".to_string())).expect("x should retain its FlatZinc identifier");
    let variable_annotations = annotation_values(package.metadata.object_annotations[&variable].values());
    assert!(variable_annotations.contains(&"output_var"));
    assert!(variable_annotations.contains(&"is_defined_var"));

    let constraint = ModelObject::Constraint(qayd::model::ConstraintRef(0));
    assert!(annotation_values(package.metadata.object_annotations[&constraint].values()).contains(&"defines_var(x)"));

    let objective = ModelObject::Objective(qayd::model::ObjectiveRef(0));
    assert!(annotation_values(package.metadata.object_annotations[&objective].values())
        .contains(&"int_search([x], first_fail, indomain_min, complete)"));

    let expected_variable_source = SourceRange { source: "flatzinc".to_string(), start: 0, end: source.find(';').unwrap() + 1 };
    let constraint_start = source.find("constraint").unwrap();
    let expected_constraint_source = SourceRange {
        source: "flatzinc".to_string(),
        start: constraint_start,
        end: source[constraint_start..].find(';').unwrap() + constraint_start + 1,
    };
    let solve_start = source.find("solve").unwrap();
    let expected_objective_source =
        SourceRange { source: "flatzinc".to_string(), start: solve_start, end: source[solve_start..].find(';').unwrap() + solve_start + 1 };
    assert_eq!(package.metadata.sources[&variable], vec![expected_variable_source]);
    assert_eq!(package.metadata.sources[&constraint], vec![expected_constraint_source]);
    assert_eq!(package.metadata.sources[&objective], vec![expected_objective_source]);
}

#[test]
fn malformed_annotation_is_an_explicit_unsupported_error() {
    let error = parse::parse("var 0..5: x; solve :: int_search([x], first_fail satisfy;").err().expect("malformed annotation must fail");
    assert!(error.contains("Unsupported malformed FlatZinc annotation"), "{error}");
}

#[test]
fn declaration_annotation_before_initializer_is_retained() {
    let parsed = parse::parse("var 0..5: x :: output_var :: is_defined_var = 3; solve satisfy;")
        .expect("standard annotation-before-initializer form should parse");
    let (package, outputs) = parsed.into_package();
    let variable = package.metadata.frontend_ids[&("flatzinc".to_string(), "x".to_string())];
    let values = annotation_values(package.metadata.object_annotations[&variable].values());
    assert!(values.contains(&"output_var"));
    assert!(values.contains(&"is_defined_var"));
    assert_eq!(outputs.len(), 1);
    assert_eq!(package.model.constraints().len(), 1, "the initializer must still be lowered");
}
