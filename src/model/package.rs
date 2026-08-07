//! Frontend-neutral model metadata.
//!
//! A frontend owns parsing and rendering, but the orchestrator needs stable
//! names and source locations without depending on a concrete input format.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{ConstraintRef, IntVarRef, IntervalModeRef, IntervalVarRef, ListVarRef, Model, ObjectiveRef, SetVarRef};

/// Stable semantic object referenced by names and source locations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ModelObject {
    IntVar(IntVarRef),
    SetVar(SetVarRef),
    ListVar(ListVarRef),
    IntervalVar(IntervalVarRef),
    IntervalMode(IntervalModeRef),
    Constraint(ConstraintRef),
    Objective(ObjectiveRef),
}

/// A format-neutral source range. Frontends that do not expose locations leave
/// the source map empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRange {
    pub source: String,
    pub start: usize,
    pub end: usize,
}

/// Symbols and source locations retained alongside the semantic model.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelMetadata {
    /// Primary display name for each object.
    pub names: BTreeMap<ModelObject, String>,
    /// All aliases accepted by the source format.
    pub aliases: BTreeMap<ModelObject, Vec<String>>,
    /// One semantic object may originate from several source fragments.
    pub sources: BTreeMap<ModelObject, Vec<SourceRange>>,
    /// Array dimensions and lower bounds retained for protocol rendering.
    pub shapes: BTreeMap<ModelObject, Vec<(i64, i64)>>,
    /// Stable frontend identifiers, grouped by protocol name.
    pub frontend_ids: BTreeMap<(String, String), ModelObject>,
    /// Ordered output projection. Frontends decide how each object is rendered.
    pub outputs: Vec<ModelObject>,
    /// Output annotations are deliberately opaque to the core. Their keys are
    /// frontend-defined, while their values remain deterministic plain text.
    pub annotations: BTreeMap<String, String>,
    /// Object-local deterministic annotations.
    pub object_annotations: BTreeMap<ModelObject, BTreeMap<String, String>>,
}

/// Canonical handoff from every parser or model builder to the orchestrator.
#[derive(Clone, Default)]
pub struct ModelPackage {
    pub model: Model,
    pub metadata: ModelMetadata,
}

impl ModelPackage {
    pub fn new(model: Model) -> Self {
        Self { model, metadata: ModelMetadata::default() }
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let stop = AtomicBool::new(false);
        match self.validate_interruptible(&stop) {
            Ok(true) => Ok(()),
            Ok(false) => unreachable!("a private false cancellation flag cannot interrupt package validation"),
            Err(errors) => Err(errors),
        }
    }

    pub(crate) fn validate_interruptible(&self, stop: &AtomicBool) -> Result<bool, Vec<String>> {
        if interrupted(stop) {
            return Ok(false);
        }
        let mut errors = match self.model.validate_interruptible(stop) {
            Ok(true) => Vec::new(),
            Ok(false) => return Ok(false),
            Err(errors) => errors,
        };
        for object in self
            .metadata
            .names
            .keys()
            .chain(self.metadata.aliases.keys())
            .chain(self.metadata.sources.keys())
            .chain(self.metadata.shapes.keys())
            .chain(self.metadata.object_annotations.keys())
            .chain(self.metadata.outputs.iter())
            .chain(self.metadata.frontend_ids.values())
        {
            if interrupted(stop) {
                return Ok(false);
            }
            if !self.model.contains_object(*object) {
                errors.push(format!("metadata references unknown object {object:?}"));
            }
        }
        for (object, ranges) in &self.metadata.sources {
            for (index, range) in ranges.iter().enumerate() {
                if interrupted(stop) {
                    return Ok(false);
                }
                let Some(has_identifier) = has_non_whitespace(&range.source, stop) else {
                    return Ok(false);
                };
                if !has_identifier {
                    errors.push(format!("source range {index} for {object:?} has an empty source identifier"));
                }
                if range.start > range.end {
                    errors.push(format!("source range {index} for {object:?} starts at {} after its end {}", range.start, range.end));
                }
            }
        }
        if interrupted(stop) {
            Ok(false)
        } else if errors.is_empty() {
            Ok(true)
        } else {
            Err(errors)
        }
    }

    /// Project metadata into one independently compiled component. `None`
    /// means cancellation was observed before the projection was complete.
    pub(crate) fn project_interruptible(
        &self,
        model: Model,
        objects: &BTreeMap<ModelObject, ModelObject>,
        stop: &AtomicBool,
    ) -> Option<Self> {
        if interrupted(stop) {
            return None;
        }
        let metadata = ModelMetadata {
            names: project_entries(&self.metadata.names, objects, stop)?,
            aliases: project_vec_entries(&self.metadata.aliases, objects, stop)?,
            sources: project_vec_entries(&self.metadata.sources, objects, stop)?,
            shapes: project_vec_entries(&self.metadata.shapes, objects, stop)?,
            frontend_ids: project_frontend_ids(&self.metadata.frontend_ids, objects, stop)?,
            outputs: project_outputs(&self.metadata.outputs, objects, stop)?,
            annotations: clone_annotations(&self.metadata.annotations, stop)?,
            object_annotations: project_object_annotations(&self.metadata.object_annotations, objects, stop)?,
        };
        (!interrupted(stop)).then_some(Self { model, metadata })
    }
}

impl From<Model> for ModelPackage {
    fn from(model: Model) -> Self {
        Self::new(model)
    }
}

fn interrupted(stop: &AtomicBool) -> bool {
    stop.load(Ordering::Acquire)
}

fn has_non_whitespace(value: &str, stop: &AtomicBool) -> Option<bool> {
    for character in value.chars() {
        if interrupted(stop) {
            return None;
        }
        if !character.is_whitespace() {
            return Some(true);
        }
    }
    (!interrupted(stop)).then_some(false)
}

fn project_entries<T: Clone>(
    entries: &BTreeMap<ModelObject, T>,
    objects: &BTreeMap<ModelObject, ModelObject>,
    stop: &AtomicBool,
) -> Option<BTreeMap<ModelObject, T>> {
    let mut projected = BTreeMap::new();
    for (object, value) in entries {
        if interrupted(stop) {
            return None;
        }
        if let Some(mapped) = objects.get(object) {
            projected.insert(*mapped, value.clone());
            if interrupted(stop) {
                return None;
            }
        }
    }
    Some(projected)
}

fn project_vec_entries<T: Clone>(
    entries: &BTreeMap<ModelObject, Vec<T>>,
    objects: &BTreeMap<ModelObject, ModelObject>,
    stop: &AtomicBool,
) -> Option<BTreeMap<ModelObject, Vec<T>>> {
    let mut projected = BTreeMap::new();
    for (object, values) in entries {
        if interrupted(stop) {
            return None;
        }
        if let Some(mapped) = objects.get(object) {
            let mut copied = Vec::with_capacity(values.len());
            for value in values {
                if interrupted(stop) {
                    return None;
                }
                copied.push(value.clone());
            }
            projected.insert(*mapped, copied);
        }
    }
    Some(projected)
}

fn project_frontend_ids(
    entries: &BTreeMap<(String, String), ModelObject>,
    objects: &BTreeMap<ModelObject, ModelObject>,
    stop: &AtomicBool,
) -> Option<BTreeMap<(String, String), ModelObject>> {
    let mut projected = BTreeMap::new();
    for (key, object) in entries {
        if interrupted(stop) {
            return None;
        }
        if let Some(mapped) = objects.get(object) {
            projected.insert(key.clone(), *mapped);
            if interrupted(stop) {
                return None;
            }
        }
    }
    Some(projected)
}

fn project_outputs(outputs: &[ModelObject], objects: &BTreeMap<ModelObject, ModelObject>, stop: &AtomicBool) -> Option<Vec<ModelObject>> {
    let mut projected = Vec::new();
    for object in outputs {
        if interrupted(stop) {
            return None;
        }
        if let Some(mapped) = objects.get(object) {
            projected.push(*mapped);
        }
    }
    Some(projected)
}

fn clone_annotations(entries: &BTreeMap<String, String>, stop: &AtomicBool) -> Option<BTreeMap<String, String>> {
    let mut copied = BTreeMap::new();
    for (key, value) in entries {
        if interrupted(stop) {
            return None;
        }
        copied.insert(key.clone(), value.clone());
        if interrupted(stop) {
            return None;
        }
    }
    Some(copied)
}

fn project_object_annotations(
    entries: &BTreeMap<ModelObject, BTreeMap<String, String>>,
    objects: &BTreeMap<ModelObject, ModelObject>,
    stop: &AtomicBool,
) -> Option<BTreeMap<ModelObject, BTreeMap<String, String>>> {
    let mut projected = BTreeMap::new();
    for (object, annotations) in entries {
        if interrupted(stop) {
            return None;
        }
        if let Some(mapped) = objects.get(object) {
            projected.insert(*mapped, clone_annotations(annotations, stop)?);
        }
    }
    Some(projected)
}
