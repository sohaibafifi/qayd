//! Specialized solver engines.
//!
//! Frontends build a semantic model and call the crate-level solve entry point.
//! Only the orchestrator compiles and executes these engines.

use crate::model::list::CollectionModel;
use crate::model::{CompiledCollection, Model};
use crate::orchestrator::SolveRequest;

/// Typed result of a physical compiler that declined or could not finish a
/// semantic model. Capability rejection is distinct from cancellation and
/// from a malformed physical lowering.
#[derive(Debug)]
pub(crate) enum CompileFailure {
    Unsupported { code: &'static str, detail: &'static str },
    Interrupted { phase: &'static str },
    Invalid { reason: String },
}

/// Sealed collection-compilation input shared by the orchestrator and the
/// specialized collection backends. The semantic model and request remain the
/// source contract; the compiled collection IR is reused to avoid reparsing or
/// duplicating large allocations.
pub(crate) struct CollectionCompileContext<'a> {
    semantic: &'a Model,
    request: &'a SolveRequest,
    compiled: &'a CompiledCollection,
}

impl<'a> CollectionCompileContext<'a> {
    pub(crate) fn new(semantic: &'a Model, request: &'a SolveRequest, compiled: &'a CompiledCollection) -> Self {
        Self { semantic, request, compiled }
    }

    pub(crate) fn semantic(&self) -> &'a Model {
        self.semantic
    }

    pub(crate) fn request(&self) -> &'a SolveRequest {
        self.request
    }

    pub(crate) fn physical(&self) -> &'a CollectionModel {
        self.compiled.as_model()
    }
}

#[allow(dead_code)]
pub(crate) mod schedule;

pub(crate) mod cp;

#[allow(dead_code)]
pub(crate) mod dual;

#[allow(dead_code)]
pub(crate) mod list_exact;

#[allow(dead_code)]
pub(crate) mod ls;

#[allow(dead_code)]
pub(crate) mod sat;

pub(crate) mod routing;
