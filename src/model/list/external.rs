use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

type Evaluator = dyn Fn(&[i64]) -> Result<i64, String> + Send + Sync + 'static;

fn registry() -> &'static RwLock<HashMap<String, Arc<Evaluator>>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, Arc<Evaluator>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register a deterministic, side-effect-free function usable by collection
/// lambda expressions. Names are process-global and cannot be silently
/// replaced, which keeps a compiled model stable across threads and solves.
pub fn register_external_function(
    name: impl Into<String>,
    evaluator: impl Fn(&[i64]) -> Result<i64, String> + Send + Sync + 'static,
) -> Result<(), String> {
    let name = name.into();
    if name.trim().is_empty() {
        return Err("external function name cannot be empty".to_string());
    }
    let mut functions = registry().write().map_err(|_| "external function registry is poisoned".to_string())?;
    if functions.contains_key(&name) {
        return Err(format!("external function '{name}' is already registered"));
    }
    functions.insert(name, Arc::new(evaluator));
    Ok(())
}

pub fn external_function_registered(name: &str) -> bool {
    registry().read().is_ok_and(|functions| functions.contains_key(name))
}

pub(crate) fn call_external_function(name: &str, args: &[i64]) -> Result<i64, String> {
    let evaluator = registry()
        .read()
        .map_err(|_| "external function registry is poisoned".to_string())?
        .get(name)
        .cloned()
        .ok_or_else(|| format!("external function '{name}' is not registered"))?;
    evaluator(args).map_err(|error| format!("external function '{name}' failed: {error}"))
}
