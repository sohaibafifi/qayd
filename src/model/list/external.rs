use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, OnceLock, RwLock};
use std::time::Duration;

type Evaluator = dyn Fn(&[i64]) -> Result<i64, String> + Send + Sync + 'static;

const EXTERNAL_CALL_POLL: Duration = Duration::from_millis(2);
const MAX_DETACHED_EXTERNAL_CALLS: usize = 4;

static ACTIVE_EXTERNAL_CALLS: AtomicUsize = AtomicUsize::new(0);

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
    let evaluator = external_function(name)?;
    evaluator(args).map_err(|error| format!("external function '{name}' failed: {error}"))
}

/// Invoke an external evaluator without allowing a non-cooperative callback to
/// hold final canonical replay beyond its cancellation boundary. `Err(())`
/// means the call was interrupted or the bounded worker pool was exhausted;
/// evaluator failures remain ordinary expression failures in the inner result.
pub(crate) fn call_external_function_interruptible(name: &str, args: &[i64], stop: &AtomicBool) -> Result<Result<i64, String>, ()> {
    if stop.load(Ordering::Acquire) {
        return Err(());
    }
    let evaluator = match external_function(name) {
        Ok(evaluator) => evaluator,
        Err(error) => return Ok(Err(error)),
    };
    let Some(permit) = ExternalCallPermit::acquire() else {
        return Err(());
    };
    let values = args.to_vec();
    let name = name.to_string();
    let (sender, receiver) = mpsc::sync_channel(1);
    let worker = std::thread::Builder::new().name("qayd-external-eval".to_string()).spawn(move || {
        let _permit = permit;
        let result = evaluator(&values).map_err(|error| format!("external function '{name}' failed: {error}"));
        let _ = sender.send(result);
    });
    if worker.is_err() {
        return Err(());
    }
    loop {
        match receiver.recv_timeout(EXTERNAL_CALL_POLL) {
            Ok(result) => return Ok(result),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if stop.load(Ordering::Acquire) {
                    return Err(());
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Ok(Err("external function worker terminated without a result".to_string()));
            }
        }
    }
}

fn external_function(name: &str) -> Result<Arc<Evaluator>, String> {
    registry()
        .read()
        .map_err(|_| "external function registry is poisoned".to_string())?
        .get(name)
        .cloned()
        .ok_or_else(|| format!("external function '{name}' is not registered"))
}

struct ExternalCallPermit;

impl ExternalCallPermit {
    fn acquire() -> Option<Self> {
        ACTIVE_EXTERNAL_CALLS
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| (active < MAX_DETACHED_EXTERNAL_CALLS).then_some(active + 1))
            .ok()
            .map(|_| Self)
    }
}

impl Drop for ExternalCallPermit {
    fn drop(&mut self) {
        ACTIVE_EXTERNAL_CALLS.fetch_sub(1, Ordering::AcqRel);
    }
}
