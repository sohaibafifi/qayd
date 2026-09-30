use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::certificate::{CertificateClaim, CertificateError, CertificateOptions};
use crate::orchestrator::{certificate::run_checker, SolveBudget};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn directory() -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("qayd-certificate-runtime-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir(&path).unwrap();
    path
}

#[test]
fn certificate_missing_checker_cannot_accept() {
    let path = directory();
    let error = run_checker(
        path.join("nonexistent-interpreter").to_str().unwrap(),
        &path.join("model"),
        &path.join("proof"),
        &"0".repeat(64),
        &"0".repeat(64),
        CertificateClaim::Unsatisfiable,
        &CertificateOptions::new(&path),
        &SolveBudget::new(None),
        "VERIFIED\n",
    )
    .unwrap_err();
    assert!(matches!(error, CertificateError::CheckerUnavailable(_)), "{error:?}");
    std::fs::remove_dir_all(path).unwrap();
}

#[cfg(unix)]
fn mock_checker(path: &std::path::Path, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let script = path.join("checker");
    std::fs::write(&script, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
    script
}

#[test]
#[cfg(unix)]
fn certificate_checker_success_without_exact_ack_cannot_accept() {
    let path = directory();
    let checker = mock_checker(&path, "exit 0");
    let error = run_checker(
        checker.to_str().unwrap(),
        &path.join("model"),
        &path.join("proof"),
        &"0".repeat(64),
        &"0".repeat(64),
        CertificateClaim::Unsatisfiable,
        &CertificateOptions::new(&path),
        &SolveBudget::new(None),
        "VERIFIED\n",
    )
    .unwrap_err();
    assert!(matches!(error, CertificateError::CheckerRejected(_)), "{error:?}");
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
#[cfg(unix)]
fn certificate_checker_deadline_cannot_accept() {
    let path = directory();
    // exec ensures the child killed by the wrapper is the sleeper itself.
    let checker = mock_checker(&path, "exec sleep 10");
    let error = run_checker(
        checker.to_str().unwrap(),
        &path.join("model"),
        &path.join("proof"),
        &"0".repeat(64),
        &"0".repeat(64),
        CertificateClaim::Unsatisfiable,
        &CertificateOptions::new(&path),
        &SolveBudget::new(Some(Duration::from_millis(40))),
        "VERIFIED\n",
    )
    .unwrap_err();
    assert_eq!(error, CertificateError::Interrupted);
    std::fs::remove_dir_all(path).unwrap();
}
