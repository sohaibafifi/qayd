use std::path::PathBuf;
use std::process::Command;

struct TemporaryInstance(PathBuf);

impl TemporaryInstance {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!("qayd-search-policy-{}-{label}.xml", std::process::id()));
        std::fs::write(
            &path,
            r#"<instance format="XCSP3" type="CSP">
                 <variables>
                   <var id="wide"> 0..2 </var>
                   <var id="narrow"> 0..1 </var>
                 </variables>
                 <constraints><intension> ne(wide,narrow) </intension></constraints>
               </instance>"#,
        )
        .unwrap();
        Self(path)
    }
}

impl Drop for TemporaryInstance {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn command_line_search_phases_reach_the_exact_engine() {
    let instance = TemporaryInstance::new("valid");

    let output = Command::new(env!("CARGO_BIN_EXE_qayd"))
        .args(["--verbose", "--search-phase", "0,1:input-order:min"])
        .arg(&instance.0)
        .output()
        .unwrap();

    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("c search phases 1"), "{stdout}");
    assert!(stdout.contains("v 0 1"), "{stdout}");
    assert!(stdout.contains("s SATISFIABLE"), "{stdout}");
}

#[test]
fn command_line_search_phases_reject_noncanonical_selectors() {
    let instance = TemporaryInstance::new("invalid");

    let output = Command::new(env!("CARGO_BIN_EXE_qayd")).args(["--search-phase", "0:first_fail:min"]).arg(&instance.0).output().unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("unknown variable selector"), "{stderr}");
}
