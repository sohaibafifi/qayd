use std::collections::BTreeSet;
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
fn command_line_accepts_the_canonical_max_regret_selector() {
    let instance = TemporaryInstance::new("max-regret");

    let output = Command::new(env!("CARGO_BIN_EXE_qayd"))
        .args(["--verbose", "--search-phase", "0,1:max-regret:min"])
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
fn command_line_help_is_grouped_complete_and_terminal_friendly() {
    let output = Command::new(env!("CARGO_BIN_EXE_qayd")).arg("--help").output().unwrap();

    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let help = String::from_utf8(output.stdout).unwrap();
    for section in
        ["Usage:\n", "Arguments:\n", "General options:\n", "Search options:\n", "Search phase format:\n", "Linear relaxation options:\n"]
    {
        assert!(help.contains(section), "missing {section:?} in:\n{help}");
    }
    let documented = help
        .lines()
        .flat_map(|line| {
            let line = line.trim_start();
            if !line.starts_with('-') {
                return Vec::new();
            }
            line.split_whitespace()
                .take_while(|part| part.starts_with('-'))
                .map(|part| part.trim_end_matches(',').to_string())
                .collect::<Vec<_>>()
        })
        .collect::<BTreeSet<_>>();
    let expected = [
        "-h",
        "-v",
        "-t",
        "-p",
        "--help",
        "--verbose",
        "--time",
        "--seed",
        "--threads",
        "--mem-limit",
        "--core",
        "--all-solutions",
        "--ls",
        "--split",
        "--probe",
        "--lns",
        "--no-learn-csp",
        "--semantic-branching",
        "--search-phase",
        "--force-scope-reasons",
        "--semantic-minimization",
        "--check-inferences",
        "--deferred-explanations",
        "--explanation-metrics",
        "--extended-nogoods",
        "--cpip",
        "--unary-edge-finding",
        "--unary-edge-finding-metrics",
        "--incremental-cumulative",
        "--lifted-cumulative-explanations",
        "--cumulative-metrics",
        "--shared-pool-cap",
        "--linear-backend",
        "--lp-root-ms",
        "--lp-node-ms",
        "--lp-node-depth",
        "--lp-max-vars",
        "--lp-max-rows",
        "--lp-max-nonzeros",
        "--lp-min-coverage",
        "--lp-phase-max-vars",
        "--lp-route-ng-size",
        "--lp-route-max-labels",
        "--lp-route-dual-stabilization-percent",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<BTreeSet<_>>();
    assert_eq!(documented, expected, "documented option set differs:\n{help}");
    for detail in [
        "default: unlimited; --ls: 240",
        "default: 1; N >= 1",
        "MiB >= 1",
        "materialized COP objectives",
        "may allocate probing and LNS roles",
        "max-regret",
        "activity phases cannot use --no-learn-csp",
        "0 disables",
        "0..100",
        "1..16",
        "--manifest-path integrations/amthal/Cargo.toml",
    ] {
        assert!(help.contains(detail), "missing {detail:?} in:\n{help}");
    }
    assert!(help.lines().all(|line| line.len() <= 100), "help contains an overlong line:\n{help}");
}

#[test]
fn missing_instance_prints_the_same_help_to_stderr() {
    let output = Command::new(env!("CARGO_BIN_EXE_qayd")).output().unwrap();

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.starts_with("Qayd XCSP3 solver\n\nUsage:\n"), "{stderr}");
    assert!(stderr.contains("--search-phase <SPEC>"), "{stderr}");
}

#[test]
fn command_line_search_phases_reject_noncanonical_selectors() {
    let instance = TemporaryInstance::new("invalid");

    let output = Command::new(env!("CARGO_BIN_EXE_qayd")).args(["--search-phase", "0:first_fail:min"]).arg(&instance.0).output().unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("unknown variable selector"), "{stderr}");
}

#[test]
fn command_line_nogood_controls_reach_lcg_and_report_checker_coverage() {
    let instance = TemporaryInstance::new("nogoods");
    let output = Command::new(env!("CARGO_BIN_EXE_qayd"))
        .args(["--verbose", "--extended-nogoods", "--cpip", "--check-inferences", "--search-phase", "0,1:activity:min"])
        .arg(&instance.0)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("s SATISFIABLE"), "{stdout}");
    let scans = stdout.lines().find_map(|line| line.strip_prefix("c nogoods scans ")).unwrap();
    assert!(scans.split_whitespace().next().unwrap().parse::<u64>().unwrap() > 0, "{stdout}");
    assert!(stdout.contains("c inference lcg checked "), "{stdout}");
}

#[test]
fn command_line_cumulative_controls_are_independent_and_reach_the_native_engine() {
    let instance = TemporaryInstance::new("cumulative");
    std::fs::write(
        &instance.0,
        r#"<instance format="XCSP3" type="CSP">
             <variables><array id="s" size="[3]"> 0..3 </array></variables>
             <constraints><cumulative>
               <origins>s[]</origins><lengths>2 2 2</lengths><heights>1 1 1</heights><condition>(le,2)</condition>
             </cumulative></constraints>
           </instance>"#,
    )
    .unwrap();
    for (incremental, lifted) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_qayd"));
        command.args(["--verbose", "--cumulative-metrics", "--check-inferences"]);
        if incremental {
            command.arg("--incremental-cumulative");
        }
        if lifted {
            command.arg("--lifted-cumulative-explanations");
        }
        let output = command.arg(&instance.0).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("s SATISFIABLE"), "{stdout}");
        let prefix = format!("c cumulative incremental {incremental} lifted {lifted} calls ");
        let report = stdout.lines().find_map(|line| line.strip_prefix(&prefix)).unwrap_or_else(|| panic!("{stdout}"));
        assert!(report.split_whitespace().next().unwrap().parse::<u64>().unwrap() > 0, "{stdout}");
        assert!(stdout.contains("c inference checking checked "), "{stdout}");
    }
}
