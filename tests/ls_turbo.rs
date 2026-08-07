use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use qayd::frontends::xcsp::{run_to_with_options, Mode, RunOptions};

fn run_turbo(xml: &str, seed: u64) -> String {
    run_turbo_with(xml, seed, false)
}

fn run_turbo_verbose(xml: &str, seed: u64) -> String {
    run_turbo_with(xml, seed, true)
}

fn run_turbo_with(xml: &str, seed: u64, verbose: bool) -> String {
    let stop = Arc::new(AtomicBool::new(false));
    let stopper = Arc::clone(&stop);
    let handle = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(150));
        stopper.store(true, Ordering::Relaxed);
    });

    let mut out = Vec::new();
    run_to_with_options(xml, verbose, &stop, &mut out, RunOptions { seed, workers: 1, mode: Mode::Ls, ..RunOptions::default() }).unwrap();
    stop.store(true, Ordering::Relaxed);
    handle.join().unwrap();
    String::from_utf8(out).unwrap()
}

fn assert_supported_sat(out: &str) {
    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(!out.contains("c local unsupported"), "{out}");
}

#[test]
fn turbo_single_worker_respects_negative_extension() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0..1 </array></variables>
        <constraints>
          <extension>
            <list> x[0] x[1] </list>
            <conflicts> (0,0) </conflicts>
          </extension>
        </constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;

    let out = run_turbo(xml, 2);

    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("o 1"), "{out}");
    assert!(!out.contains("o 0"), "{out}");
    assert!(out.contains("v 0 1") || out.contains("v 1 0"), "{out}");
}

#[test]
fn turbo_orders_functional_chain_before_delta_scoring() {
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables>
          <var id="x"> 0..1 </var>
          <var id="y"> 0..1 </var>
          <var id="z"> 0..1 </var>
        </variables>
        <constraints>
          <intension> eq(z,y) </intension>
          <intension> eq(y,x) </intension>
        </constraints>
        <objectives><maximize>z</maximize></objectives>
      </instance>"#;

    let out = run_turbo(xml, 2);

    assert!(out.contains("s SATISFIABLE"), "{out}");
    assert!(out.contains("o 1"), "{out}");
    assert!(out.contains("v 1 1 1"), "{out}");
}

#[test]
fn turbo_supports_aggregate_objective_auxiliaries() {
    let maximum = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0..2 </array></variables>
        <constraints></constraints>
        <objectives><minimize type="maximum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(maximum, 3);
    assert_supported_sat(&out);
    assert!(out.contains("o 0"), "{out}");
    assert!(out.contains("v 0 0"), "{out}");

    let nvalues = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[3]"> 0..2 </array></variables>
        <constraints></constraints>
        <objectives><minimize type="nValues"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(nvalues, 4);
    assert_supported_sat(&out);
    assert!(out.contains("o 1"), "{out}");
    assert!(out.contains("v 0 0 0") || out.contains("v 1 1 1") || out.contains("v 2 2 2"), "{out}");
}

#[test]
fn turbo_supports_exception_order_channel_and_element_shapes() {
    let all_different_rows = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[4]"> 0..1 </array></variables>
        <constraints>
          <allDifferent>
            <list>x[0] x[1]</list>
            <list>x[2] x[3]</list>
          </allDifferent>
        </constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(all_different_rows, 5);
    assert_supported_sat(&out);
    assert!(out.contains("o 1"), "{out}");

    let all_different_except = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[3]"> 0..2 </array></variables>
        <constraints>
          <allDifferent>
            x[]
            <except>0</except>
          </allDifferent>
        </constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(all_different_except, 6);
    assert_supported_sat(&out);

    let precedence = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0..1 </array></variables>
        <constraints>
          <precedence>
            <list>x[]</list>
            <values>1 0</values>
          </precedence>
        </constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(precedence, 7);
    assert_supported_sat(&out);
    assert!(out.contains("o 1"), "{out}");
    assert!(out.contains("v 1 0"), "{out}");

    let covered_precedence = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0..1 </array></variables>
        <constraints>
          <precedence>
            <list>x[]</list>
            <values covered="true">0 1</values>
          </precedence>
        </constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(covered_precedence, 17);
    assert_supported_sat(&out);
    assert!(out.contains("o 1"), "{out}");
    assert!(out.contains("v 0 1"), "{out}");

    let channel = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0..1 </array></variables>
        <constraints><channel><list>x[]</list></channel></constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(channel, 8);
    assert_supported_sat(&out);
    assert!(out.contains("o 1"), "{out}");

    let channel_onehot = r#"
      <instance format="XCSP3" type="COP">
        <variables>
          <array id="b" size="[3]"> 0..1 </array>
          <var id="idx"> 0..2 </var>
        </variables>
        <constraints><channel><list>b[]</list><value>idx</value></channel></constraints>
        <objectives><minimize>idx</minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(channel_onehot, 9);
    assert_supported_sat(&out);

    let element_member = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0..1 </array></variables>
        <constraints><element><list>x[]</list><value>1</value></element></constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(element_member, 10);
    assert_supported_sat(&out);
    assert!(out.contains("o 1"), "{out}");
}

#[test]
fn turbo_supports_scheduling_packing_and_circuit_shapes() {
    let no_overlap = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="s" size="[2]"> 0..1 </array></variables>
        <constraints><noOverlap><origins>s[]</origins><lengths>1 1</lengths></noOverlap></constraints>
        <objectives><minimize type="sum"><list>s[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(no_overlap, 11);
    assert_supported_sat(&out);
    assert!(out.contains("o 1"), "{out}");

    let cumulative = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="s" size="[2]"> 0..1 </array></variables>
        <constraints>
          <cumulative>
            <origins>s[]</origins>
            <lengths>1 1</lengths>
            <heights>1 1</heights>
            <condition>(le,1)</condition>
          </cumulative>
        </constraints>
        <objectives><minimize type="sum"><list>s[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(cumulative, 12);
    assert_supported_sat(&out);
    assert!(out.contains("o 1"), "{out}");

    let bin_packing = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0..1 </array></variables>
        <constraints>
          <binPacking><list>x[]</list><sizes>2 1</sizes><condition>(le,2)</condition></binPacking>
        </constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(bin_packing, 13);
    assert_supported_sat(&out);
    assert!(out.contains("o 1"), "{out}");

    let circuit = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="succ" size="[3]"> 0..2 </array></variables>
        <constraints><circuit>succ[]</circuit></constraints>
        <objectives><minimize type="sum"><list>succ[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(circuit, 14);
    assert_supported_sat(&out);
    assert!(out.contains("o 3"), "{out}");
}

#[test]
fn turbo_supports_regular_and_mdd_shapes_after_presolve() {
    let regular = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0..1 </array></variables>
        <constraints>
          <regular>
            <list>x[]</list>
            <transitions>(q0,1,q1)(q1,0,q2)</transitions>
            <start>q0</start>
            <final>q2</final>
          </regular>
        </constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(regular, 15);
    assert_supported_sat(&out);
    assert!(out.contains("o 1"), "{out}");
    assert!(out.contains("v 1 0"), "{out}");

    let mdd_duplicate_label = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0..1 </array></variables>
        <constraints>
          <mdd>
            <list>x[]</list>
            <transitions>(r,0,a)(r,0,b)(b,0,t)</transitions>
          </mdd>
        </constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(mdd_duplicate_label, 18);
    assert_supported_sat(&out);
    assert!(out.contains("o 0"), "{out}");
    assert!(out.contains("v 0 0"), "{out}");

    let mdd = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[2]"> 0..1 </array></variables>
        <constraints>
          <mdd>
            <list>x[]</list>
            <transitions>(r,1,n1)(n1,0,t)</transitions>
          </mdd>
        </constraints>
        <objectives><minimize type="sum"><list>x[]</list></minimize></objectives>
      </instance>"#;
    let out = run_turbo_verbose(mdd, 16);
    assert_supported_sat(&out);
    assert!(out.contains("o 1"), "{out}");
    assert!(out.contains("v 1 0"), "{out}");
}
