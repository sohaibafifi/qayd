//! Branch-and-bound must reach the true optimum on `Objective::Linear` after the
//! objective-bound rewrite that replaced per-incumbent constraint accumulation
//! with a single monotonically-tightening propagator.
//!
//! An XCSP `sum` objective only routes to `Objective::Linear` (rather than a
//! materialized objective variable) when its value span exceeds ~1e6, so these
//! instances use a large leading coefficient to force that path for both the
//! minimizing (`sum <= c`) and maximizing (`-sum <= -c`) forms of the propagator.

use qayd::frontends::xcsp::run;

#[test]
fn linear_objective_minimize_reaches_optimum() {
    // Minimize 1000000*x0 + x1 + x2, each in 0..2, subject to x0+x1+x2 >= 2.
    // Coefficient span = 2_000_004 > 1e6, so the objective stays a `Linear`.
    // Optimum avoids the huge coefficient: x0=0, x1+x2=2 -> 2.
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[3]"> 0..2 </array></variables>
        <constraints>
          <sum><list>x[]</list><condition>(ge,2)</condition></sum>
        </constraints>
        <objectives>
          <minimize type="sum"><list>x[]</list><coeffs>1000000 1 1</coeffs></minimize>
        </objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 2"), "{out}");
}

#[test]
fn linear_objective_maximize_reaches_optimum() {
    // Maximize 1000000*x0 + x1 + x2, each in 0..2, subject to x0+x1+x2 <= 3.
    // Exercises the negated `-sum <= -c` branch. Optimum grabs the huge
    // coefficient: x0=2, then one unit of budget left -> x1=1 -> 2_000_001.
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables><array id="x" size="[3]"> 0..2 </array></variables>
        <constraints>
          <sum><list>x[]</list><condition>(le,3)</condition></sum>
        </constraints>
        <objectives>
          <maximize type="sum"><list>x[]</list><coeffs>1000000 1 1</coeffs></maximize>
        </objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 2000001"), "{out}");
}

#[test]
fn linear_objective_many_improvements_reaches_optimum() {
    // A loose knapsack with wide-span profits: the incumbent improves in many
    // steps, all handled by the single reusable objective-bound propagator.
    // Maximize 1000000*a + 100000*b + 10000*c + 1000*d + 100*e (all 0/1),
    // subject to a+b+c+d+e <= 3. Best picks the three largest coeffs:
    // a+b+c = 1_110_000.
    let xml = r#"
      <instance format="XCSP3" type="COP">
        <variables>
          <var id="a"> 0..1 </var>
          <var id="b"> 0..1 </var>
          <var id="c"> 0..1 </var>
          <var id="d"> 0..1 </var>
          <var id="e"> 0..1 </var>
        </variables>
        <constraints>
          <sum><list>a b c d e</list><condition>(le,3)</condition></sum>
        </constraints>
        <objectives>
          <maximize type="sum">
            <list>a b c d e</list>
            <coeffs>1000000 100000 10000 1000 100</coeffs>
          </maximize>
        </objectives>
      </instance>"#;
    let out = run(xml).unwrap();
    assert!(out.contains("s OPTIMUM FOUND"), "{out}");
    assert!(out.contains("o 1110000"), "{out}");
}
