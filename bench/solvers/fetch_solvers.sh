#!/usr/bin/env bash
# Fetch the literature baseline solvers used by the benchmark pipelines.
#   PB       : Sat4j PB (Java)        -> s4j-{pb,core}.jar   (Maven Central)
#   CSP/COP  : Choco XCSP3 (Java)     -> choco.jar           (Maven Central)
#   SAT      : CaDiCaL                -> expected on PATH (brew install cadical
#              or build from https://github.com/arminbiere/cadical)
set -euo pipefail
cd "$(dirname "$0")"

S4J=2.3.6
mvn=https://repo1.maven.org/maven2
curl -sL "$mvn/org/ow2/sat4j/org.ow2.sat4j.pb/$S4J/org.ow2.sat4j.pb-$S4J.jar"     -o s4j-pb.jar
curl -sL "$mvn/org/ow2/sat4j/org.ow2.sat4j.core/$S4J/org.ow2.sat4j.core-$S4J.jar" -o s4j-core.jar

CHOCO=6.0.1
curl -sL "$mvn/org/choco-solver/choco-parsers/$CHOCO/choco-parsers-$CHOCO-jar-with-dependencies.jar" -o choco.jar

echo "sat4j : $(du -h s4j-pb.jar s4j-core.jar | tr '\n' ' ')"
echo "choco : $(du -h choco.jar | cut -f1)"
command -v cadical >/dev/null && echo "cadical: $(cadical --version)" || echo "cadical: NOT on PATH (brew install cadical)"
