"""Compare qayd vs a baseline solver from two run.py CSVs.

Reports, over the common instance set:
  - solved counts per solver (SAT/UNSAT/OPTIMUM = "solved"),
  - CONTRADICTIONS: one solver says SAT/OPTIMUM, the other UNSAT -> a real
    soundness bug in one of them (printed loudly),
  - objective agreement for optimization runs (min/max aware): equal optima,
    or qayd worse/better incumbent,
  - PAR2 score per solver (sum of runtimes; unsolved counts 2*timeout) -- the
    standard SAT-competition ranking metric, lower is better,
  - VBS (virtual best solver) solved count -- union of what either solved,
  - per-instance speed winner among solved-by-both.

Usage: python compare.py qayd.csv baseline.csv --timeout 10 \
                          --name-a qayd --name-b cadical
"""
import argparse
import csv

SOLVED = {"SAT", "UNSAT", "OPTIMUM"}


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["instance"]] = {
                "status": r["status"],
                "obj": int(r["obj"]) if r.get("obj") not in (None, "") else None,
                "time": float(r["time"]) if r.get("time") else 0.0,
                "to": r.get("timedout") == "1",
                "dir": r.get("dir", "?"),
            }
    return rows


def par2(rows, timeout):
    s = 0.0
    for r in rows.values():
        if r["status"] in SOLVED and not r["to"]:
            s += r["time"]
        else:
            s += 2 * timeout
    return s


def better(oa, ob, d):
    """+1 if a's objective strictly better than b's for direction d, -1 worse, 0 tie."""
    if oa is None or ob is None or oa == ob:
        return 0
    if d == "max":
        return 1 if oa > ob else -1
    return 1 if oa < ob else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="qayd CSV")
    ap.add_argument("b", help="baseline CSV")
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--name-a", default="qayd")
    ap.add_argument("--name-b", default="baseline")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B))
    na, nb = args.name_a, args.name_b

    solved_a = sum(A[i]["status"] in SOLVED for i in common)
    solved_b = sum(B[i]["status"] in SOLVED for i in common)
    vbs = sum((A[i]["status"] in SOLVED) or (B[i]["status"] in SOLVED) for i in common)

    contradictions, obj_tie, a_better, a_worse = [], 0, 0, 0
    a_faster = b_faster = 0
    only_a = only_b = both = neither = 0
    for i in common:
        sa, sb = A[i]["status"], B[i]["status"]
        oaj, obj = A[i]["obj"], B[i]["obj"]
        d = A[i]["dir"] if A[i]["dir"] != "?" else B[i]["dir"]
        hard = {"SAT", "OPTIMUM"}
        if (sa in hard and sb == "UNSAT") or (sb in hard and sa == "UNSAT"):
            contradictions.append((i, sa, sb))
        elif sa == "OPTIMUM" and sb == "OPTIMUM" and oaj is not None and obj is not None and oaj != obj:
            # both claim provable optimum but disagree -> one is unsound
            contradictions.append((i, f"OPTIMUM={oaj}", f"OPTIMUM={obj}"))
        elif d != "?" and sa == "OPTIMUM" and sb == "SAT" and better(obj, oaj, d) == 1:
            # b has a feasible solution strictly better than a's claimed optimum
            contradictions.append((i, f"OPTIMUM={oaj}", f"SAT={obj}"))
        elif d != "?" and sb == "OPTIMUM" and sa == "SAT" and better(oaj, obj, d) == 1:
            contradictions.append((i, f"SAT={oaj}", f"OPTIMUM={obj}"))
        oa, ob = A[i]["status"] in SOLVED, B[i]["status"] in SOLVED
        both += oa and ob
        only_a += oa and not ob
        only_b += ob and not oa
        neither += not oa and not ob
        if oa and ob:
            if A[i]["time"] < B[i]["time"]:
                a_faster += 1
            elif B[i]["time"] < A[i]["time"]:
                b_faster += 1
            c = better(A[i]["obj"], B[i]["obj"], A[i]["dir"])
            if c == 1:
                a_better += 1
            elif c == -1:
                a_worse += 1
            elif A[i]["obj"] is not None:
                obj_tie += 1

    print(f"instances compared: {len(common)}")
    print(f"  {na} solved : {solved_a}")
    print(f"  {nb} solved : {solved_b}")
    print(f"  VBS solved  : {vbs}   (either solver)")
    print(f"  solved by both        : {both}")
    print(f"  only {na}             : {only_a}")
    print(f"  only {nb}             : {only_b}")
    print(f"  neither               : {neither}")
    print(f"  PAR2 {na} : {par2(A, args.timeout):.1f}  (lower better)")
    print(f"  PAR2 {nb} : {par2(B, args.timeout):.1f}")
    print(f"  speed (solved-by-both): {na} faster {a_faster} | {nb} faster {b_faster}")
    if obj_tie or a_better or a_worse:
        print(f"  optimization objective: equal-optimum {obj_tie} | {na} better {a_better} | {na} worse {a_worse}")
    print(f"  CONTRADICTIONS (soundness bug!) : {len(contradictions)}")
    for i, sa, sb in contradictions:
        print(f"    !! {i}: {na}={sa} {nb}={sb}")


if __name__ == "__main__":
    main()
