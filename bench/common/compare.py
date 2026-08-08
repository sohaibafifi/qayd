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


def is_solved(status, d):
    """Did the solver fully settle this instance?

    For a decision/CSP instance (no objective direction) SATISFIABLE is a full
    answer. For an optimization (COP) instance it is NOT: SATISFIABLE means an
    incumbent was found but optimality was not proved -- effectively a timeout on
    the optimization question. Only OPTIMUM (proved optimal) or UNSAT (proved
    infeasible) count as solved there.
    """
    if status in ("OPTIMUM", "UNSAT"):
        return True
    if status == "SAT":
        return d == "?"
    return False


def found_incumbent(status):
    """Feasible solution in hand (proved optimal or not)."""
    return status in ("SAT", "OPTIMUM")


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
        # COP-aware: an unproved COP incumbent (SATISFIABLE) is charged as
        # unsolved, so a solver that gives up early with an incumbent is not
        # rewarded as if it had proved the optimum.
        if is_solved(r["status"], r["dir"]) and not r["to"]:
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
    ap.add_argument("--details", type=int, default=0, metavar="N",
                    help="list per-instance regressions: lost/gained instances, "
                    "top-N slowdowns/speedups among solved-by-both, objective regressions")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B))
    na, nb = args.name_a, args.name_b

    # PAR2 charges 2*timeout per unsolved instance: a --timeout below the runs'
    # real budget silently skews the ranking, so flag the mismatch loudly.
    max_t = max((r["time"] for r in list(A.values()) + list(B.values())), default=0.0)
    if max_t > args.timeout * 1.1:
        print(f"  !! --timeout {args.timeout} looks wrong: recorded times reach "
              f"{max_t:.0f}s — PAR2 is unreliable; pass the runs' actual timeout")

    def cdir(i):
        return A[i]["dir"] if A[i]["dir"] != "?" else B[i]["dir"]

    is_cop = any(cdir(i) in ("min", "max") for i in common)
    solved_a = sum(is_solved(A[i]["status"], cdir(i)) for i in common)
    solved_b = sum(is_solved(B[i]["status"], cdir(i)) for i in common)
    vbs = sum(is_solved(A[i]["status"], cdir(i)) or is_solved(B[i]["status"], cdir(i)) for i in common)
    incumbent_a = sum(found_incumbent(A[i]["status"]) for i in common)
    incumbent_b = sum(found_incumbent(B[i]["status"]) for i in common)

    contradictions, obj_tie, a_better, a_worse = [], 0, 0, 0
    a_faster = b_faster = 0
    only_a = only_b = both = neither = 0
    lost, gained, deltas, obj_regressions = [], [], [], []
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
        oa, ob = is_solved(sa, d), is_solved(sb, d)
        both += oa and ob
        only_a += oa and not ob
        only_b += ob and not oa
        neither += not oa and not ob
        if oa and not ob:
            gained.append((i, A[i]["time"]))
        if ob and not oa:
            lost.append((i, B[i]["time"]))
        if oa and ob:
            deltas.append((A[i]["time"] - B[i]["time"], i, A[i]["time"], B[i]["time"]))
            if A[i]["time"] < B[i]["time"]:
                a_faster += 1
            elif B[i]["time"] < A[i]["time"]:
                b_faster += 1
        # Objective quality is compared wherever BOTH hold an incumbent -- for a
        # COP that means SATISFIABLE or OPTIMUM, not just proved-by-both, else
        # every unproved-but-improved/regressed incumbent is invisible. `d` is the
        # instance's own min/max direction, so max and min are never mixed.
        if found_incumbent(sa) and found_incumbent(sb):
            c = better(A[i]["obj"], B[i]["obj"], d)
            if c == 1:
                a_better += 1
            elif c == -1:
                a_worse += 1
                obj_regressions.append((i, A[i]["obj"], B[i]["obj"], d))
            elif A[i]["obj"] is not None:
                obj_tie += 1

    print(f"instances compared: {len(common)}")
    if is_cop:
        print("  (optimization instances present: 'solved' = proved optimum/infeasible; SATISFIABLE is an incumbent, not a solve)")
    print(f"  {na} solved : {solved_a}")
    print(f"  {nb} solved : {solved_b}")
    if is_cop:
        print(f"  {na} incumbent found : {incumbent_a}")
        print(f"  {nb} incumbent found : {incumbent_b}")
    print(f"  VBS solved  : {vbs}   (either solver)")
    print(f"  solved by both        : {both}")
    print(f"  only {na}             : {only_a}")
    print(f"  only {nb}             : {only_b}")
    print(f"  neither               : {neither}")
    print(f"  PAR2 {na} : {par2(A, args.timeout):.1f}  (lower better)")
    print(f"  PAR2 {nb} : {par2(B, args.timeout):.1f}")
    print(f"  speed (solved-by-both): {na} faster {a_faster} | {nb} faster {b_faster}")
    if obj_tie or a_better or a_worse:
        print(f"  objective quality (incumbent by both): same {obj_tie} | {na} better {a_better} | {na} worse {a_worse}")
    print(f"  CONTRADICTIONS (soundness bug!) : {len(contradictions)}")
    for i, sa, sb in contradictions:
        print(f"    !! {i}: {na}={sa} {nb}={sb}")

    if args.details:
        n = args.details
        if lost:
            print(f"  lost by {na} (solved only by {nb}):")
            for i, t in sorted(lost, key=lambda x: x[1]):
                print(f"    - {i}  ({nb}: {t:.1f}s)")
        if gained:
            print(f"  gained by {na}:")
            for i, t in sorted(gained, key=lambda x: x[1]):
                print(f"    + {i}  ({na}: {t:.1f}s)")
        slow = sorted((d for d in deltas if d[0] > 0), reverse=True)[:n]
        if slow:
            print(f"  top slowdowns for {na} (solved-by-both, ordered by lost seconds):")
            for dt, i, ta, tb in slow:
                print(f"    {i}: {tb:.1f}s -> {ta:.1f}s  (+{dt:.1f}s)")
        fast = sorted(d for d in deltas if d[0] < 0)[:n]
        if fast:
            print(f"  top speedups for {na}:")
            for dt, i, ta, tb in fast:
                print(f"    {i}: {tb:.1f}s -> {ta:.1f}s  ({dt:.1f}s)")
        if obj_regressions:
            print(f"  objective regressions for {na} (incumbent by both):")
            for i, oa_v, ob_v, dd in obj_regressions[:n]:
                print(f"    {i} [{dd}]: {na}={oa_v} {nb}={ob_v}")


if __name__ == "__main__":
    main()
