#!/usr/bin/env python3
"""Put verified Qayd quality beside published Hexaly values, with caveats."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cluster.lib.campaign import load_plan, write
from cluster.lib.report import audit, vector


def compare(plan: dict, audited: dict) -> dict:
    references = plan["config"]["published_reference"]["jssp_incumbents"]
    groups = defaultdict(list)
    by_run = {(row["task"], row["solver"]): row for row in audited["records"]}
    for index, task in enumerate(plan["tasks"]):
        item = task["item"]
        for solver in task["solvers"]:
            row = by_run.get((index, solver), {
                "has_incumbent": False, "objective": None, "wall_seconds": task["seconds"] + 1,
                "budget": task["seconds"], "seed": task["seed"], "details": {},
            })
            key = (item["track"], solver, Path(item["path"]).stem)
            groups[key].append((row, item))
    jssp, cvrp, rcpsp, missing_bks = [], defaultdict(list), defaultdict(list), []
    for (track, solver, name), group in sorted(groups.items()):
        def eligible(row):
            return row["has_incumbent"] and row["wall_seconds"] <= row["budget"] and not row.get("timed_out", False)
        values = [vector(row["objective"])[0] for row, _ in group
                  if eligible(row) and vector(row["objective"]) is not None]
        item = group[0][1]
        if track == "jssp":
            reference = references.get(name)
            if reference is None or any(row["budget"] != 600 for row, _ in group):
                raise ValueError(f"not a matched 600-second Large-TA reference: {name}")
            jssp.append({"instance": name, "solver": solver, "verified_runs": len(values),
                         "expected_runs": len(group), "published_hexaly_makespan": reference,
                         "late_verified_runs": sum(row["has_incumbent"] and not eligible(row) for row, _ in group),
                         "qayd_median_makespan": statistics.median(values) if values else None,
                         "qayd_best_makespan": min(values) if values else None,
                         "median_gap_to_published_percent": 100 * (statistics.median(values) - reference) / reference if values else None})
        elif track == "cvrp":
            if any(row["budget"] != 60 for row, _ in group):
                raise ValueError("Hexaly CVRP reference requires 60 seconds")
            bks = item.get("reference_bks") or next((row.get("details", {}).get("best_known") for row, _ in group
                        if row.get("details", {}).get("best_known")), None)
            if not bks or bks <= 0:
                missing_bks.append(name)
                continue
            n = item["customers"]
            label = next(label for low, high, label in (
                (100,200,"100-200"), (200,400,"200-400"), (400,600,"400-600"),
                (600,800,"600-800"), (800,1001,"800-1000")) if low <= n < high)
            cvrp[(solver, label)].append({"instance": name, "verified_runs": len(values),
                "expected_runs": len(group),
                "bks": bks, "mean_seed_gap": statistics.mean(100*(v-bks)/bks for v in values) if values else None})
        elif track == "rcpsp":
            if any(row["budget"] != 60 for row, _ in group):
                raise ValueError("Hexaly RCPSP reference requires 60 seconds")
            bks = item.get("reference_bks")
            if not bks or bks <= 0:
                missing_bks.append(name)
                continue
            for row, _ in group:
                value = vector(row["objective"]) if eligible(row) else None
                rcpsp[(solver, row["seed"])].append(100*(value[0]-bks)/bks if value else None)
    published_cvrp = {"100-200": .1, "200-400": .5, "400-600": .6, "600-800": .8, "800-1000": .9}
    cvrp_rows = [{"solver": solver, "size_bin": label, "instances": len(group),
                  "instances_with_all_seeds": sum(r["verified_runs"] == r["expected_runs"] for r in group),
                  "qayd_mean_instance_gap": statistics.mean(r["mean_seed_gap"] for r in group if r["mean_seed_gap"] is not None)
                  if any(r["mean_seed_gap"] is not None for r in group) else None,
                  "published_hexaly_mean_gap": published_cvrp[label], "per_instance": group}
                 for (solver, label), group in sorted(cvrp.items())]
    rcpsp_rows = [{"solver": solver, "seed": seed, "denominator": 480, "observed": len(gaps),
                   "verified": sum(g is not None for g in gaps),
                   "threshold_rates_percent": {str(t): 100*sum(g is not None and g < t for g in gaps)/480 for t in (10,5,2.5,1)}}
                  for (solver, seed), gaps in sorted(rcpsp.items())]
    return {"kind": "contextual comparison to fixed vendor publication, not same-machine performance",
            "source_pages": plan["config"].get("published_reference", {}).get("pages"),
            "coverage_ready": audited["ready"], "missing_bks": missing_bks,
            "jssp": jssp, "cvrp": cvrp_rows, "rcpsp": rcpsp_rows,
            "published_rcpsp_rates": {"10":100,"5":90,"2.5":64,"1":42},
            "unresolved": ["Published seed and thread counts are not specified.",
                           "Identical routing BKS snapshots and size-bin boundaries are not established.",
                           "RCPLIB v2.4 BKS equivalence to the vendor snapshot is not established.",
                           "VRPTW distance/fleet and rounding conventions are not aligned; no numeric comparison.",
                           "Incomplete or unverified solutions stay in coverage and threshold denominators.",
                           "Conditional mean gaps must be accompanied by verified coverage."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    report = compare(plan, audit(plan, args.plan.resolve().parent))
    write(args.plan.parent / "hexaly-comparison.json", report)
    print(args.plan.parent / "hexaly-comparison.json")


if __name__ == "__main__":
    main()
