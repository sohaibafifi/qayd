#!/usr/bin/env python3
"""Build the reproducible VRPTW fleet-first qualification report."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(5))
FEASIBLE = {"SAT", "SATISFIABLE", "FEASIBLE", "OPTIMAL", "OPTIMUM"}
TARGETS = {
    "C101": {"exact": (10, 8273)},
    "C201": {"exact": (3, 5891)},
    "R101": {"fleet": 19},
    "R201": {"fleet": 4},
    "RC101": {"fleet": 16},
    "RC201": {"fleet": 4},
}


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
                if isinstance(value, dict):
                    records.append(value)
    return records


def finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def objectives(record: dict[str, Any]) -> tuple[int, int] | None:
    values = record.get("objectives")
    if not isinstance(values, list) or len(values) < 2:
        return None
    fleet, distance = values[:2]
    if isinstance(fleet, bool) or isinstance(distance, bool) or not isinstance(fleet, int) or not isinstance(distance, int):
        return None
    return fleet, distance


def verified(record: dict[str, Any]) -> bool:
    return bool(record.get("status") in FEASIBLE and record.get("verified") is True and objectives(record) is not None)


def canonical_instance(record: dict[str, Any]) -> str:
    name = record.get("instance")
    if isinstance(name, str) and name:
        return name.upper()
    path = record.get("instance_path")
    return Path(path).stem.upper() if isinstance(path, str) else ""


def seed(record: dict[str, Any]) -> int | None:
    value = record.get("seed")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def at_ten_seconds(record: dict[str, Any]) -> bool:
    return finite(record.get("checkpoint_seconds")) == 10.0


def select_runs(records: Sequence[dict[str, Any]], solver: str) -> tuple[dict[tuple[str, int], dict[str, Any]], int]:
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates = 0
    for record in records:
        run_seed = seed(record)
        instance = canonical_instance(record)
        if record.get("solver") != solver or record.get("problem") != "cvrptw" or run_seed is None or not at_ten_seconds(record):
            continue
        key = instance, run_seed
        duplicates += int(key in selected)
        selected[key] = record
    return selected, duplicates


def sentinel_rows(qayd: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for instance, target in TARGETS.items():
        runs = [qayd.get((instance, run_seed)) for run_seed in SEEDS]
        valid = [record for record in runs if record is not None and verified(record)]
        values = [objectives(record) for record in valid]
        fleet = [value[0] for value in values if value is not None]
        distance = [value[1] for value in values if value is not None]
        if "exact" in target:
            target_passes = sum(value == target["exact"] for value in values)
        else:
            target_passes = sum(value[0] <= target["fleet"] for value in values if value is not None)
        rows.append({
            "instance": instance,
            "runs": len(valid),
            "target": list(target.get("exact", (target.get("fleet"),))),
            "target_passes": target_passes,
            "fleet_median": statistics.median(fleet) if fleet else None,
            "distance_median": statistics.median(distance) if distance else None,
            "objectives": [list(value) for value in values if value is not None],
            "pass": len(valid) == len(SEEDS) and target_passes == len(SEEDS),
        })
    return rows


def distance_comparison(
    qayd: dict[tuple[str, int], dict[str, Any]],
    hexaly: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    paired_gaps = []
    comparable_instance_gaps = []
    complete_instances = 0
    for instance in TARGETS:
        pairs = []
        qayd_values = []
        hexaly_values = []
        for run_seed in SEEDS:
            left = qayd.get((instance, run_seed))
            right = hexaly.get((instance, run_seed))
            if left is None or right is None or not verified(left) or not verified(right):
                continue
            left_value = objectives(left)
            right_value = objectives(right)
            if left_value is None or right_value is None:
                continue
            qayd_values.append(left_value)
            hexaly_values.append(right_value)
            if left_value[0] == right_value[0] and right_value[1] > 0:
                gap = 100.0 * (left_value[1] - right_value[1]) / right_value[1]
                pairs.append(gap)
                paired_gaps.append(gap)
        complete_instances += int(len(qayd_values) == len(SEEDS) and len(hexaly_values) == len(SEEDS))
        qayd_fleet = statistics.median(value[0] for value in qayd_values) if qayd_values else None
        hexaly_fleet = statistics.median(value[0] for value in hexaly_values) if hexaly_values else None
        median_gap = None
        if qayd_fleet is not None and qayd_fleet == hexaly_fleet:
            qayd_distance = statistics.median(value[1] for value in qayd_values if value[0] == qayd_fleet)
            hexaly_distance = statistics.median(value[1] for value in hexaly_values if value[0] == hexaly_fleet)
            if hexaly_distance > 0:
                median_gap = 100.0 * (qayd_distance - hexaly_distance) / hexaly_distance
                comparable_instance_gaps.append(median_gap)
        rows.append({
            "instance": instance,
            "qayd_median_fleet": qayd_fleet,
            "hexaly_median_fleet": hexaly_fleet,
            "comparable_median_gap_percent": median_gap,
            "same_fleet_pairs": len(pairs),
            "paired_median_gap_percent": statistics.median(pairs) if pairs else None,
            "paired_max_gap_percent": max(pairs) if pairs else None,
        })
    median_gap = statistics.median(comparable_instance_gaps) if comparable_instance_gaps else None
    return {
        "instances": rows,
        "complete_instances": complete_instances,
        "comparable_instances": len(comparable_instance_gaps),
        "same_fleet_pairs": len(paired_gaps),
        "median_gap_percent": median_gap,
        "pass": complete_instances == len(TARGETS)
        and len(comparable_instance_gaps) >= len(TARGETS) // 2
        and median_gap is not None
        and median_gap <= 3.0,
    }


def rss_summary(qayd: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    values = []
    for instance in TARGETS:
        for run_seed in SEEDS:
            record = qayd.get((instance, run_seed))
            if record is not None and verified(record):
                rss = finite(record.get("peak_memory_mb"))
                if rss is not None and rss >= 0:
                    values.append(rss)
    median = statistics.median(values) if values else None
    return {"runs": len(values), "median_mb": median, "limit_mb": 64.0, "pass": len(values) == 30 and median is not None and median < 64.0}


def solomon_summary(records: Sequence[dict[str, Any]], solver: str) -> dict[str, Any]:
    expected = {path.stem.upper() for path in (ROOT / "bench/routing/instances/solomon/In").glob("*.txt")}
    selected, duplicates = select_runs(records, solver)
    covered = {instance for instance, _run_seed in selected if instance in expected and verified(selected[(instance, _run_seed)])}
    return {
        "expected_instances": len(expected),
        "covered_instances": len(covered),
        "missing_instances": sorted(expected - covered),
        "duplicate_runs": duplicates,
        "pass": bool(expected) and covered == expected,
    }


HOMBERGER = re.compile(r"^(RC1|RC2|C1|C2|R1|R2)_([24])_(\d+)$", re.IGNORECASE)


def homberger_summary(records: Sequence[dict[str, Any]], solver: str) -> dict[str, Any]:
    selected, duplicates = select_runs(records, solver)
    grouped: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for (instance, _run_seed), record in selected.items():
        match = HOMBERGER.match(instance)
        value = objectives(record)
        if match is None or value is None or not verified(record):
            continue
        size = int(match.group(2)) * 100
        grouped[(match.group(1).upper(), match.group(3))][size].append(value[0])
    rows = []
    for (family, index), sizes in sorted(grouped.items()):
        fleet_200 = statistics.median(sizes.get(200, [])) if sizes.get(200) else None
        fleet_400 = statistics.median(sizes.get(400, [])) if sizes.get(400) else None
        strict_limit = 2 * fleet_200 if fleet_200 is not None else None
        integral_limit = strict_limit + 1 if strict_limit is not None else None
        strict_pass = fleet_400 is not None and strict_limit is not None and fleet_400 <= strict_limit
        passed = fleet_400 is not None and integral_limit is not None and fleet_400 <= integral_limit
        rows.append({
            "family": family,
            "index": index,
            "fleet_200": fleet_200,
            "fleet_400": fleet_400,
            "strict_limit": strict_limit,
            "integral_limit": integral_limit,
            "strict_pass": strict_pass,
            "pass": passed,
        })
    return {"pairs": rows, "duplicate_runs": duplicates, "pass": bool(rows) and all(row["pass"] for row in rows)}


def build_summary(
    qayd_records: Sequence[dict[str, Any]],
    hexaly_records: Sequence[dict[str, Any]],
    solomon_records: Sequence[dict[str, Any]],
    homberger_records: Sequence[dict[str, Any]],
    qayd_solver: str = "qayd-api",
    hexaly_solver: str = "hexaly",
) -> dict[str, Any]:
    qayd, qayd_duplicates = select_runs(qayd_records, qayd_solver)
    hexaly, hexaly_duplicates = select_runs(hexaly_records, hexaly_solver)
    sentinels = sentinel_rows(qayd)
    summary = {
        "schema_version": 1,
        "qayd_solver": qayd_solver,
        "hexaly_solver": hexaly_solver,
        "sentinels": sentinels,
        "sentinel_duplicate_runs": qayd_duplicates,
        "hexaly_duplicate_runs": hexaly_duplicates,
        "distance": distance_comparison(qayd, hexaly),
        "rss": rss_summary(qayd),
        "solomon": solomon_summary(solomon_records, qayd_solver),
        "homberger": homberger_summary(homberger_records, qayd_solver),
    }
    summary["pass"] = bool(
        all(row["pass"] for row in sentinels)
        and summary["distance"]["pass"]
        and summary["rss"]["pass"]
        and summary["solomon"]["pass"]
        and summary["homberger"]["pass"]
        and qayd_duplicates == 0
        and hexaly_duplicates == 0
    )
    return summary


def markdown(summary: dict[str, Any]) -> str:
    lines = ["# VRPTW fleet-first qualification", "", f"Overall: **{'PASS' if summary['pass'] else 'FAIL'}**", ""]
    lines.extend(["## Sentinel matrix", "", "| Instance | Runs | Target passes | Fleet median | Distance median | Gate |", "|---|---:|---:|---:|---:|---|"])
    for row in summary["sentinels"]:
        lines.append(
            f"| {row['instance']} | {row['runs']}/5 | {row['target_passes']}/5 | {row['fleet_median']} | "
            f"{row['distance_median']} | {'PASS' if row['pass'] else 'FAIL'} |"
        )
    distance = summary["distance"]
    rss = summary["rss"]
    lines.extend([
        "",
        "## Quality and resources",
        "",
        f"- Median same-fleet instance gap against Hexaly: {distance['median_gap_percent']}% over "
        f"{distance['comparable_instances']} comparable instances; {distance['same_fleet_pairs']} seed pairs are also reported.",
        f"- Median RSS: {rss['median_mb']} MB over {rss['runs']}/30 runs, limit {rss['limit_mb']} MB.",
        "",
        "| Instance | Median fleets | Comparable median gap | Same-fleet seed pairs | Paired median gap |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in distance["instances"]:
        lines.append(
            f"| {row['instance']} | {row['qayd_median_fleet']} / {row['hexaly_median_fleet']} | "
            f"{row['comparable_median_gap_percent']}% | {row['same_fleet_pairs']} | {row['paired_median_gap_percent']}% |"
        )
    solomon = summary["solomon"]
    homberger = summary["homberger"]
    lines.extend([
        "",
        "## Scale coverage",
        "",
        f"- Solomon incumbents: {solomon['covered_instances']}/{solomon['expected_instances']} instances.",
        f"- Homberger 200/400 matched families: {len(homberger['pairs'])}.",
    ])
    for row in homberger["pairs"]:
        lines.append(
            f"  - {row['family']}_*_{row['index']}: fleet200={row['fleet_200']}, fleet400={row['fleet_400']}, "
            f"strict_limit={row['strict_limit']} ({'PASS' if row['strict_pass'] else 'FAIL'}), "
            f"integral_limit={row['integral_limit']} ({'PASS' if row['pass'] else 'FAIL'})."
        )
    return "\n".join(lines) + "\n"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qayd-campaign", type=Path, action="append", required=True)
    parser.add_argument("--hexaly-campaign", type=Path, action="append", required=True)
    parser.add_argument("--solomon-campaign", type=Path, action="append", required=True)
    parser.add_argument("--homberger-campaign", type=Path, action="append", required=True)
    parser.add_argument("--qayd-solver", default="qayd-api")
    parser.add_argument("--hexaly-solver", default="hexaly")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--require-acceptance", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    summary = build_summary(
        load_jsonl(args.qayd_campaign),
        load_jsonl(args.hexaly_campaign),
        load_jsonl(args.solomon_campaign),
        load_jsonl(args.homberger_campaign),
        args.qayd_solver,
        args.hexaly_solver,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(summary), encoding="utf-8")
    print(args.markdown)
    return int(args.require_acceptance and not summary["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
