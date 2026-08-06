#!/usr/bin/env python3
"""Run clean-room black-box behavior probes against public solver interfaces.

The probe creates CVRPLIB pairs that differ by one controlled transformation,
delegates execution to ``bench/campaign.py``, and summarizes paired sensitivity.
It never disassembles, decompiles, or inspects executable code, non-public
files, or implementation details. Independent time checkpoints approximate
anytime behavior using only the final result returned by each public solve call.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from random import Random
import statistics
import subprocess
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "bench" / "campaign.py"
SOLVERS = ("qayd-api", "qayd-native", "hexaly", "hgs", "lkh")
FEASIBLE_STATUSES = {"SAT", "SATISFIABLE", "FEASIBLE", "OPTIMAL", "OPTIMUM"}


def integer_list(values: Iterable[str], *, positive: bool) -> list[int]:
    parsed: list[int] = []
    for value in values:
        for field in value.split(","):
            if not field.strip():
                continue
            number = int(field)
            if (positive and number <= 0) or (not positive and number < 0):
                condition = "positive" if positive else "non-negative"
                raise argparse.ArgumentTypeError(f"values must be {condition}")
            parsed.append(number)
    return sorted(set(parsed))


def workspace_relative(path: Path) -> tuple[Path, str]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as error:
        raise SystemExit(f"probe workspace must be inside {ROOT}: {resolved}") from error
    return resolved, relative.as_posix()


def euclidean_matrix(customers: int, seed: int) -> tuple[list[list[int]], list[int]]:
    rng = Random(seed)
    coordinates = [(500, 500)]
    seen = {coordinates[0]}
    while len(coordinates) <= customers:
        candidate = (rng.randint(0, 1000), rng.randint(0, 1000))
        if candidate not in seen:
            coordinates.append(candidate)
            seen.add(candidate)
    matrix = []
    for ax, ay in coordinates:
        row = []
        for bx, by in coordinates:
            row.append(round(math.hypot(ax - bx, ay - by)))
        matrix.append(row)
    demands = [0] + [rng.randint(1, 9) for _ in range(customers)]
    return matrix, demands


def base_capacity(demands: Sequence[int]) -> int:
    customers = len(demands) - 1
    target_routes = max(2, round(math.sqrt(customers)))
    return max(max(demands) + 1, math.ceil(sum(demands) / target_routes))


def write_cvrplib(
    path: Path, *, name: str, matrix: Sequence[Sequence[int]],
    demands: Sequence[int], capacity: int, comment: str,
) -> None:
    dimension = len(demands)
    if dimension != len(matrix) or any(len(row) != dimension for row in matrix):
        raise ValueError("matrix and demand dimensions differ")
    lines = [
        f"NAME : {name}",
        f"COMMENT : {comment}",
        "TYPE : CVRP",
        f"DIMENSION : {dimension}",
        "EDGE_WEIGHT_TYPE : EXPLICIT",
        "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
        f"CAPACITY : {capacity}",
        "EDGE_WEIGHT_SECTION",
    ]
    lines.extend(" ".join(str(value) for value in row) for row in matrix)
    lines.append("DEMAND_SECTION")
    lines.extend(f"{node + 1} {demand}" for node, demand in enumerate(demands))
    lines.extend(("DEPOT_SECTION", "1", "-1", "EOF"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def permuted_problem(
    matrix: Sequence[Sequence[int]], demands: Sequence[int], seed: int,
) -> tuple[list[list[int]], list[int], dict[str, int]]:
    order = list(range(1, len(demands)))
    Random(seed).shuffle(order)
    if order == list(range(1, len(demands))) and len(order) > 1:
        order[0], order[1] = order[1], order[0]
    order = [0, *order]
    permuted_matrix = [[matrix[source_row][source_col] for source_col in order] for source_row in order]
    permuted_demands = [demands[source] for source in order]
    # Public node ids in the treatment map back to public ids in the control.
    node_map = {str(position + 1): source + 1 for position, source in enumerate(order)}
    return permuted_matrix, permuted_demands, node_map


def nearest_customer_edge(matrix: Sequence[Sequence[int]]) -> tuple[int, int]:
    return min(
        ((left, right) for left in range(1, len(matrix)) for right in range(left + 1, len(matrix))),
        key=lambda edge: (matrix[edge[0]][edge[1]], edge),
    )


def generate_probe_suite(
    workspace: Path, *, sizes: Sequence[int], instance_seed: int,
    distance_scale: int, edge_delta: int,
) -> tuple[Path, Path, dict[str, Any]]:
    if not sizes or min(sizes) < 2:
        raise ValueError("behavior probes require at least two customers")
    workspace, workspace_from_root = workspace_relative(workspace)
    instances_dir = workspace / "instances"
    probes: list[dict[str, Any]] = []
    datasets: list[dict[str, str]] = []

    def add_pair(
        *, size: int, factor: str, control_matrix: Sequence[Sequence[int]],
        control_demands: Sequence[int], control_capacity: int,
        treatment_matrix: Sequence[Sequence[int]], treatment_demands: Sequence[int],
        treatment_capacity: int, treatment_scale: float,
        treatment_node_map: dict[str, int], change: str,
        semantic_equivalent: bool,
    ) -> None:
        pair_id = f"n{size}-{factor}"
        family = f"probe-{factor}-n{size}"
        identity = {str(node): node for node in range(1, size + 2)}
        for role, matrix, demands, capacity, scale, node_map in (
            ("control", control_matrix, control_demands, control_capacity, 1.0, identity),
            (
                "treatment", treatment_matrix, treatment_demands,
                treatment_capacity, treatment_scale, treatment_node_map,
            ),
        ):
            name = f"probe-{pair_id}-{role}"
            path = instances_dir / f"{name}.vrp"
            write_cvrplib(
                path, name=name, matrix=matrix, demands=demands, capacity=capacity,
                comment=f"qayd clean-room behavior probe: {change}",
            )
            relative = path.relative_to(ROOT).as_posix()
            datasets.append({"family": family, "problem": "cvrp", "glob": relative})
            probes.append({
                "probe_id": f"{pair_id}-{role}",
                "pair_id": pair_id,
                "factor": factor,
                "size": size,
                "role": role,
                "instance_path": relative,
                "objective_scale": scale,
                "depot_node_id": 1,
                "node_map_to_control": node_map,
                "semantic_equivalent": semantic_equivalent,
                "change": change,
            })

    for size in sizes:
        matrix, demands = euclidean_matrix(size, instance_seed + size * 1_000_003)
        capacity = base_capacity(demands)
        identity = {str(node): node for node in range(1, size + 2)}

        scaled = [[value * distance_scale for value in row] for row in matrix]
        add_pair(
            size=size, factor="distance-scale", control_matrix=matrix,
            control_demands=demands, control_capacity=capacity,
            treatment_matrix=scaled, treatment_demands=demands,
            treatment_capacity=capacity, treatment_scale=float(distance_scale),
            treatment_node_map=identity,
            change=f"multiply every edge cost by {distance_scale}",
            semantic_equivalent=True,
        )

        permuted_matrix, permuted_demands, node_map = permuted_problem(
            matrix, demands, instance_seed ^ (size * 2_000_033),
        )
        add_pair(
            size=size, factor="index-permutation", control_matrix=matrix,
            control_demands=demands, control_capacity=capacity,
            treatment_matrix=permuted_matrix, treatment_demands=permuted_demands,
            treatment_capacity=capacity, treatment_scale=1.0,
            treatment_node_map=node_map,
            change="permute customer positions while preserving the CVRP isomorphism",
            semantic_equivalent=True,
        )

        changed_matrix = [list(row) for row in matrix]
        left, right = nearest_customer_edge(matrix)
        increment = max(edge_delta, math.ceil(matrix[left][right] / 2))
        changed_matrix[left][right] += increment
        changed_matrix[right][left] += increment
        add_pair(
            size=size, factor="single-edge", control_matrix=matrix,
            control_demands=demands, control_capacity=capacity,
            treatment_matrix=changed_matrix, treatment_demands=demands,
            treatment_capacity=capacity, treatment_scale=1.0,
            treatment_node_map=identity,
            change=f"increase symmetric edge ({left + 1}, {right + 1}) by {increment}",
            semantic_equivalent=False,
        )

        tightened = max(max(demands), capacity - max(1, capacity // 8))
        add_pair(
            size=size, factor="capacity-threshold", control_matrix=matrix,
            control_demands=demands, control_capacity=capacity,
            treatment_matrix=matrix, treatment_demands=demands,
            treatment_capacity=tightened, treatment_scale=1.0,
            treatment_node_map=identity,
            change=f"reduce capacity from {capacity} to {tightened}",
            semantic_equivalent=False,
        )

    suite = {
        "name": "clean-room-behavior-probe",
        "description": "Controlled CVRP pairs for black-box solver behavior measurements.",
        "datasets": datasets,
    }
    manifest = {
        "schema_version": 1,
        "method": "clean-room-black-box",
        "boundary": (
            "Generated public inputs and documented solver API outputs only. "
            "Vendor-distributed runtime files are loaded only to call that API, and an opaque executable hash may be recorded for provenance. "
            "No disassembly, decompilation, executable-code inspection, hidden API use, or non-public file access."
        ),
        "generator": {
            "instance_seed": instance_seed,
            "sizes": list(sizes),
            "distance_scale": distance_scale,
            "edge_delta": edge_delta,
        },
        "workspace": workspace_from_root,
        "probes": probes,
    }
    suite_path = workspace / "suite.json"
    manifest_path = workspace / "manifest.json"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return suite_path, manifest_path, manifest


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
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
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def median(values: Iterable[object]) -> float | None:
    numbers = [number for value in values if (number := finite(value)) is not None]
    return statistics.median(numbers) if numbers else None


def ratio(numerator: object, denominator: object) -> float | None:
    top, bottom = finite(numerator), finite(denominator)
    if top is None or bottom is None or bottom == 0:
        return None
    return top / bottom


def primary_objective(record: dict[str, Any], scale: float) -> float | None:
    objectives = record.get("objectives")
    if not isinstance(objectives, list) or not objectives or scale <= 0:
        return None
    value = finite(objectives[0])
    return value / scale if value is not None else None


def mapped_route_edges(record: dict[str, Any], probe: dict[str, Any]) -> set[tuple[int, int]] | None:
    routes = record.get("routes")
    if not isinstance(routes, list):
        return None
    mapping = {int(node): int(target) for node, target in probe["node_map_to_control"].items()}
    depot = mapping[int(probe["depot_node_id"])]
    edges: set[tuple[int, int]] = set()
    for route in routes:
        nodes = route.get("nodes") if isinstance(route, dict) else None
        if not isinstance(nodes, list) or not nodes:
            continue
        try:
            sequence = [depot, *(mapping[int(node)] for node in nodes), depot]
        except (KeyError, TypeError, ValueError):
            return None
        edges.update(tuple(sorted((left, right))) for left, right in zip(sequence, sequence[1:]))
    return edges


def jaccard(left: set[tuple[int, int]] | None, right: set[tuple[int, int]] | None) -> float | None:
    if left is None or right is None:
        return None
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def paired_runs(manifest: dict[str, Any], records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    probes = {probe["instance_path"]: probe for probe in manifest["probes"]}
    runs: dict[tuple[object, ...], dict[str, tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(dict)
    for record in records:
        probe = probes.get(record.get("instance_path"))
        if probe is None:
            continue
        key = (
            record.get("solver"), record.get("checkpoint_seconds"),
            record.get("seed"), probe["pair_id"],
        )
        runs[key][probe["role"]] = (record, probe)

    pairs = []
    for (solver, budget, seed, pair_id), roles in sorted(runs.items(), key=lambda item: tuple(map(str, item[0]))):
        if set(roles) != {"control", "treatment"}:
            continue
        control, control_probe = roles["control"]
        treatment, treatment_probe = roles["treatment"]
        control_objective = primary_objective(control, float(control_probe["objective_scale"]))
        treatment_objective = primary_objective(treatment, float(treatment_probe["objective_scale"]))
        objective_delta = None
        if control_objective is not None and treatment_objective is not None:
            objective_delta = 100.0 * (treatment_objective - control_objective) / max(1.0, abs(control_objective))
        control_status = str(control.get("status", "UNKNOWN"))
        treatment_status = str(treatment.get("status", "UNKNOWN"))
        control_feasible = control_status in FEASIBLE_STATUSES and bool(control.get("verified"))
        treatment_feasible = treatment_status in FEASIBLE_STATUSES and bool(treatment.get("verified"))
        pairs.append({
            "solver": solver,
            "checkpoint_seconds": budget,
            "seed": seed,
            "pair_id": pair_id,
            "factor": control_probe["factor"],
            "size": control_probe["size"],
            "semantic_equivalent": control_probe["semantic_equivalent"],
            "change": control_probe["change"],
            "control_run_id": control.get("run_id"),
            "treatment_run_id": treatment.get("run_id"),
            "status_match": control_status == treatment_status,
            "both_verified_feasible": control_feasible and treatment_feasible,
            "control_objective": control_objective,
            "treatment_normalized_objective": treatment_objective,
            "normalized_objective_delta_percent": objective_delta,
            "normalized_objective_agreement": (
                objective_delta is not None and abs(objective_delta) <= 1e-9
            ),
            "wall_time_ratio": ratio(treatment.get("wall_seconds"), control.get("wall_seconds")),
            "peak_memory_ratio": ratio(treatment.get("peak_memory_mb"), control.get("peak_memory_mb")),
            "candidate_throughput_ratio": ratio(
                treatment.get("candidates_per_second"), control.get("candidates_per_second"),
            ),
            "candidate_count_ratio": ratio(
                treatment.get("candidates_evaluated"), control.get("candidates_evaluated"),
            ),
            "dual_ratio": ratio(
                finite(treatment.get("dual_bound")) / float(treatment_probe["objective_scale"])
                if finite(treatment.get("dual_bound")) is not None else None,
                finite(control.get("dual_bound")) / float(control_probe["objective_scale"])
                if finite(control.get("dual_bound")) is not None else None,
            ),
            "route_edge_jaccard": jaccard(
                mapped_route_edges(control, control_probe),
                mapped_route_edges(treatment, treatment_probe),
            ),
        })
    return pairs


def aggregate_pairs(pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[(pair["solver"], pair["factor"], pair["size"], pair["checkpoint_seconds"])].append(pair)
    rows = []
    for (solver, factor, size, budget), values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        objective_deltas = [finite(value["normalized_objective_delta_percent"]) for value in values]
        objective_deltas = [value for value in objective_deltas if value is not None]
        semantic = bool(values[0]["semantic_equivalent"])
        rows.append({
            "solver": solver,
            "factor": factor,
            "size": size,
            "checkpoint_seconds": budget,
            "paired_runs": len(values),
            "verified_pair_percent": 100.0 * sum(bool(value["both_verified_feasible"]) for value in values) / len(values),
            "status_match_percent": 100.0 * sum(bool(value["status_match"]) for value in values) / len(values),
            "semantic_equivalent": semantic,
            "normalized_objective_agreement_percent": (
                100.0 * sum(bool(value["normalized_objective_agreement"]) for value in values) / len(values)
                if semantic else None
            ),
            "objective_response_percent": (
                100.0 * sum(abs(value) > 1e-9 for value in objective_deltas) / len(objective_deltas)
                if objective_deltas else None
            ),
            "median_normalized_objective_delta_percent": median(objective_deltas),
            "median_wall_time_ratio": median(value["wall_time_ratio"] for value in values),
            "median_peak_memory_ratio": median(value["peak_memory_ratio"] for value in values),
            "median_candidate_throughput_ratio": median(value["candidate_throughput_ratio"] for value in values),
            "median_candidate_count_ratio": median(value["candidate_count_ratio"] for value in values),
            "median_dual_ratio": median(value["dual_ratio"] for value in values),
            "median_route_edge_jaccard": median(value["route_edge_jaccard"] for value in values),
        })
    return rows


def observations(rows: Sequence[dict[str, Any]]) -> list[str]:
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["solver"]), str(row["factor"]), int(row["size"]))
        if key not in latest or row["checkpoint_seconds"] > latest[key]["checkpoint_seconds"]:
            latest[key] = row
    notes = []
    for (solver, factor, size), row in sorted(latest.items()):
        agreement = finite(row["normalized_objective_agreement_percent"])
        if factor == "index-permutation" and agreement is not None:
            if agreement < 90:
                notes.append(
                    f"{solver}, n={size}: {agreement:.0f}% d'accord après permutation. "
                    "Le comportement observé est compatible avec des tie-breaks ou voisinages sensibles aux indices."
                )
            else:
                notes.append(
                    f"{solver}, n={size}: {agreement:.0f}% d'accord après permutation, "
                    "donc faible sensibilité indicielle observée à ce checkpoint."
                )
        if factor == "distance-scale" and agreement is not None:
            if agreement < 90:
                notes.append(
                    f"{solver}, n={size}: {agreement:.0f}% d'accord après changement d'échelle. "
                    "Le résultat est compatible avec des seuils absolus dans la recherche."
                )
            else:
                notes.append(
                    f"{solver}, n={size}: {agreement:.0f}% d'accord après changement d'échelle, "
                    "donc invariance numérique observée à ce checkpoint."
                )
    if not notes:
        notes.append("Pas assez de paires faisables et vérifiées pour produire une observation.")
    return notes


def display(value: object, digits: int = 2) -> str:
    number = finite(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def display_percent(value: object, digits: int = 2) -> str:
    rendered = display(value, digits)
    return rendered if rendered == "n/a" else f"{rendered}%"


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Clean-room solver behavior probe",
        "",
        "Ce rapport utilise uniquement des instances générées, les interfaces publiques des solveurs et leurs sorties publiques.",
        "Les fichiers runtime distribués servent uniquement à charger l'API, avec éventuellement un hash opaque pour la provenance.",
        "Il ne contient aucun désassemblage, décompilation, inspection du code exécutable, API cachée ou fichier non public.",
        "Les checkpoints sont des résolutions indépendantes, pas une trace interne des incumbents.",
        "",
        "## Sensibilité appariée",
        "",
        "| Solveur | Facteur | n | Budget | Paires | Vérifiées | Accord objectif | Delta objectif | Temps x | RSS x | Débit x | Arêtes Jaccard |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['solver']} | {row['factor']} | {row['size']} | {row['checkpoint_seconds']}s | "
            f"{row['paired_runs']} | {display_percent(row['verified_pair_percent'])} | "
            f"{display_percent(row['normalized_objective_agreement_percent'])} | "
            f"{display_percent(row['median_normalized_objective_delta_percent'])} | "
            f"{display(row['median_wall_time_ratio'])} | {display(row['median_peak_memory_ratio'])} | "
            f"{display(row['median_candidate_throughput_ratio'])} | {display(row['median_route_edge_jaccard'], 3)} |"
        )
    lines.extend((
        "",
        "L'objectif du traitement est normalisé pour le probe de changement d'échelle. "
        "L'accord ne signifie pas que les routes sont identiques. Jaccard vaut 1 pour les mêmes arêtes après remappage des indices.",
        "",
        "## Observations prudentes",
        "",
    ))
    lines.extend(f"- {note}" for note in summary["observations"])
    lines.extend((
        "",
        "Ces observations indiquent des comportements compatibles avec certaines stratégies. "
        "Elles ne prouvent aucun détail d'implémentation.",
        "",
    ))
    return "\n".join(lines)


def analyze(manifest_path: Path, results_path: Path, markdown_path: Path, json_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = load_jsonl(results_path)
    pairs = paired_runs(manifest, records)
    rows = aggregate_pairs(pairs)
    summary = {
        "schema_version": 1,
        "method": manifest["method"],
        "boundary": manifest["boundary"],
        "record_count": len(records),
        "paired_run_count": len(pairs),
        "rows": rows,
        "observations": observations(rows),
        "paired_runs": pairs,
    }
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_report(summary), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def campaign_command(args: argparse.Namespace, suite_path: Path, results_path: Path) -> list[str]:
    command = [
        sys.executable, str(CAMPAIGN), "--suite", str(suite_path),
        "--budget", ",".join(map(str, args.budgets)),
        "--seed", ",".join(map(str, args.seeds)),
        "--threads", str(args.threads),
        "--memory-limit-mb", str(args.memory_limit_mb),
        "--grace-seconds", str(args.grace_seconds),
        "--qayd-engine", args.qayd_engine,
        "--out", str(results_path),
    ]
    for solver in args.solver:
        command.extend(("--solver", solver))
    if args.max_iterations is not None:
        command.extend(("--max-iterations", str(args.max_iterations)))
    command.append("--routing-two-way" if args.routing_two_way else "--no-routing-two-way")
    command.append("--routing-nearest-neighbor" if args.routing_nearest_neighbor else "--no-routing-nearest-neighbor")
    command.append("--routing-warm-start" if args.routing_warm_start else "--no-routing-warm-start")
    for flag, value in (
        ("--hexaly-home", args.hexaly_home),
        ("--hgs-binary", args.hgs_binary),
        ("--lkh-binary", args.lkh_binary),
    ):
        if value:
            command.extend((flag, str(Path(value).resolve())))
    if args.restart:
        command.append("--restart")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT / "bench" / "results" / "behavior-probe")
    parser.add_argument("--solver", action="append", choices=SOLVERS)
    parser.add_argument("--customers", action="append", default=[], help="repeat or use comma-separated sizes")
    parser.add_argument("--budget", action="append", default=[], help="repeat or use comma-separated seconds")
    parser.add_argument("--seed", action="append", default=[], help="repeat or use comma-separated solver seeds")
    parser.add_argument("--instance-seed", type=int, default=20260806)
    parser.add_argument("--distance-scale", type=int, default=7)
    parser.add_argument("--edge-delta", type=int, default=25)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit-mb", type=int, default=0)
    parser.add_argument("--grace-seconds", type=float, default=20.0)
    parser.add_argument("--qayd-engine", choices=("auto", "exact", "ls"), default="ls")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--routing-two-way", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--routing-nearest-neighbor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--routing-warm-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hexaly-home", type=Path)
    parser.add_argument("--hgs-binary", type=Path)
    parser.add_argument("--lkh-binary", type=Path)
    parser.add_argument("--restart", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    args.solver = list(dict.fromkeys(args.solver or ["qayd-api"]))
    args.sizes = integer_list(args.customers or ["40"], positive=True)
    args.budgets = integer_list(args.budget or ["1,3"], positive=True)
    args.seeds = integer_list(args.seed or ["0,1,2,3,4"], positive=False)
    if (
        args.instance_seed < 0 or args.distance_scale <= 1 or args.edge_delta <= 0
        or args.threads <= 0 or args.memory_limit_mb < 0 or args.grace_seconds < 0
        or min(args.sizes) < 2
        or (args.max_iterations is not None and args.max_iterations < 0)
    ):
        raise SystemExit(
            "instance seed and limits must be non-negative; at least two customers, positive scale, edge delta, and threads are required"
        )
    executes_solver = not args.generate_only and not args.analyze_only
    if executes_solver and "hexaly" in args.solver and not args.hexaly_home:
        raise SystemExit("hexaly requires --hexaly-home")
    if executes_solver and "hgs" in args.solver and not args.hgs_binary:
        raise SystemExit("hgs requires --hgs-binary")
    if executes_solver and "lkh" in args.solver and not args.lkh_binary:
        raise SystemExit("lkh requires --lkh-binary")

    workspace, _ = workspace_relative(args.workspace)
    manifest_path = workspace / "manifest.json"
    suite_path = workspace / "suite.json"
    results_path = workspace / "results.jsonl"
    markdown_path = workspace / "report.md"
    json_path = workspace / "report.json"

    if args.analyze_only:
        if not manifest_path.is_file() or not results_path.is_file():
            raise SystemExit(f"missing manifest or results under {workspace}")
    else:
        suite_path, manifest_path, manifest = generate_probe_suite(
            workspace, sizes=args.sizes, instance_seed=args.instance_seed,
            distance_scale=args.distance_scale, edge_delta=args.edge_delta,
        )
        print(f"generated {len(manifest['probes'])} probe instances under {workspace}", file=sys.stderr)
        if args.generate_only:
            print(manifest_path)
            return

        completed = subprocess.run(campaign_command(args, suite_path, results_path), cwd=ROOT)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    summary = analyze(manifest_path, results_path, markdown_path, json_path)
    print(
        f"behavior probe: {summary['record_count']} records, "
        f"{summary['paired_run_count']} pairs, report {markdown_path}",
    )


if __name__ == "__main__":
    main()
