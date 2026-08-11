#!/usr/bin/env python3
"""Build a detailed, per-instance positioning report from campaign JSONL files."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence


FEASIBLE = {"OPTIMAL", "SATISFIABLE"}


def load_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
    return records


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def median(values: Iterable[object]) -> float | None:
    numbers = [float(value) for value in values if is_number(value)]
    return statistics.median(numbers) if numbers else None


def successful(record: dict[str, Any]) -> bool:
    return bool(
        record.get("status") in FEASIBLE
        and record.get("verified") is True
        and isinstance(record.get("objectives"), list)
        and record["objectives"]
        and all(is_number(value) for value in record["objectives"])
    )


def objective(record: dict[str, Any]) -> tuple[float, ...] | None:
    if not successful(record):
        return None
    return tuple(float(value) for value in record["objectives"])


def objective_order(values: Iterable[tuple[float, ...]]) -> list[tuple[float, ...]]:
    return sorted(values)


def actual_median_objective(values: Iterable[tuple[float, ...]]) -> tuple[float, ...] | None:
    ordered = objective_order(values)
    return ordered[len(ordered) // 2] if ordered else None


def shown_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}"


def shown_objective(value: tuple[float, ...] | None) -> str:
    return "n/a" if value is None else "[" + ", ".join(shown_number(item) for item in value) + "]"


def shown_percent(value: object) -> str:
    return "n/a" if not is_number(value) else f"{float(value):.2f}%"


def shown_float(value: object, digits: int = 1) -> str:
    return "n/a" if not is_number(value) else f"{float(value):.{digits}f}"


def bks_gap(objective_value: tuple[float, ...] | None, best_known: object) -> float | None:
    if objective_value is None or len(objective_value) != 1 or not is_number(best_known):
        return None
    reference = float(best_known)
    return 100.0 * (objective_value[0] - reference) / max(1.0, abs(reference))


def summarize_instance_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record["problem"], record["family"], record["instance"], record["solver"])
        groups[key].append(record)
    rows = []
    for (problem, family, instance, solver), runs in sorted(groups.items()):
        solved = [run for run in runs if successful(run)]
        objectives = [value for run in solved if (value := objective(run)) is not None]
        ordered = objective_order(objectives)
        best_known = next((run.get("best_known") for run in runs if is_number(run.get("best_known"))), None)
        median_objective = actual_median_objective(objectives)
        rows.append({
            "problem": problem,
            "family": family,
            "instance": instance,
            "solver": solver,
            "runs": len(runs),
            "successes": len(solved),
            "optimal": sum(run.get("status") == "OPTIMAL" for run in runs),
            "best_objective": list(ordered[0]) if ordered else None,
            "median_objective": list(median_objective) if median_objective else None,
            "worst_objective": list(ordered[-1]) if ordered else None,
            "best_known": best_known,
            "median_bks_gap_percent": bks_gap(median_objective, best_known),
            "median_peak_memory_mb": median(run.get("peak_memory_mb") for run in runs),
            "median_wall_seconds": median(run.get("wall_seconds") for run in runs),
            "median_candidates_per_second": median(run.get("candidates_per_second") for run in runs),
        })
    return rows


def compare(left: dict[str, Any], right: dict[str, Any]) -> int | None:
    left_objective, right_objective = objective(left), objective(right)
    if left_objective is None and right_objective is None:
        return None
    if left_objective is None:
        return 1
    if right_objective is None:
        return -1
    return -1 if left_objective < right_objective else 1 if left_objective > right_objective else 0


def paired_rows(records: list[dict[str, Any]], baseline: str = "qayd-api") -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (
            record["problem"], record["instance"],
            int(record["checkpoint_seconds"]), int(record["seed"]),
        )
        groups[key][record["solver"]] = record
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"baseline_wins": 0, "competitor_wins": 0, "ties": 0, "unresolved": 0}
    )
    for (problem, _instance, _checkpoint, _seed), solvers in groups.items():
        if baseline not in solvers:
            continue
        for competitor, right in solvers.items():
            if competitor == baseline:
                continue
            result = compare(solvers[baseline], right)
            row = counts[(problem, competitor)]
            if result is None:
                row["unresolved"] += 1
            elif result < 0:
                row["baseline_wins"] += 1
            elif result > 0:
                row["competitor_wins"] += 1
            else:
                row["ties"] += 1
    return [
        {"problem": problem, "baseline": baseline, "competitor": competitor, **values}
        for (problem, competitor), values in sorted(counts.items())
    ]


def solver_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["problem"], record["solver"])].append(record)
    rows = []
    for (problem, solver), runs in sorted(groups.items()):
        solved = [run for run in runs if successful(run)]
        scalar_gaps = []
        for run in solved:
            gap = bks_gap(objective(run), run.get("best_known"))
            if gap is not None:
                scalar_gaps.append(gap)
        rows.append({
            "problem": problem,
            "solver": solver,
            "runs": len(runs),
            "successes": len(solved),
            "optimal": sum(run.get("status") == "OPTIMAL" for run in runs),
            "median_bks_gap_percent": median(scalar_gaps),
            "median_peak_memory_mb": median(run.get("peak_memory_mb") for run in runs),
            "median_wall_seconds": median(run.get("wall_seconds") for run in runs),
        })
    return rows


def provenance(paths: Sequence[Path]) -> dict[str, Any]:
    sidecars = []
    for path in paths:
        sidecar = path.with_suffix(path.suffix + ".provenance.json")
        if sidecar.is_file():
            sidecars.append(json.loads(sidecar.read_text(encoding="utf-8")))
    hosts = [value.get("host", {}) for value in sidecars]
    return {
        "campaigns": len(paths),
        "sidecars": len(sidecars),
        "dirty": any(bool(host.get("dirty")) for host in hosts),
        "commits": sorted({host.get("commit") for host in hosts if host.get("commit")}),
        "platforms": sorted({host.get("platform") for host in hosts if host.get("platform")}),
        "solver_versions": {
            solver: version
            for sidecar in sidecars
            for solver, version in sidecar.get("solvers", {}).items()
        },
    }


def markdown(summary: dict[str, Any]) -> str:
    if summary["provenance"]["dirty"]:
        provenance_limit = (
            "La provenance dirty interdit de présenter ces chiffres comme un résultat de release "
            "tant que le code mesuré et les outils d'étude ne sont pas figés dans un commit propre."
        )
    else:
        provenance_limit = (
            "La provenance est propre et permet d'identifier exactement le code mesuré. "
            "Elle ne compense toutefois pas les limites statistiques du protocole."
        )
    lines = [
        "# Première étude de positionnement de qayd",
        "",
        f"Campagnes: {summary['provenance']['campaigns']}. Exécutions: {summary['record_count']}. "
        f"Seeds: {', '.join(map(str, summary['seeds']))}. Budget: {', '.join(map(str, summary['budgets']))} s. "
        f"Threads: {', '.join(map(str, summary['threads']))}.",
        "",
        "## Intégrité et méthode",
        "",
        f"- Solutions faisables non vérifiées: {summary['unverified_feasible']}.",
        f"- Erreurs de processus: {summary['process_errors']}.",
        f"- Provenance dirty: {'oui' if summary['provenance']['dirty'] else 'non'}.",
        "- Chaque ligne par instance utilise cinq seeds indépendants. La médiane lexicographique est une solution réellement observée.",
        "- Les victoires appariées comparent le même problème, la même instance, le même seed et le même budget.",
        "- Le gap BKS est calculé seulement pour les objectifs scalaires disposant d'une référence. Aucun scalaire artificiel n'est créé pour le VRPTW lexicographique.",
        "",
        "## Vue par domaine",
        "",
        "| Problème | Solveur | Succès | Optimal prouvé | Gap BKS médian | Temps médian | RSS médiane |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["solver_rows"]:
        lines.append(
            f"| {row['problem']} | {row['solver']} | {row['successes']}/{row['runs']} | "
            f"{row['optimal']}/{row['runs']} | {shown_percent(row['median_bks_gap_percent'])} | "
            f"{shown_float(row['median_wall_seconds'], 3)} s | {shown_float(row['median_peak_memory_mb'])} Mo |"
        )
    lines.extend((
        "",
        "## Comparaisons appariées à qayd-api",
        "",
        "| Problème | Concurrent | Victoires qayd | Victoires concurrent | Égalités | Non résolues |",
        "|---|---|---:|---:|---:|---:|",
    ))
    for row in summary["paired_rows"]:
        lines.append(
            f"| {row['problem']} | {row['competitor']} | {row['baseline_wins']} | "
            f"{row['competitor_wins']} | {row['ties']} | {row['unresolved']} |"
        )
    lines.extend((
        "",
        "## Résultats détaillés par instance",
        "",
        "| Problème | Instance | Solveur | Succès | Optimal | Objectif meilleur / médian / pire | Gap BKS médian | RSS médiane |",
        "|---|---|---|---:|---:|---|---:|---:|",
    ))
    for row in summary["instance_rows"]:
        best = tuple(row["best_objective"]) if row["best_objective"] else None
        middle = tuple(row["median_objective"]) if row["median_objective"] else None
        worst = tuple(row["worst_objective"]) if row["worst_objective"] else None
        objective_range = f"{shown_objective(best)} / {shown_objective(middle)} / {shown_objective(worst)}"
        lines.append(
            f"| {row['problem']} | {row['instance']} | {row['solver']} | "
            f"{row['successes']}/{row['runs']} | {row['optimal']}/{row['runs']} | {objective_range} | "
            f"{shown_percent(row['median_bks_gap_percent'])} | {shown_float(row['median_peak_memory_mb'])} Mo |"
        )
    failed = [
        row for row in summary["instance_rows"]
        if row["solver"] == "qayd-api" and row["successes"] < row["runs"]
    ]
    lines.extend((
        "",
        "## Limites observées de qayd",
        "",
    ))
    if failed:
        for row in failed:
            lines.append(
                f"- {row['problem']} {row['instance']}: {row['successes']}/{row['runs']} incumbents vérifiés, "
                f"RSS médiane {shown_float(row['median_peak_memory_mb'])} Mo."
            )
    else:
        lines.append("- Aucun échec de production d'incumbent dans cette matrice.")
    lines.extend((
        "",
        "## Limites de l'étude",
        "",
        "Cette matrice est une première étude locale stratifiée, pas une publication définitive. "
        "Cinq seeds permettent d'estimer la médiane et la robustesse, mais restent insuffisants "
        "pour une inférence statistique fine. Le budget unique de dix secondes favorise le "
        f"démarrage rapide et ne décrit pas la convergence longue. {provenance_limit}",
        "",
    ))
    return "\n".join(lines)


def build_summary(paths: Sequence[Path]) -> dict[str, Any]:
    records = load_records(paths)
    return {
        "schema_version": 1,
        "record_count": len(records),
        "campaigns": [str(path) for path in paths],
        "seeds": sorted({int(record["seed"]) for record in records}),
        "budgets": sorted({int(record["checkpoint_seconds"]) for record in records}),
        "threads": sorted({int(record["threads"]) for record in records}),
        "unverified_feasible": sum(record.get("status") in FEASIBLE and not successful(record) for record in records),
        "process_errors": sum(record.get("status") == "ERROR" or record.get("return_code") not in (0, None) for record in records),
        "provenance": provenance(paths),
        "solver_rows": solver_rows(records),
        "paired_rows": paired_rows(records),
        "instance_rows": summarize_instance_groups(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, nargs="+")
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", type=Path, dest="json_output")
    args = parser.parse_args()
    paths = [path.resolve() for path in args.campaign]
    summary = build_summary(paths)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown(summary), encoding="utf-8")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"positioning report: {args.markdown} ({summary['record_count']} records)")


if __name__ == "__main__":
    main()
