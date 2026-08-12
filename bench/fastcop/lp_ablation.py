#!/usr/bin/env python3
"""Report the paired contribution of Qayd's optional LP relaxation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Optional, Sequence


RESULT_SCHEMA = "qayd.fastcop.result/v1"
REPORT_SCHEMA = "qayd.fastcop.lp-ablation/v1"
NODE_RE = re.compile(r"^c nodes (\d+) failures (\d+)$", re.MULTILINE)
LP_RE = re.compile(
    r"^c lp rows (\d+) root_bound (\?|[-+]?\d+) solves (\d+) "
    r"certified (\d+)(?: node_prunes (\d+))? timeouts (\d+) refactorizations (\d+) "
    r"time_ms ([0-9]+(?:\.[0-9]+)?)$",
    re.MULTILINE,
)
LP_MODEL_RE = re.compile(
    r"^c lp model ([a-z-]+) vars (\d+) columns (\d+) covered (\d+) objective_vars (\d+) "
    r"objective_covered (\d+)(?: auxiliary (\d+) fallback_terms (\d+))? source_rows (\d+) rows (\d+) nonzeros (\d+)$",
    re.MULTILINE,
)


class AblationError(RuntimeError):
    """The supplied records do not form a controlled paired experiment."""


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.median(finite) if finite else None


def ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def load_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for path in paths:
        try:
            source = path.open(encoding="utf-8")
        except OSError as error:
            raise AblationError(f"cannot read {path}: {error}") from error
        with source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AblationError(f"invalid JSON at {path}:{line_number}: {error}") from error
                if not isinstance(record, dict) or record.get("schema") != RESULT_SCHEMA:
                    raise AblationError(f"unsupported result at {path}:{line_number}")
                key = record.get("execution_key", record.get("run_key"))
                if not isinstance(key, str) or not key:
                    raise AblationError(f"missing execution identity at {path}:{line_number}")
                if key in seen:
                    raise AblationError(f"duplicate execution identity at {path}:{line_number}")
                seen.add(key)
                records.append(record)
    return records


def stdout_path(record: dict[str, Any]) -> Path:
    logs = record.get("logs")
    value = logs.get("stdout") if isinstance(logs, dict) else None
    if not isinstance(value, str) or not value:
        raise AblationError(
            f"missing stdout log for {record.get('solver')}/{record.get('instance')}"
        )
    path = Path(value)
    if not path.is_file():
        raise AblationError(f"stdout log does not exist: {path}")
    return path


def parse_solver_stats(record: dict[str, Any]) -> dict[str, Any]:
    text = stdout_path(record).read_text(encoding="utf-8", errors="replace")
    node_match = NODE_RE.search(text)
    lp_match = LP_RE.search(text)
    model_match = LP_MODEL_RE.search(text)
    stats: dict[str, Any] = {
        "nodes": int(node_match.group(1)) if node_match else None,
        "failures": int(node_match.group(2)) if node_match else None,
        "lp_rows": 0,
        "lp_root_bound": None,
        "lp_solves": 0,
        "lp_certified": 0,
        "lp_node_prunes": 0,
        "lp_timeouts": 0,
        "lp_refactorizations": 0,
        "lp_time_ms": 0.0,
        "lp_model_status": "not-attempted",
        "lp_variables": 0,
        "lp_columns": 0,
        "lp_covered_variables": 0,
        "lp_objective_variables": 0,
        "lp_objective_covered_variables": 0,
        "lp_auxiliary_columns": 0,
        "lp_objective_fallbacks": 0,
        "lp_source_rows": 0,
        "lp_nonzeros": 0,
    }
    if model_match:
        stats.update(
            {
                "lp_model_status": model_match.group(1),
                "lp_variables": int(model_match.group(2)),
                "lp_columns": int(model_match.group(3)),
                "lp_covered_variables": int(model_match.group(4)),
                "lp_objective_variables": int(model_match.group(5)),
                "lp_objective_covered_variables": int(model_match.group(6)),
                "lp_auxiliary_columns": int(model_match.group(7) or 0),
                "lp_objective_fallbacks": int(model_match.group(8) or 0),
                "lp_source_rows": int(model_match.group(9)),
                "lp_rows": int(model_match.group(10)),
                "lp_nonzeros": int(model_match.group(11)),
            }
        )
    if lp_match:
        bound = lp_match.group(2)
        stats.update(
            {
                "lp_rows": int(lp_match.group(1)),
                "lp_root_bound": None if bound == "?" else int(bound),
                "lp_solves": int(lp_match.group(3)),
                "lp_certified": int(lp_match.group(4)),
                "lp_node_prunes": int(lp_match.group(5) or 0),
                "lp_timeouts": int(lp_match.group(6)),
                "lp_refactorizations": int(lp_match.group(7)),
                "lp_time_ms": float(lp_match.group(8)),
            }
        )
    return stats


def artifact_hash(record: dict[str, Any]) -> Optional[str]:
    provenance = record.get("provenance")
    value = provenance.get("artifact_sha256") if isinstance(provenance, dict) else None
    return value if isinstance(value, str) else None


def verified_objective(record: dict[str, Any]) -> Optional[int]:
    validation = record.get("validation")
    incumbent = record.get("best_incumbent")
    if (
        not isinstance(validation, dict)
        or validation.get("valid") is not True
        or not isinstance(incumbent, dict)
    ):
        return None
    value = incumbent.get("value")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def elapsed(record: dict[str, Any], field: str) -> Optional[float]:
    value = record.get(field)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def event_elapsed(record: dict[str, Any], field: str) -> Optional[float]:
    event = record.get(field)
    value = event.get("elapsed_seconds") if isinstance(event, dict) else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def advantage(native: int, amthal: int, sense: str) -> int:
    if sense == "min":
        return native - amthal
    if sense == "max":
        return amthal - native
    raise AblationError(f"unknown objective sense: {sense}")


def certified_gap(objective: Optional[int], bound: Optional[int], sense: str) -> Optional[int]:
    if objective is None or bound is None:
        return None
    gap = objective - bound if sense == "min" else bound - objective
    return gap if gap >= 0 else None


def certified_bound_valid(objective: Optional[int], bound: Optional[int], sense: str) -> Optional[bool]:
    if objective is None or bound is None:
        return None
    if sense == "min":
        return bound <= objective
    if sense == "max":
        return bound >= objective
    raise AblationError(f"unknown objective sense: {sense}")


def pair_records(
    records: Sequence[dict[str, Any]],
    native_name: str,
    amthal_name: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    expected = {native_name, amthal_name}
    paired: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        solver = record.get("solver")
        if solver not in expected:
            continue
        instance = record.get("instance")
        seed = record.get("seed")
        if not isinstance(instance, str) or not isinstance(seed, int):
            raise AblationError("record is missing its instance or seed")
        slot = paired[(instance, seed)]
        if solver in slot:
            raise AblationError(f"duplicate {solver} record for {instance}, seed {seed}")
        slot[solver] = record

    missing = [key for key, variants in paired.items() if set(variants) != expected]
    if missing:
        shown = ", ".join(f"{instance}/seed-{seed}" for instance, seed in missing[:5])
        raise AblationError(f"incomplete A/B pairs: {shown}")
    if not paired:
        raise AblationError(f"no pairs for {native_name} and {amthal_name}")

    result = []
    for key in sorted(paired):
        native = paired[key][native_name]
        amthal = paired[key][amthal_name]
        for field in ("instance", "instance_sha256", "objective_sense", "seed", "limits"):
            if native.get(field) != amthal.get(field):
                raise AblationError(f"uncontrolled field {field} for {key[0]}, seed {key[1]}")
        native_artifact = artifact_hash(native)
        amthal_artifact = artifact_hash(amthal)
        if not native_artifact or native_artifact != amthal_artifact:
            raise AblationError(f"variants used different binaries for {key[0]}, seed {key[1]}")
        result.append((native, amthal))
    return result


def summarize_pairs(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    rows = []
    outcomes = Counter()
    for native, amthal in pairs:
        native_stats = parse_solver_stats(native)
        amthal_stats = parse_solver_stats(amthal)
        native_objective = verified_objective(native)
        amthal_objective = verified_objective(amthal)
        sense = str(native["objective_sense"])
        if native_objective is None and amthal_objective is None:
            outcome = "neither"
        elif native_objective is None:
            outcome = "amthal"
        elif amthal_objective is None:
            outcome = "native"
        else:
            delta = advantage(native_objective, amthal_objective, sense)
            outcome = "amthal" if delta > 0 else "native" if delta < 0 else "tie"
        outcomes[outcome] += 1
        gap = certified_gap(amthal_objective, amthal_stats["lp_root_bound"], sense)
        bound_valid = certified_bound_valid(amthal_objective, amthal_stats["lp_root_bound"], sense)
        rows.append(
            {
                "instance": native["instance"],
                "family": native.get("family"),
                "seed": native["seed"],
                "objective_sense": sense,
                "outcome": outcome,
                "native_status": native.get("status"),
                "amthal_status": amthal.get("status"),
                "native_proof": bool(native.get("proof")),
                "amthal_proof": bool(amthal.get("proof")),
                "native_objective": native_objective,
                "amthal_objective": amthal_objective,
                "objective_advantage": (
                    advantage(native_objective, amthal_objective, sense)
                    if native_objective is not None and amthal_objective is not None
                    else None
                ),
                "native_wall_seconds": elapsed(native, "elapsed_wall_seconds"),
                "amthal_wall_seconds": elapsed(amthal, "elapsed_wall_seconds"),
                "wall_ratio": ratio(
                    elapsed(amthal, "elapsed_wall_seconds"),
                    elapsed(native, "elapsed_wall_seconds"),
                ),
                "first_incumbent_ratio": ratio(
                    event_elapsed(amthal, "first_incumbent"),
                    event_elapsed(native, "first_incumbent"),
                ),
                "best_incumbent_ratio": ratio(
                    event_elapsed(amthal, "best_incumbent"),
                    event_elapsed(native, "best_incumbent"),
                ),
                "node_ratio": ratio(amthal_stats["nodes"], native_stats["nodes"]),
                "native_nodes": native_stats["nodes"],
                "amthal_nodes": amthal_stats["nodes"],
                "lp_model_status": amthal_stats["lp_model_status"],
                "lp_variables": amthal_stats["lp_variables"],
                "lp_columns": amthal_stats["lp_columns"],
                "lp_covered_variables": amthal_stats["lp_covered_variables"],
                "lp_objective_variables": amthal_stats["lp_objective_variables"],
                "lp_objective_covered_variables": amthal_stats[
                    "lp_objective_covered_variables"
                ],
                "lp_auxiliary_columns": amthal_stats["lp_auxiliary_columns"],
                "lp_objective_fallbacks": amthal_stats["lp_objective_fallbacks"],
                "lp_source_rows": amthal_stats["lp_source_rows"],
                "lp_rows": amthal_stats["lp_rows"],
                "lp_nonzeros": amthal_stats["lp_nonzeros"],
                "lp_root_bound": amthal_stats["lp_root_bound"],
                "lp_solves": amthal_stats["lp_solves"],
                "lp_certified": amthal_stats["lp_certified"],
                "lp_node_prunes": amthal_stats["lp_node_prunes"],
                "lp_timeouts": amthal_stats["lp_timeouts"],
                "lp_time_ms": amthal_stats["lp_time_ms"],
                "certified_absolute_gap": gap,
                "certified_bound_valid": bound_valid,
                "certified_relative_gap": (
                    gap / max(1, abs(amthal_objective))
                    if gap is not None and amthal_objective is not None
                    else None
                ),
                "root_bound_matches_incumbent": gap == 0 if gap is not None else False,
            }
        )

    eligible = [row for row in rows if row["lp_rows"] > 0]
    certified = [row for row in rows if row["lp_root_bound"] is not None]
    lifted = [row for row in eligible if row["lp_auxiliary_columns"] > 0]
    family_rows = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    for family, values in sorted(grouped.items()):
        family_rows.append(
            {
                "family": family,
                "pairs": len(values),
                "lp_eligible": sum(row["lp_rows"] > 0 for row in values),
                "lp_certified": sum(row["lp_root_bound"] is not None for row in values),
                "amthal_wins": sum(row["outcome"] == "amthal" for row in values),
                "native_wins": sum(row["outcome"] == "native" for row in values),
                "ties": sum(row["outcome"] == "tie" for row in values),
            }
        )

    return {
        "schema": REPORT_SCHEMA,
        "pairs": len(rows),
        "same_binary": len({artifact_hash(record) for pair in pairs for record in pair}) == 1,
        "outcomes": dict(sorted(outcomes.items())),
        "native_verified_incumbents": sum(row["native_objective"] is not None for row in rows),
        "amthal_verified_incumbents": sum(row["amthal_objective"] is not None for row in rows),
        "native_proofs": sum(row["native_proof"] for row in rows),
        "amthal_proofs": sum(row["amthal_proof"] for row in rows),
        "lp_eligible": len(eligible),
        "lp_model_statuses": dict(
            sorted(Counter(row["lp_model_status"] for row in rows).items())
        ),
        "lp_certified": len(certified),
        "lp_invalid_certified_bounds": sum(row["certified_bound_valid"] is False for row in rows),
        "lp_lifted_models": len(lifted),
        "lp_lifted_certified": sum(row["lp_root_bound"] is not None for row in lifted),
        "lp_exact_at_incumbent": sum(row["root_bound_matches_incumbent"] for row in certified),
        "lp_within_10_percent": sum(
            row["certified_relative_gap"] is not None
            and row["certified_relative_gap"] <= 0.1
            for row in certified
        ),
        "lp_timeouts": sum(row["lp_timeouts"] for row in rows),
        "lp_node_prunes": sum(row["lp_node_prunes"] for row in rows),
        "median_lp_time_ms_eligible": median(row["lp_time_ms"] for row in eligible),
        "median_lp_columns_eligible": median(row["lp_columns"] for row in eligible),
        "median_lp_auxiliary_columns_eligible": median(
            row["lp_auxiliary_columns"] for row in eligible
        ),
        "median_lp_auxiliary_columns_lifted": median(
            row["lp_auxiliary_columns"] for row in lifted
        ),
        "lp_objective_fallbacks": sum(row["lp_objective_fallbacks"] for row in rows),
        "median_lp_retained_row_fraction": median(
            row["lp_rows"] / row["lp_source_rows"]
            for row in eligible
            if row["lp_source_rows"] > 0
        ),
        "median_certified_relative_gap": median(row["certified_relative_gap"] for row in certified),
        "median_wall_ratio": median(row["wall_ratio"] for row in rows),
        "median_first_incumbent_ratio": median(row["first_incumbent_ratio"] for row in rows),
        "median_best_incumbent_ratio": median(row["best_incumbent_ratio"] for row in rows),
        "median_node_ratio": median(row["node_ratio"] for row in rows),
        "families": family_rows,
        "details": rows,
    }


def comparison_projection(
    summary: dict[str, Any], control: str, treatment: str
) -> dict[str, Any]:
    outcomes = summary["outcomes"]
    return {
        "control": control,
        "treatment": treatment,
        "pairs": summary["pairs"],
        "control_verified_incumbents": summary["native_verified_incumbents"],
        "treatment_verified_incumbents": summary["amthal_verified_incumbents"],
        "control_proofs": summary["native_proofs"],
        "treatment_proofs": summary["amthal_proofs"],
        "treatment_wins": outcomes.get("amthal", 0),
        "control_wins": outcomes.get("native", 0),
        "ties": outcomes.get("tie", 0),
        "neither": outcomes.get("neither", 0),
        "median_wall_ratio": summary["median_wall_ratio"],
        "median_first_incumbent_ratio": summary["median_first_incumbent_ratio"],
        "median_best_incumbent_ratio": summary["median_best_incumbent_ratio"],
        "median_node_ratio": summary["median_node_ratio"],
        "treatment_lp_node_prunes": summary["lp_node_prunes"],
    }


def attach_attribution(
    summary: dict[str, Any],
    records: Sequence[dict[str, Any]],
    native_name: str,
    bound_only_name: str,
    amthal_name: str,
    search_name: Optional[str] = None,
) -> None:
    bound_pairs = pair_records(records, native_name, bound_only_name)
    phase_pairs = pair_records(records, bound_only_name, amthal_name)
    bound_summary = summarize_pairs(bound_pairs)
    phase_summary = summarize_pairs(phase_pairs)
    comparisons = [
        comparison_projection(summary, native_name, search_name or amthal_name),
        comparison_projection(bound_summary, native_name, bound_only_name),
        comparison_projection(phase_summary, bound_only_name, amthal_name),
    ]
    if search_name is not None:
        search_pairs = pair_records(records, amthal_name, search_name)
        search_summary = summarize_pairs(search_pairs)
        comparisons.append(comparison_projection(search_summary, amthal_name, search_name))

    mismatches = 0
    for (_native, bound_only), (_bound_only, full) in zip(bound_pairs, phase_pairs):
        left = parse_solver_stats(bound_only)
        right = parse_solver_stats(full)
        fields = ("lp_rows", "lp_root_bound", "lp_solves", "lp_certified")
        mismatches += any(left[field] != right[field] for field in fields)
    summary["comparisons"] = comparisons
    summary["bound_reproducibility_mismatches"] = mismatches


def shown(value: Optional[float], *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if percent:
        return f"{100 * value:.2f}%"
    return f"{value:.3f}"


def markdown(summary: dict[str, Any]) -> str:
    outcomes = summary["outcomes"]
    lines = [
        "# Qayd LP ablation",
        "",
        f"Paired runs: {summary['pairs']}. Same executable: {'yes' if summary['same_binary'] else 'no'}.",
        "",
        "## Overall",
        "",
        f"- Verified incumbents: native {summary['native_verified_incumbents']}, Amthal {summary['amthal_verified_incumbents']}.",
        f"- Proofs: native {summary['native_proofs']}, Amthal {summary['amthal_proofs']}.",
        f"- Objective outcomes: Amthal {outcomes.get('amthal', 0)}, native {outcomes.get('native', 0)}, ties {outcomes.get('tie', 0)}, neither {outcomes.get('neither', 0)}.",
        f"- LP eligibility: {summary['lp_eligible']}/{summary['pairs']}; certified root bounds: {summary['lp_certified']}/{summary['pairs']}; timeouts: {summary['lp_timeouts']}.",
        f"- Certified bounds crossing a verified incumbent: {summary['lp_invalid_certified_bounds']}.",
        f"- LP model outcomes: {summary['lp_model_statuses']}.",
        f"- Median active LP columns: {shown(summary['median_lp_columns_eligible'])}; median retained-row fraction: {shown(summary['median_lp_retained_row_fraction'], percent=True)}.",
        f"- Nonlinear extended formulations: {summary['lp_lifted_models']}/{summary['pairs']}; certified: {summary['lp_lifted_certified']}/{summary['lp_lifted_models']}; median auxiliary columns when lifted: {shown(summary['median_lp_auxiliary_columns_lifted'])}; interval fallback terms: {summary['lp_objective_fallbacks']}.",
        f"- Exactly certified in-search LP prunes: {summary['lp_node_prunes']}.",
        f"- Root bound quality: {summary['lp_exact_at_incumbent']} match the incumbent; {summary['lp_within_10_percent']} are within 10%.",
        f"- Median LP time on eligible models: {shown(summary['median_lp_time_ms_eligible'])} ms.",
        f"- Median certified relative gap: {shown(summary['median_certified_relative_gap'], percent=True)}.",
        f"- Median Amthal/native wall ratio: {shown(summary['median_wall_ratio'])}.",
        f"- Median first-incumbent time ratio: {shown(summary['median_first_incumbent_ratio'])}.",
        f"- Median best-incumbent time ratio: {shown(summary['median_best_incumbent_ratio'])}.",
        f"- Median searched-node ratio: {shown(summary['median_node_ratio'])}.",
        "",
        "Ratios below 1 favor Amthal. Objective outcomes use only checker-accepted incumbents.",
    ]
    comparisons = summary.get("comparisons")
    if isinstance(comparisons, list):
        lines.extend(
            [
                "",
                "## Attribution",
                "",
                "The bound-only arm disables LP primal phase guidance. It isolates dual-bound cost and availability from the search-phase effect.",
                "",
                "| Treatment vs control | Treatment wins | Control wins | Ties | Neither | Verified incumbents | Proofs | LP node prunes | Wall ratio | First ratio | Best ratio | Nodes ratio |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in comparisons:
            lines.append(
                f"| {row['treatment']} vs {row['control']} | "
                f"{row['treatment_wins']} | {row['control_wins']} | "
                f"{row['ties']} | {row['neither']} | "
                f"{row['treatment_verified_incumbents']}/{row['control_verified_incumbents']} | "
                f"{row['treatment_proofs']}/{row['control_proofs']} | "
                f"{row['treatment_lp_node_prunes']} | "
                f"{shown(row['median_wall_ratio'])} | "
                f"{shown(row['median_first_incumbent_ratio'])} | "
                f"{shown(row['median_best_incumbent_ratio'])} | "
                f"{shown(row['median_node_ratio'])} |"
            )
        lines.extend(
            [
                "",
                f"Root-stat mismatches between the two Amthal arms: {summary['bound_reproducibility_mismatches']}.",
            ]
        )

    lines.extend(
        [
            "",
            "## By family",
            "",
            "| Family | Pairs | LP eligible | Certified | Amthal wins | Native wins | Ties |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["families"]:
        lines.append(
            f"| {row['family']} | {row['pairs']} | {row['lp_eligible']} | "
            f"{row['lp_certified']} | {row['amthal_wins']} | "
            f"{row['native_wins']} | {row['ties']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--native", default="qayd-native")
    parser.add_argument("--amthal", default="qayd-amthal")
    parser.add_argument("--search", default="qayd-amthal-search")
    parser.add_argument(
        "--bound-only",
        default="qayd-amthal-bound-only",
        help="optional third arm with LP phase guidance disabled",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    try:
        records = load_records(args.results)
        solvers = {record.get("solver") for record in records}
        treatment = args.search if args.search in solvers else args.amthal
        pairs = pair_records(records, args.native, treatment)
        summary = summarize_pairs(pairs)
        if args.bound_only in solvers:
            attach_attribution(
                summary,
                records,
                args.native,
                args.bound_only,
                args.amthal,
                args.search if args.search in solvers else None,
            )
        report = markdown(summary)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
        else:
            print(report, end="")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except (AblationError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
