import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.fastcop import anytime as fastcop_anytime
from bench.fastcop import score as fastcop_score


REPO_ROOT = Path(__file__).resolve().parents[2]


def result(
    solver,
    instance,
    sense,
    events,
    *,
    status="SAT",
    proof=False,
    proof_time=None,
    elapsed=10.0,
):
    best = None
    if events:
        choose = min if sense == "min" else max
        value = choose(event["value"] for event in events)
        matching = next(event for event in events if event["value"] == value)
        best = dict(matching)
    return {
        "schema": fastcop_score.RESULT_SCHEMA,
        "run_key": f"{solver}-{instance}",
        "solver": solver,
        "instance": instance,
        "family": "F",
        "family_group": "F",
        "objective_sense": sense,
        "status": status,
        "proof": proof,
        "proof_elapsed_seconds": proof_time,
        "incumbents": events,
        "best_incumbent": best,
        "validation": {
            "valid": True if best is not None else None,
            "expected_objective": best["value"] if best is not None else None,
            "reported_objective": best["value"] if best is not None else None,
        },
        "invalid": False,
        "returncode": 0,
        "execution_error": None,
        "timed_out": False,
        "killed": False,
        "elapsed_wall_seconds": elapsed,
    }


@pytest.mark.parametrize(
    ("sense", "events", "expected_at_three", "expected_best"),
    [
        (
            "min",
            [
                {"value": 5, "elapsed_seconds": 4},
                {"value": 10, "elapsed_seconds": 1},
                {"value": 8, "elapsed_seconds": 2},
                {"value": 9, "elapsed_seconds": 3},
            ],
            8,
            5,
        ),
        (
            "max",
            [
                {"value": 9, "elapsed_seconds": 4},
                {"value": 4, "elapsed_seconds": 1},
                {"value": 7, "elapsed_seconds": 2},
                {"value": 6, "elapsed_seconds": 3},
            ],
            7,
            9,
        ),
    ],
)
def test_best_so_far_uses_sense_and_timestamps_not_event_order(
    sense, events, expected_at_three, expected_best
):
    trace = fastcop_anytime.record_timeline(
        result("a", "i", sense, events), [3, 5]
    )

    assert [state["objective"] for state in trace["checkpoints"]] == [
        expected_at_three,
        expected_best,
    ]
    assert trace["first_incumbent_seconds"] == 1
    assert trace["best_incumbent_seconds"] == 4
    assert trace["last_improvement_seconds"] == 4
    assert trace["improvement_count"] == 3
    assert trace["strict_improvement_count"] == 2
    assert trace["plateau_seconds"] == 6
    assert trace["checkpoints"][0]["plateau_seconds"] == 1


def test_proof_and_dynamic_scores_appear_only_at_proof_timestamp():
    records = [
        result(
            "a",
            "i",
            "min",
            [
                {"value": 12, "elapsed_seconds": 1},
                {"value": 10, "elapsed_seconds": 2},
            ],
            status="OPTIMUM",
            proof=True,
            proof_time=5,
        ),
        result(
            "b",
            "i",
            "min",
            [{"value": 10, "elapsed_seconds": 1}],
        ),
    ]

    report = fastcop_anytime.analyze_records(
        records, [3, 5], mode="both", invalidation="none"
    )
    early, proved = report["checkpoint_reports"]
    a_states = report["runs"][0]["checkpoints"]

    assert a_states[0]["status"] == "SAT"
    assert a_states[0]["proof"] is False
    assert a_states[1]["status"] == "OPTIMUM"
    assert a_states[1]["proof"] is True
    assert early["scores"]["pool"]["solvers"]["a"]["classes"] == {"BB1": 1}
    assert early["scores"]["pool"]["solvers"]["b"]["classes"] == {"BB1": 1}
    assert proved["scores"]["pool"]["solvers"]["a"]["classes"] == {"Opt": 1}
    assert proved["scores"]["pool"]["solvers"]["b"]["classes"] == {"BB2": 1}
    assert proved["scores"]["pairwise"]["a__vs__b"]["solvers"]["b"][
        "score"
    ] == 0.5


def test_missing_proof_timestamp_never_backdates_a_proof():
    record = result(
        "a",
        "unsat",
        "min",
        [],
        status="UNSAT",
        proof=True,
        proof_time=None,
    )

    trace = fastcop_anytime.record_timeline(record, [1, 100])

    assert trace["proof"] is False
    assert [state["proof"] for state in trace["checkpoints"]] == [False, False]
    assert [state["status"] for state in trace["checkpoints"]] == [
        "UNKNOWN",
        "UNKNOWN",
    ]


def test_checkpoint_parser_rejects_ambiguous_or_non_finite_values():
    assert fastcop_anytime.parse_checkpoints(["10, 1", "5", "1"]) == [
        1.0,
        5.0,
        10.0,
    ]
    for values in (["1,,2"], ["nan"], ["-1"]):
        with pytest.raises(fastcop_anytime.AnytimeError):
            fastcop_anytime.parse_checkpoints(values)


def test_cli_emits_json_and_text_summary(tmp_path):
    input_path = tmp_path / "results.jsonl"
    input_path.write_text(
        json.dumps(
            result(
                "solver",
                "instance",
                "max",
                [{"value": 4, "elapsed_seconds": 2}],
            )
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "bench" / "fastcop" / "anytime.py"),
            str(input_path),
            "--checkpoints",
            "1,3",
            "--mode",
            "pool",
            "--invalidation",
            "none",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(completed.stdout)
    assert report["schema"] == fastcop_anytime.ANYTIME_SCHEMA
    assert report["checkpoints_seconds"] == [1.0, 3.0]
    assert report["checkpoint_reports"][0]["solvers"]["solver"][
        "runs_with_incumbent"
    ] == 0
    assert report["checkpoint_reports"][1]["solvers"]["solver"][
        "runs_with_incumbent"
    ] == 1
    assert "Anytime checkpoints: 1s, 3s" in completed.stderr
    assert "Checkpoint 3s" in completed.stderr
