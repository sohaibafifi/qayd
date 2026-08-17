"""CSPLib prob043: differential diagnosis of digital circuits.

Specification: https://www.csplib.org/Problems/prob043/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values
from .circuit_common import Circuit, evaluate, parse_circuit, parse_json, post_circuit

DEFAULT_INSTANCE = {
    "circuit": {
        "input_count": 2,
        "gates": [
            {"op": "XOR", "inputs": [0, 1]},
            {"op": "AND", "inputs": [0, 1]},
        ],
        "outputs": [2, 3],
    },
    "diagnosis_a": [1, -1],
    "diagnosis_b": [-1, 1],
}


@dataclass(frozen=True)
class DifferentialInstance:
    circuit: Circuit
    diagnosis_a: tuple[int, ...]
    diagnosis_b: tuple[int, ...]


@dataclass(frozen=True)
class DifferentialModel:
    model: cp.Model
    instance: DifferentialInstance
    inputs: list[cp.IntVar]
    outputs_a: list[cp.IntVar]
    outputs_b: list[cp.IntVar]


def parse_instance(data: str | bytes) -> DifferentialInstance:
    raw = parse_json(data)
    try:
        instance = DifferentialInstance(
            parse_circuit(raw["circuit"]),
            tuple(int(value) for value in raw["diagnosis_a"]),
            tuple(int(value) for value in raw["diagnosis_b"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid differential-diagnosis JSON instance") from error
    gate_count = len(instance.circuit.gates)
    if (
        len(instance.diagnosis_a) != gate_count
        or len(instance.diagnosis_b) != gate_count
    ):
        raise ValueError("diagnosis dimensions do not match the circuit")
    if any(
        value not in (-1, 0, 1)
        for value in (*instance.diagnosis_a, *instance.diagnosis_b)
    ):
        raise ValueError("fault values must be healthy, stuck-at-0, or stuck-at-1")
    return instance


def load_instance(path: str | Path) -> DifferentialInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: DifferentialInstance) -> DifferentialModel:
    model = cp.Model()
    inputs = [
        model.bool_var(name=f"input_{index}")
        for index in range(instance.circuit.input_count)
    ]
    _, outputs_a = post_circuit(
        model,
        instance.circuit,
        inputs,
        instance.diagnosis_a,
        prefix="diagnosis_a",
    )
    _, outputs_b = post_circuit(
        model,
        instance.circuit,
        inputs,
        instance.diagnosis_b,
        prefix="diagnosis_b",
    )
    differences = []
    for index, (left, right) in enumerate(zip(outputs_a, outputs_b)):
        different = model.bool_var(name=f"output_difference_{index}")
        model.table(
            [left, right, different],
            [(a, b, int(a != b)) for a in (0, 1) for b in (0, 1)],
        )
        differences.append(different)
    model.add(sum(differences) >= 1)
    return DifferentialModel(model, instance, inputs, outputs_a, outputs_b)


def decode(
    built: DifferentialModel,
    solution: cp.Solution,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(values(solution, built.inputs)),
        tuple(values(solution, built.outputs_a)),
        tuple(values(solution, built.outputs_b)),
    )


def validate(
    built: DifferentialModel,
    inputs: tuple[int, ...],
    outputs_a: tuple[int, ...],
    outputs_b: tuple[int, ...],
) -> None:
    expected_a = evaluate(built.instance.circuit, inputs, built.instance.diagnosis_a)
    expected_b = evaluate(built.instance.circuit, inputs, built.instance.diagnosis_b)
    if outputs_a != expected_a or outputs_b != expected_b:
        raise AssertionError("decoded circuit outputs are inconsistent")
    if outputs_a == outputs_b:
        raise AssertionError("the test vector does not distinguish the diagnoses")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON differential-diagnosis instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob043 status={solution.status}")
    if not solution.is_sat():
        return 1
    inputs, outputs_a, outputs_b = decode(built, solution)
    validate(built, inputs, outputs_a, outputs_b)
    print(f"inputs={inputs} outputs_a={outputs_a} outputs_b={outputs_b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
