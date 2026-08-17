"""CSPLib prob042: model-based diagnosis of digital circuits.

Specification: https://www.csplib.org/Problems/prob042/
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
    "inputs": [0, 0],
    "observed": [1, 0],
}


@dataclass(frozen=True)
class DiagnosisInstance:
    circuit: Circuit
    inputs: tuple[int, ...]
    observed: tuple[int, ...]


@dataclass(frozen=True)
class DiagnosisModel:
    model: cp.Model
    instance: DiagnosisInstance
    faults: list[cp.IntVar]


def parse_instance(data: str | bytes) -> DiagnosisInstance:
    raw = parse_json(data)
    try:
        instance = DiagnosisInstance(
            parse_circuit(raw["circuit"]),
            tuple(int(value) for value in raw["inputs"]),
            tuple(int(value) for value in raw["observed"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid diagnosis JSON instance") from error
    if len(instance.inputs) != instance.circuit.input_count or len(
        instance.observed
    ) != len(instance.circuit.outputs):
        raise ValueError("circuit input or observation dimensions are invalid")
    if any(value not in (0, 1) for value in (*instance.inputs, *instance.observed)):
        raise ValueError("inputs and observations must be binary")
    return instance


def load_instance(path: str | Path) -> DiagnosisInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(
    instance: DiagnosisInstance, *, exact_faults: int | None = None
) -> DiagnosisModel:
    if exact_faults is not None and not 0 <= exact_faults <= len(
        instance.circuit.gates
    ):
        raise ValueError("exact_faults is invalid")
    model = cp.Model()
    input_variables = [
        model.bool_var(name=f"input_{index}") for index in range(len(instance.inputs))
    ]
    for variable, value in zip(input_variables, instance.inputs):
        model.add(variable == value)
    faults = model.int_vars(len(instance.circuit.gates), -1, 1, name="fault")
    faulted = []
    for index, fault in enumerate(faults):
        model.table([fault], [(-1,), (0,), (1,)])
        indicator = model.bool_var(name=f"faulted_{index}")
        model.table([fault, indicator], [(-1, 0), (0, 1), (1, 1)])
        faulted.append(indicator)
    _, outputs = post_circuit(
        model, instance.circuit, input_variables, faults, prefix="diagnosis"
    )
    for output, observed in zip(outputs, instance.observed):
        model.add(output == observed)
    if exact_faults is None:
        model.minimize(sum(faulted))
    else:
        model.add(sum(faulted) == exact_faults)
    return DiagnosisModel(model, instance, faults)


def decode(built: DiagnosisModel, solution: cp.Solution) -> tuple[int, ...]:
    return tuple(values(solution, built.faults))


def validate(
    built: DiagnosisModel, faults: tuple[int, ...], objective: int | None = None
) -> None:
    if len(faults) != len(built.instance.circuit.gates) or any(
        value not in (-1, 0, 1) for value in faults
    ):
        raise AssertionError("the decoded fault vector is invalid")
    if (
        evaluate(built.instance.circuit, built.instance.inputs, faults)
        != built.instance.observed
    ):
        raise AssertionError("the faults do not explain the observations")
    if objective is not None and sum(value != -1 for value in faults) != objective:
        raise AssertionError("the objective does not match the fault cardinality")


def enumerate_minimal_diagnoses(
    instance: DiagnosisInstance, *, time_limit: int = 30
) -> list[tuple[int, ...]]:
    optimum_model = build_model(instance)
    optimum_solution = optimum_model.model.solve(
        engine="exact", time_limit=time_limit, seed=0, threads=1
    )
    if optimum_solution.status != "OPTIMAL":
        raise RuntimeError(
            f"could not prove minimum fault cardinality: {optimum_solution.status}"
        )
    cardinality = int(optimum_solution.objective)
    built = build_model(instance, exact_faults=cardinality)
    diagnoses = []
    while True:
        solution = built.model.solve(
            engine="exact", time_limit=time_limit, seed=0, threads=1
        )
        if solution.status == "UNSATISFIABLE":
            break
        if not solution.is_sat():
            raise RuntimeError(
                f"diagnosis enumeration stopped with status {solution.status}"
            )
        faults = decode(built, solution)
        validate(built, faults)
        diagnoses.append(faults)
        built.model.add(
            cp.any([variable != value for variable, value in zip(built.faults, faults)])
        )
    return sorted(diagnoses)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON diagnosis instance")
    parser.add_argument(
        "--all", action="store_true", help="enumerate all minimum-cardinality diagnoses"
    )
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    if args.all:
        diagnoses = enumerate_minimal_diagnoses(instance, time_limit=args.time_limit)
        print(f"prob042 diagnoses={diagnoses}")
        return 0
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob042 status={solution.status}")
    if not solution.is_sat():
        return 1
    faults = decode(built, solution)
    validate(built, faults, solution.objective)
    print(f"faults={faults} cardinality={solution.objective}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
