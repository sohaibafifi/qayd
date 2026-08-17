"""Shared finite-gate circuit helpers for CSPLib diagnosis models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product

import qayd as cp


@dataclass(frozen=True)
class Gate:
    operation: str
    inputs: tuple[int, ...]


@dataclass(frozen=True)
class Circuit:
    input_count: int
    gates: tuple[Gate, ...]
    outputs: tuple[int, ...]


def parse_circuit(raw: dict) -> Circuit:
    try:
        gates = tuple(
            Gate(str(gate["op"]).upper(), tuple(int(value) for value in gate["inputs"]))
            for gate in raw["gates"]
        )
        circuit = Circuit(
            int(raw["input_count"]),
            gates,
            tuple(int(value) for value in raw["outputs"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid circuit JSON") from error
    validate_circuit(circuit)
    return circuit


def validate_circuit(circuit: Circuit) -> None:
    arities = {"NOT": 1, "AND": 2, "OR": 2, "XOR": 2, "NAND": 2, "NOR": 2}
    if circuit.input_count < 1 or not circuit.gates or not circuit.outputs:
        raise ValueError("circuit inputs, gates, and outputs must be non-empty")
    for gate_index, gate in enumerate(circuit.gates):
        if gate.operation not in arities or len(gate.inputs) != arities[gate.operation]:
            raise ValueError("a gate operation or arity is invalid")
        output_wire = circuit.input_count + gate_index
        if any(wire < 0 or wire >= output_wire for wire in gate.inputs):
            raise ValueError("gate inputs must reference earlier wires")
    wire_count = circuit.input_count + len(circuit.gates)
    if any(wire < 0 or wire >= wire_count for wire in circuit.outputs):
        raise ValueError("a circuit output wire is invalid")


def operation_value(operation: str, inputs: tuple[int, ...]) -> int:
    if operation == "NOT":
        return 1 - inputs[0]
    if operation == "AND":
        return inputs[0] & inputs[1]
    if operation == "OR":
        return inputs[0] | inputs[1]
    if operation == "XOR":
        return inputs[0] ^ inputs[1]
    if operation == "NAND":
        return 1 - (inputs[0] & inputs[1])
    if operation == "NOR":
        return 1 - (inputs[0] | inputs[1])
    raise ValueError(f"unknown gate operation {operation}")


def post_circuit(
    model: cp.Model,
    circuit: Circuit,
    inputs: list[cp.IntVar],
    faults: list[cp.IntVar] | tuple[int, ...],
    *,
    prefix: str,
) -> tuple[list[cp.IntVar], list[cp.IntVar]]:
    wires = [*inputs]
    gate_outputs = []
    for gate_index, gate in enumerate(circuit.gates):
        output = model.bool_var(name=f"{prefix}_gate_{gate_index}")
        gate_inputs = [wires[wire] for wire in gate.inputs]
        fault = faults[gate_index]
        if isinstance(fault, int):
            allowed = [
                (
                    *values,
                    fault
                    if fault in (0, 1)
                    else operation_value(gate.operation, values),
                )
                for values in product((0, 1), repeat=len(gate.inputs))
            ]
            model.table([*gate_inputs, output], allowed)
        else:
            allowed = [
                (
                    *values,
                    fault_value,
                    fault_value
                    if fault_value in (0, 1)
                    else operation_value(gate.operation, values),
                )
                for values in product((0, 1), repeat=len(gate.inputs))
                for fault_value in (-1, 0, 1)
            ]
            model.table([*gate_inputs, fault, output], allowed)
        wires.append(output)
        gate_outputs.append(output)
    return wires, [wires[index] for index in circuit.outputs]


def evaluate(
    circuit: Circuit, inputs: tuple[int, ...], faults: tuple[int, ...]
) -> tuple[int, ...]:
    wires = list(inputs)
    for gate, fault in zip(circuit.gates, faults):
        values = tuple(wires[index] for index in gate.inputs)
        wires.append(
            fault if fault in (0, 1) else operation_value(gate.operation, values)
        )
    return tuple(wires[index] for index in circuit.outputs)


def parse_json(data: str | bytes) -> dict:
    return json.loads(data)
