"""CSPLib prob082: patient transportation.

Specification: https://www.csplib.org/Problems/prob082/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "max_wait_time": 20,
    "same_vehicle_backward": True,
    "distances": [[0, 10, 20, 15], [10, 0, 10, 10], [20, 10, 0, 10], [15, 10, 10, 0]],
    "vehicles": [
        {"can_take": [0, 1], "capacity": 2, "availability": [[0, 100]]},
        {"can_take": [0], "capacity": 1, "availability": [[10, 90]]},
    ],
    "patients": [
        {
            "category": 0,
            "load": 1,
            "start": 0,
            "destination": 1,
            "end": 0,
            "appointment": 35,
            "appointment_duration": 15,
            "service": 2,
            "priority": 2,
        },
        {
            "category": 1,
            "load": 1,
            "start": 2,
            "destination": 1,
            "end": -1,
            "appointment": 45,
            "appointment_duration": 10,
            "service": 1,
            "priority": 1,
        },
        {
            "category": 0,
            "load": 1,
            "start": -1,
            "destination": 1,
            "end": 3,
            "appointment": 25,
            "appointment_duration": 15,
            "service": 1,
            "priority": 1,
        },
    ],
}


@dataclass(frozen=True)
class TransportVehicle:
    can_take: frozenset[int]
    capacity: int
    availability: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Patient:
    category: int
    load: int
    start: int
    destination: int
    end: int
    appointment: int
    appointment_duration: int
    service: int
    priority: int


@dataclass(frozen=True)
class Trip:
    patient: int
    origin: int
    destination: int
    earliest: int
    latest_drop: int
    service: int


@dataclass(frozen=True)
class PatientTransportInstance:
    max_wait_time: int
    same_vehicle_backward: bool
    distances: tuple[tuple[int, ...], ...]
    vehicles: tuple[TransportVehicle, ...]
    patients: tuple[Patient, ...]


@dataclass(frozen=True)
class PatientTransportModel:
    model: cp.Model
    instance: PatientTransportInstance
    trips: tuple[Trip, ...]
    trip_indices: tuple[tuple[int, ...], ...]
    assignments: list[cp.IntVar]
    starts: list[cp.IntVar]
    served: list[cp.IntVar]


def parse_instance(data: str | bytes) -> PatientTransportInstance:
    raw = json.loads(data)
    try:
        vehicles = tuple(
            TransportVehicle(
                frozenset(int(value) for value in item["can_take"]),
                int(item["capacity"]),
                tuple(
                    (int(window[0]), int(window[1])) for window in item["availability"]
                ),
            )
            for item in raw["vehicles"]
        )
        patients = tuple(
            Patient(
                int(item["category"]),
                int(item["load"]),
                int(item["start"]),
                int(item["destination"]),
                int(item["end"]),
                int(item["appointment"]),
                int(item["appointment_duration"]),
                int(item.get("service", 0)),
                int(item.get("priority", 1)),
            )
            for item in raw["patients"]
        )
        return PatientTransportInstance(
            int(raw["max_wait_time"]),
            bool(raw.get("same_vehicle_backward", False)),
            tuple(tuple(int(value) for value in row) for row in raw["distances"]),
            vehicles,
            patients,
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("invalid patient-transportation JSON instance") from error


def load_instance(path: str | Path) -> PatientTransportInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def make_trips(
    instance: PatientTransportInstance,
) -> tuple[tuple[Trip, ...], tuple[tuple[int, ...], ...]]:
    trips = []
    indices = []
    for patient_index, patient in enumerate(instance.patients):
        patient_trips = []
        if patient.start >= 0:
            patient_trips.append(len(trips))
            trips.append(
                Trip(
                    patient_index,
                    patient.start,
                    patient.destination,
                    patient.appointment - instance.max_wait_time,
                    patient.appointment,
                    patient.service,
                )
            )
        if patient.end >= 0:
            ready = patient.appointment + patient.appointment_duration
            patient_trips.append(len(trips))
            trips.append(
                Trip(
                    patient_index,
                    patient.destination,
                    patient.end,
                    ready,
                    ready + instance.max_wait_time,
                    patient.service,
                )
            )
        indices.append(tuple(patient_trips))
    return tuple(trips), tuple(indices)


def trip_duration(instance: PatientTransportInstance, trip: Trip) -> int:
    return trip.service * 2 + instance.distances[trip.origin][trip.destination]


def build_model(instance: PatientTransportInstance) -> PatientTransportModel:
    location_count = len(instance.distances)
    if location_count < 1 or any(
        len(row) != location_count for row in instance.distances
    ):
        raise ValueError("distances must be a square matrix")
    if instance.max_wait_time < 0 or not instance.vehicles or not instance.patients:
        raise ValueError("vehicles, patients, and maximum wait are invalid")
    trips, trip_indices = make_trips(instance)
    if any(not indices for indices in trip_indices):
        raise ValueError("each patient must request at least one trip")
    vehicle_count = len(instance.vehicles)
    sentinel = vehicle_count
    model = cp.Model()
    served = [
        model.bool_var(name=f"served_{patient}")
        for patient in range(len(instance.patients))
    ]
    assignments = []
    starts = []
    domains = []
    for trip_index, trip in enumerate(trips):
        patient = instance.patients[trip.patient]
        duration = trip_duration(instance, trip)
        latest_start = trip.latest_drop - duration
        if trip.earliest > latest_start:
            raise ValueError("a patient trip has an empty time window")
        compatible = [
            vehicle
            for vehicle, item in enumerate(instance.vehicles)
            if patient.category in item.can_take and patient.load <= item.capacity
        ]
        domains.append([*compatible, sentinel])
        assignment = model.int_var(0, sentinel, name=f"vehicle_{trip_index}")
        start = model.int_var(trip.earliest, latest_start, name=f"start_{trip_index}")
        model.table([assignment], [(vehicle,) for vehicle in domains[-1]])
        allowed = []
        for vehicle in compatible:
            for candidate in range(trip.earliest, latest_start + 1):
                if any(
                    low <= candidate and candidate + duration <= high
                    for low, high in instance.vehicles[vehicle].availability
                ):
                    allowed.append((vehicle, candidate, 1))
        allowed.extend(
            (sentinel, candidate, 0)
            for candidate in range(trip.earliest, latest_start + 1)
        )
        model.table([assignment, start, served[trip.patient]], allowed)
        assignments.append(assignment)
        starts.append(start)
    if instance.same_vehicle_backward:
        for indices in trip_indices:
            if len(indices) == 2:
                model.add(assignments[indices[0]] == assignments[indices[1]])
    for first in range(len(trips)):
        left_duration = trip_duration(instance, trips[first])
        left_range = range(
            trips[first].earliest, trips[first].latest_drop - left_duration + 1
        )
        for second in range(first + 1, len(trips)):
            right_duration = trip_duration(instance, trips[second])
            right_range = range(
                trips[second].earliest, trips[second].latest_drop - right_duration + 1
            )
            allowed = []
            for left_vehicle in domains[first]:
                for right_vehicle in domains[second]:
                    for left_start in left_range:
                        for right_start in right_range:
                            separated = (
                                left_start
                                + left_duration
                                + instance.distances[trips[first].destination][
                                    trips[second].origin
                                ]
                                <= right_start
                                or right_start
                                + right_duration
                                + instance.distances[trips[second].destination][
                                    trips[first].origin
                                ]
                                <= left_start
                            )
                            if (
                                left_vehicle == sentinel
                                or right_vehicle == sentinel
                                or left_vehicle != right_vehicle
                                or separated
                            ):
                                allowed.append(
                                    (
                                        left_vehicle,
                                        left_start,
                                        right_vehicle,
                                        right_start,
                                    )
                                )
            model.table(
                [
                    assignments[first],
                    starts[first],
                    assignments[second],
                    starts[second],
                ],
                allowed,
            )
    model.maximize(
        sum(
            patient.priority * served[index]
            for index, patient in enumerate(instance.patients)
        )
    )
    return PatientTransportModel(
        model, instance, trips, trip_indices, assignments, starts, served
    )


def decode(
    built: PatientTransportModel, solution: cp.Solution
) -> tuple[list[int], list[int], list[int]]:
    return (
        values(solution, built.assignments),
        values(solution, built.starts),
        values(solution, built.served),
    )


def validate(
    built: PatientTransportModel,
    result: tuple[list[int], list[int], list[int]],
    objective: int | None,
) -> None:
    assignments, starts, served = result
    sentinel = len(built.instance.vehicles)
    for trip_index, trip in enumerate(built.trips):
        duration = trip_duration(built.instance, trip)
        if bool(assignments[trip_index] != sentinel) != bool(served[trip.patient]):
            raise AssertionError("a patient is only partially served")
        if assignments[trip_index] != sentinel:
            vehicle = built.instance.vehicles[assignments[trip_index]]
            patient = built.instance.patients[trip.patient]
            if (
                patient.category not in vehicle.can_take
                or patient.load > vehicle.capacity
            ):
                raise AssertionError("a vehicle is incompatible with a patient")
            if not any(
                low <= starts[trip_index] and starts[trip_index] + duration <= high
                for low, high in vehicle.availability
            ):
                raise AssertionError("a trip lies outside vehicle availability")
    for first in range(len(built.trips)):
        for second in range(first + 1, len(built.trips)):
            if (
                assignments[first] == sentinel
                or assignments[first] != assignments[second]
            ):
                continue
            left_end = starts[first] + trip_duration(built.instance, built.trips[first])
            right_end = starts[second] + trip_duration(
                built.instance, built.trips[second]
            )
            if not (
                left_end
                + built.instance.distances[built.trips[first].destination][
                    built.trips[second].origin
                ]
                <= starts[second]
                or right_end
                + built.instance.distances[built.trips[second].destination][
                    built.trips[first].origin
                ]
                <= starts[first]
            ):
                raise AssertionError("two vehicle trips overlap")
    reward = sum(
        patient.priority * served[index]
        for index, patient in enumerate(built.instance.patients)
    )
    if objective is not None and reward != objective:
        raise AssertionError("the objective does not match satisfied requests")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON patient-transportation instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob082 patients={len(instance.patients)} status={solution.status}")
    if not solution.is_sat():
        return 1
    result = decode(built, solution)
    validate(built, result, solution.objective)
    print(
        f"reward={solution.objective} vehicles={result[0]} starts={result[1]} served={result[2]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
