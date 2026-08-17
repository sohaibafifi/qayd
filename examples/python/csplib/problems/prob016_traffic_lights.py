"""CSPLib prob016: traffic lights.

Specification: https://www.csplib.org/Problems/prob016/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

VEHICLE_NAMES = ("red", "red-yellow", "green", "yellow")
PEDESTRIAN_NAMES = ("red", "green")
ALLOWED = (
    (0, 0, 2, 1),
    (1, 0, 3, 0),
    (2, 1, 0, 0),
    (3, 0, 1, 0),
)


@dataclass(frozen=True)
class TrafficLightsModel:
    model: cp.Model
    vehicles: list[cp.IntVar]
    pedestrians: list[cp.IntVar]


def build_model() -> TrafficLightsModel:
    model = cp.Model()
    vehicles = model.int_vars(4, 0, 3, name="vehicle")
    pedestrians = model.int_vars(4, 0, 1, name="pedestrian")
    for index in range(4):
        following = (index + 1) % 4
        model.table(
            [
                vehicles[index],
                pedestrians[index],
                vehicles[following],
                pedestrians[following],
            ],
            ALLOWED,
        )
    return TrafficLightsModel(model, vehicles, pedestrians)


def decode(
    built: TrafficLightsModel, solution: cp.Solution
) -> tuple[list[str], list[str]]:
    vehicles = [VEHICLE_NAMES[value] for value in values(solution, built.vehicles)]
    pedestrians = [
        PEDESTRIAN_NAMES[value] for value in values(solution, built.pedestrians)
    ]
    return vehicles, pedestrians


def validate(vehicles: list[str], pedestrians: list[str]) -> None:
    if len(vehicles) != 4 or len(pedestrians) != 4:
        raise AssertionError("exactly four vehicle and pedestrian lights are required")
    encoded_vehicles = [VEHICLE_NAMES.index(value) for value in vehicles]
    encoded_pedestrians = [PEDESTRIAN_NAMES.index(value) for value in pedestrians]
    for index in range(4):
        following = (index + 1) % 4
        state = (
            encoded_vehicles[index],
            encoded_pedestrians[index],
            encoded_vehicles[following],
            encoded_pedestrians[following],
        )
        if state not in ALLOWED:
            raise AssertionError("adjacent traffic lights have an inconsistent state")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    built = build_model()
    solution = solve_from_args(built.model, args)
    print(f"prob016 status={solution.status}")
    if not solution.is_sat():
        return 1
    vehicles, pedestrians = decode(built, solution)
    validate(vehicles, pedestrians)
    print(f"vehicles={vehicles} pedestrians={pedestrians}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
