"""CSPLib prob027: Alien Tiles.

Specification: https://www.csplib.org/Problems/prob027/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class AlienTilesModel:
    model: cp.Model
    goal: Grid
    colours: int
    clicks: list[list[cp.IntVar]]


@dataclass(frozen=True)
class HardestGoal:
    goal: Grid
    minimum_clicks: int
    clicks: list[list[int]]


def _validate_goal(goal: Grid, colours: int) -> None:
    size = len(goal)
    if size < 1 or any(len(row) != size for row in goal):
        raise ValueError("the goal must be a non-empty square grid")
    if colours < 2 or any(
        value < 0 or value >= colours for row in goal for value in row
    ):
        raise ValueError("goal values must lie in 0..colours-1")


def build_model(goal: Grid, *, colours: int) -> AlienTilesModel:
    _validate_goal(goal, colours)
    size = len(goal)
    model = cp.Model()
    clicks = [
        model.int_vars(size, 0, colours - 1, name=f"click_row_{row}")
        for row in range(size)
    ]
    for row in range(size):
        for column in range(size):
            effect = (
                sum(clicks[row])
                + sum(clicks[other_row][column] for other_row in range(size))
                - clicks[row][column]
            )
            model.add(effect % colours == goal[row][column])
    model.minimize(sum(click for row in clicks for click in row))
    return AlienTilesModel(model, goal, colours, clicks)


def decode(built: AlienTilesModel, solution: cp.Solution) -> list[list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [[solution.value(click) for click in row] for row in built.clicks]


def resulting_grid(clicks: list[list[int]], *, colours: int) -> Grid:
    size = len(clicks)
    return tuple(
        tuple(
            (
                sum(clicks[row])
                + sum(clicks[other_row][column] for other_row in range(size))
                - clicks[row][column]
            )
            % colours
            for column in range(size)
        )
        for row in range(size)
    )


def validate(
    built: AlienTilesModel, clicks: list[list[int]], objective: int | None
) -> None:
    if resulting_grid(clicks, colours=built.colours) != built.goal:
        raise AssertionError("the clicks do not produce the requested goal")
    if sum(click for row in clicks for click in row) != objective:
        raise AssertionError("the objective does not match the number of clicks")


def find_hardest_goal(size: int, colours: int, *, time_limit: int = 10) -> HardestGoal:
    if size < 1 or colours < 2:
        raise ValueError("size must be positive and colours must be at least two")
    state_count = colours ** (size * size)
    if state_count > 100_000:
        raise ValueError("hardest-goal enumeration is limited to 100,000 states")
    best: HardestGoal | None = None
    for flat_goal in product(range(colours), repeat=size * size):
        goal = tuple(
            tuple(flat_goal[row * size : (row + 1) * size]) for row in range(size)
        )
        built = build_model(goal, colours=colours)
        solution = built.model.solve(
            engine="exact", time_limit=time_limit, seed=0, threads=1
        )
        if not solution.is_sat():
            continue
        if solution.status != "OPTIMAL" or solution.objective is None:
            raise RuntimeError(
                "hardest-goal enumeration requires every reachable goal to be solved optimally"
            )
        clicks = decode(built, solution)
        validate(built, clicks, solution.objective)
        if best is None or solution.objective > best.minimum_clicks:
            best = HardestGoal(goal, solution.objective, clicks)
    if best is None:
        raise RuntimeError("no reachable goal found")
    return best


def parse_grid(text: str) -> Grid:
    rows = tuple(
        tuple(int(value) for value in row.replace(",", " ").split())
        for row in text.split(";")
    )
    if not rows or any(not row for row in rows):
        raise ValueError("grid text is empty")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", default="1,1;1,0")
    parser.add_argument("--colours", type=int, default=3)
    parser.add_argument("--hardest", action="store_true")
    parser.add_argument("--size", type=int, default=2, help="board size for --hardest")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    if args.hardest:
        hardest = find_hardest_goal(args.size, args.colours, time_limit=args.time_limit)
        print(
            f"prob027 hardest size={args.size} colours={args.colours} "
            f"minimum_clicks={hardest.minimum_clicks}"
        )
        print(f"goal={hardest.goal}")
        print(f"clicks={hardest.clicks}")
        return 0

    built = build_model(parse_grid(args.goal), colours=args.colours)
    solution = solve_from_args(built.model, args)
    print(
        f"prob027 size={len(built.goal)} colours={args.colours} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    clicks = decode(built, solution)
    validate(built, clicks, solution.objective)
    print(f"minimum_clicks={solution.objective} clicks={clicks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
