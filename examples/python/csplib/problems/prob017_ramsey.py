"""CSPLib prob017: Ramsey edge-colouring problems.

Specification: https://www.csplib.org/Problems/prob017/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

Edge = tuple[int, int]


@dataclass(frozen=True)
class RamseyModel:
    model: cp.Model
    vertices: int
    edge_colours: dict[Edge, cp.IntVar]
    colours: int
    forbidden_cliques: tuple[int, int] | None


def _edge_variables(
    model: cp.Model, vertices: int, colours: int
) -> dict[Edge, cp.IntVar]:
    return {
        edge: model.int_var(0, colours - 1, name=f"edge_{edge[0]}_{edge[1]}")
        for edge in combinations(range(vertices), 2)
    }


def _inside(
    edge_colours: dict[Edge, cp.IntVar], vertices: tuple[int, ...]
) -> list[cp.IntVar]:
    return [edge_colours[edge] for edge in combinations(vertices, 2)]


def build_two_colour_model(
    vertices: int, *, red_clique: int = 3, blue_clique: int = 3
) -> RamseyModel:
    """Find a colouring with no red K_red and no blue K_blue."""

    if vertices < 1:
        raise ValueError("vertices must be positive")
    if red_clique < 2 or blue_clique < 2:
        raise ValueError("clique sizes must be at least two")

    model = cp.Model()
    edge_colours = _edge_variables(model, vertices, 2)
    for subset in combinations(range(vertices), red_clique):
        edges = _inside(edge_colours, subset)
        model.add(sum(edges) <= len(edges) - 1)
    for subset in combinations(range(vertices), blue_clique):
        model.add(sum(_inside(edge_colours, subset)) >= 1)

    if red_clique == blue_clique and edge_colours:
        model.add(next(iter(edge_colours.values())) == 0)
    return RamseyModel(model, vertices, edge_colours, 2, (red_clique, blue_clique))


def build_triangle_model(vertices: int, *, colours: int = 3) -> RamseyModel:
    """Find a k-colouring with no monochromatic triangle."""

    if vertices < 1:
        raise ValueError("vertices must be positive")
    if colours < 1:
        raise ValueError("colours must be positive")

    model = cp.Model()
    edge_colours = _edge_variables(model, vertices, colours)
    forbidden = [(colour, colour, colour) for colour in range(colours)]
    for triangle in combinations(range(vertices), 3):
        model.table(_inside(edge_colours, triangle), forbidden, positive=False)
    if edge_colours:
        model.add(next(iter(edge_colours.values())) == 0)
    return RamseyModel(model, vertices, edge_colours, colours, None)


def build_model(
    vertices: int = 5,
    *,
    variant: str = "ramsey",
    red_clique: int = 3,
    blue_clique: int = 3,
    colours: int = 3,
) -> RamseyModel:
    """Build either the two-colour Ramsey or multicolour triangle variant."""

    if variant == "ramsey":
        return build_two_colour_model(
            vertices,
            red_clique=red_clique,
            blue_clique=blue_clique,
        )
    if variant == "triangles":
        return build_triangle_model(vertices, colours=colours)
    raise ValueError("variant must be 'ramsey' or 'triangles'")


def decode(built: RamseyModel, solution: cp.Solution) -> dict[Edge, int]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return {
        edge: solution.value(variable) for edge, variable in built.edge_colours.items()
    }


def validate(built: RamseyModel, colouring: dict[Edge, int]) -> None:
    expected_edges = set(combinations(range(built.vertices), 2))
    if set(colouring) != expected_edges:
        raise AssertionError("the colouring must assign every edge exactly once")
    if any(colour < 0 or colour >= built.colours for colour in colouring.values()):
        raise AssertionError("an edge has an invalid colour")

    if built.forbidden_cliques is not None:
        red_clique, blue_clique = built.forbidden_cliques
        for subset in combinations(range(built.vertices), red_clique):
            if all(colouring[edge] == 1 for edge in combinations(subset, 2)):
                raise AssertionError("a forbidden red clique exists")
        for subset in combinations(range(built.vertices), blue_clique):
            if all(colouring[edge] == 0 for edge in combinations(subset, 2)):
                raise AssertionError("a forbidden blue clique exists")
        return

    for triangle in combinations(range(built.vertices), 3):
        colours = {colouring[edge] for edge in combinations(triangle, 2)}
        if len(colours) == 1:
            raise AssertionError("a monochromatic triangle exists")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("ramsey", "triangles"), default="ramsey")
    parser.add_argument("--vertices", type=int, default=5)
    parser.add_argument("--red-clique", type=int, default=3)
    parser.add_argument("--blue-clique", type=int, default=3)
    parser.add_argument("--colours", type=int, default=3)
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    if args.variant == "ramsey":
        built = build_two_colour_model(
            args.vertices,
            red_clique=args.red_clique,
            blue_clique=args.blue_clique,
        )
    else:
        built = build_triangle_model(args.vertices, colours=args.colours)
    solution = solve_from_args(built.model, args)
    print(
        f"prob017 variant={args.variant} vertices={args.vertices} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    colouring = decode(built, solution)
    validate(built, colouring)
    by_colour = {
        colour: [edge for edge, assigned in colouring.items() if assigned == colour]
        for colour in range(built.colours)
    }
    print(f"edges_by_colour={by_colour}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
