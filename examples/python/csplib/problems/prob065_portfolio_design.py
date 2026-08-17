"""CSPLib prob065: optimal financial portfolio design.

Specification: https://www.csplib.org/Problems/prob065/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations

import qayd as cp

from ..common import add_solver_arguments, solve_from_args


@dataclass(frozen=True)
class PortfolioModel:
    model: cp.Model
    portfolio_count: int
    asset_count: int
    assets_per_portfolio: int
    selected: list[list[cp.IntVar]]


def build_model(
    portfolio_count: int, asset_count: int, assets_per_portfolio: int
) -> PortfolioModel:
    if (
        portfolio_count < 2
        or asset_count < 1
        or not 1 <= assets_per_portfolio <= asset_count
    ):
        raise ValueError("portfolio dimensions are invalid")
    model = cp.Model()
    selected = [
        model.int_vars(asset_count, 0, 1, name=f"portfolio_{portfolio}")
        for portfolio in range(portfolio_count)
    ]
    for row in selected:
        model.add(sum(row) == assets_per_portfolio)
    maximum_overlap = model.int_var(0, assets_per_portfolio, name="maximum_overlap")
    product_table = [(left, right, left * right) for left in (0, 1) for right in (0, 1)]
    for first, second in combinations(range(portfolio_count), 2):
        products = []
        for asset in range(asset_count):
            both = model.bool_var(name=f"both_{first}_{second}_{asset}")
            model.table(
                [selected[first][asset], selected[second][asset], both], product_table
            )
            products.append(both)
        model.add(sum(products) <= maximum_overlap)
    model.minimize(maximum_overlap)
    return PortfolioModel(
        model,
        portfolio_count,
        asset_count,
        assets_per_portfolio,
        selected,
    )


def decode(built: PortfolioModel, solution: cp.Solution) -> list[set[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [
        {asset for asset, variable in enumerate(row) if solution.value(variable)}
        for row in built.selected
    ]


def validate(
    built: PortfolioModel, portfolios: list[set[int]], objective: int | None
) -> None:
    if len(portfolios) != built.portfolio_count:
        raise AssertionError("the number of portfolios is invalid")
    if any(len(portfolio) != built.assets_per_portfolio for portfolio in portfolios):
        raise AssertionError("a portfolio contains the wrong number of assets")
    maximum_overlap = max(
        len(portfolios[first].intersection(portfolios[second]))
        for first, second in combinations(range(len(portfolios)), 2)
    )
    if objective is not None and maximum_overlap != objective:
        raise AssertionError("the objective does not match maximum portfolio overlap")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolios", type=int, default=4)
    parser.add_argument("--assets", type=int, default=6)
    parser.add_argument("--selected", type=int, default=3)
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    built = build_model(args.portfolios, args.assets, args.selected)
    solution = solve_from_args(built.model, args)
    print(f"prob065 portfolios={args.portfolios} status={solution.status}")
    if not solution.is_sat():
        return 1
    portfolios = decode(built, solution)
    validate(built, portfolios, solution.objective)
    print(f"maximum_overlap={solution.objective} portfolios={portfolios}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
