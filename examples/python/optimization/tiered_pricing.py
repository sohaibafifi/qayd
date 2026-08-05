"""Select two contracts with fixed-point and extensible collection scoring.

All monetary values use deterministic fixed-point integers. The cost combines
scaled multiplication and division, a piecewise volume fee, and a registered
external risk adjustment. A second objective maximizes quality among contracts
with minimum cost.
"""

import qayd as cp


SCALE = 1_000
contracts = list(range(1, 6))


def raw(value: float) -> int:
    return cp.fixed(value, scale=SCALE)


def round_ratio(numerator: int, denominator: int) -> int:
    quotient = abs(numerator) // abs(denominator)
    remainder = abs(numerator) % abs(denominator)
    if 2 * remainder >= abs(denominator):
        quotient += 1
    return quotient if numerator * denominator >= 0 else -quotient


def risk_adjustment(risk: int) -> int:
    """Quadratic risk premium, returned in the same fixed-point scale."""

    return round_ratio(risk * risk, SCALE)


external_name = "tiered_pricing_risk_adjustment_v1"
cp.register_external(external_name, risk_adjustment)

# Values are raw fixed-point integers. Index 0 is unused.
volume = cp.array([0, raw(0.8), raw(1.0), raw(1.2), raw(1.4), raw(1.6)])
rate = cp.array([0, raw(8.0), raw(7.5), raw(6.0), raw(5.5), raw(5.0)])
years = cp.array([0, raw(2.0), raw(2.5), raw(2.0), raw(3.5), raw(4.0)])
risk = cp.array([0, raw(0.4), raw(0.8), raw(0.3), raw(1.0), raw(1.4)])
quality = cp.array([0, 70, 95, 80, 90, 85])
fee_points = [
    (raw(0.0), raw(0.0)),
    (raw(1.0), raw(0.2)),
    (raw(1.5), raw(0.7)),
    (raw(2.0), raw(1.5)),
]


def contract_cost(contract):
    total = cp.mul_scaled(volume[contract], rate[contract], SCALE)
    annualized = cp.div_scaled(total, years[contract], SCALE)
    volume_fee = cp.piecewise(volume[contract], fee_points)
    premium = cp.external(external_name, risk[contract])
    return annualized + volume_fee + premium


model = cp.Model()
selected, _rejected = model.set_vars(contracts, count=2)
model.add(cp.count(selected) == 2)
model.minimize(cp.sum(selected, contract_cost))
model.then_maximize(cp.sum(selected, lambda contract: quality[contract]))

solution = model.solve(engine="exact", time_limit=5)

print(f"status: {solution.status}")
print(f"selected contracts: {solution.lists[0] if solution.lists else []}")
print(f"fixed-point objectives: {solution.objectives}")

assert solution.status == "OPTIMAL"
assert solution.lists is not None
assert len(solution.lists[0]) == 2
assert sorted([*solution.lists[0], *solution.lists[1]]) == contracts


def piecewise(value: int) -> int:
    if value <= fee_points[0][0]:
        return fee_points[0][1]
    for (x0, y0), (x1, y1) in zip(fee_points, fee_points[1:]):
        if value <= x1:
            return y0 + round_ratio((value - x0) * (y1 - y0), x1 - x0)
    return fee_points[-1][1]


raw_volume = [0, raw(0.8), raw(1.0), raw(1.2), raw(1.4), raw(1.6)]
raw_rate = [0, raw(8.0), raw(7.5), raw(6.0), raw(5.5), raw(5.0)]
raw_years = [0, raw(2.0), raw(2.5), raw(2.0), raw(3.5), raw(4.0)]
raw_risk = [0, raw(0.4), raw(0.8), raw(0.3), raw(1.0), raw(1.4)]
raw_quality = [0, 70, 95, 80, 90, 85]


def replay_cost(contract: int) -> int:
    total = round_ratio(raw_volume[contract] * raw_rate[contract], SCALE)
    annualized = round_ratio(total * SCALE, raw_years[contract])
    return annualized + piecewise(raw_volume[contract]) + risk_adjustment(raw_risk[contract])


selected_contracts = solution.lists[0]
assert solution.objectives[0] == sum(replay_cost(contract) for contract in selected_contracts)
assert solution.objectives[1] == sum(raw_quality[contract] for contract in selected_contracts)
