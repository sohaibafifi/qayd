"""Use ordered semantic search phases without weakening exact proofs."""

import qayd as cp


def build_model():
    model = cp.Model()
    colors = model.int_vars(3, 0, 2, name="color")
    model.all_different(colors)
    model.minimize(sum(colors))
    search_policy = cp.SearchPolicy(
        [
            cp.SearchPhase(colors[1:], "first-fail", "min"),
            cp.SearchPhase([colors[0]], "input-order", "max"),
        ]
    )
    return model, colors, search_policy


def decode(solution, colors):
    return [solution.value(color) for color in colors]


def validate(values, objective):
    return sorted(values) == [0, 1, 2] and objective == sum(values) == 3


def main():
    model, colors, search_policy = build_model()
    solution = model.solve(search_policy=search_policy, seed=7)
    values = decode(solution, colors)
    if solution.status != "OPTIMAL" or not validate(values, solution.objective):
        raise RuntimeError("invalid search-policy solution")
    print(f"status: {solution.status}")
    print(f"colors: {values}")
    print(f"objective: {solution.objective}")


if __name__ == "__main__":
    main()
