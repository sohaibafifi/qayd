"""Diagnose an over-constrained model with a Minimal Unsatisfiable Subset (MUS).

Three talks must each be scheduled in one of two slots (morning / afternoon).
A handful of *soft* requirements are added, each in its own ``with model.soft(...)``
block so it can be named in a conflict. The requirements are jointly infeasible;
``model.mus()`` returns the minimal subset that is *already* impossible on its
own — the smallest set of requirements to relax to make the model solvable.

Here the three "different slots" requirements clash with the two-slot capacity
(pigeonhole: three talks cannot all differ across two slots). The "A in the
morning" preference is irrelevant to that conflict, so the MUS excludes it.

Use ``--time-limit`` to bound core extraction.
"""

import argparse

import qayd as cp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--time-limit", type=int, default=8)
args = parser.parse_args()

MORNING, AFTERNOON = 1, 2

model = cp.Model()
# Hard fact (the model's structure): each talk lands in one of the two slots.
a = model.int_var(MORNING, AFTERNOON, name="A")
b = model.int_var(MORNING, AFTERNOON, name="B")
c = model.int_var(MORNING, AFTERNOON, name="C")

# Soft, individually-named requirements. Any mix of them may be the culprit.
with model.soft("A≠B"):
    model.not_equal(a, b)
with model.soft("B≠C"):
    model.not_equal(b, c)
with model.soft("A≠C"):
    model.not_equal(a, c)
with model.soft("A in the morning"):  # a mere preference, not part of the clash
    model.sum([a], "==", MORNING)

core = model.mus(time_limit=args.time_limit)

if core is None:
    raise SystemExit("model is satisfiable — no MUS")

print(f"requirements: 4    conflicting core: {len(core)}")
print(f"MUS (relax any one of these to make it solvable): {core}")

# The three "different slots" rules are the conflict; the preference is not.
assert set(core) == {"A≠B", "B≠C", "A≠C"}, core
assert "A in the morning" not in core, "an irrelevant preference must be excluded"

# Confirm the core is genuinely minimal: dropping any single requirement from it
# makes the remaining requirements satisfiable. We rebuild the model each time,
# keeping only the requirements we want to test.
def build(active):
    m = cp.Model()
    x = {name: m.int_var(MORNING, AFTERNOON, name=name) for name in ("A", "B", "C")}
    pairs = {"A≠B": ("A", "B"), "B≠C": ("B", "C"), "A≠C": ("A", "C")}
    for name in active:
        u, v = pairs[name]
        with m.soft(name):
            m.not_equal(x[u], x[v])
    return m


assert build(core).mus() is not None, "the whole core must still be a conflict"
for dropped in core:
    kept = [name for name in core if name != dropped]
    assert build(kept).mus() is None, f"without {dropped!r} the rest must be satisfiable"

print("verified: the core is unsatisfiable and minimal")
