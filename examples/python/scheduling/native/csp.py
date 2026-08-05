"""Car sequencing: order cars on the assembly line so that, for each option, no
more than ``p`` of every ``q`` consecutive cars require it. The capacity rules
are soft: minimise the total overflow, summed with a sliding-window reduction.

The sequence is one permutation list over the cars. For each option, a
``cp.windows`` reduction sums the option flag over each window and penalises the
excess over the limit.

Use ``--cars`` and ``--time-limit`` to control the generated instance.
"""

import argparse
from random import Random

import qayd as cp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cars", type=int, default=30)
parser.add_argument("--time-limit", type=int, default=8)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
n = args.cars
time_limit = args.time_limit

rng = Random(0)
# Three options, each with a p/q capacity rule. options[o] is the flag per car.
rules = [(1, 2), (2, 3), (1, 3)]  # (p, q) per option
options = [[1 if rng.random() < 0.4 else 0 for _ in range(n)] for _ in rules]
# Paint colour per car; index n is a sentinel colour for the run-length scan.
colours = 3
colour = [rng.randint(0, colours - 1) for _ in range(n)] + [-1]
paint_limit = 5  # no more than this many same-colour cars in a row

model = cp.Model()
(seq,) = model.list_vars(list(range(n)), count=1)
penalty = 0
for (p, q), flag in zip(rules, options):
    F = cp.array(flag)
    penalty = penalty + cp.windows(seq, q, inner=lambda c, F=F: F[c], emit=lambda t, p=p: cp.max(0, t - p))
model.minimize(penalty)
# Paint batching: forbid a run of same-colour cars longer than the limit. A scan
# tracks the current run length; a step that exceeds the limit is a violation.
C = cp.array(colour)
model.add(
    cp.scan_sum(
        seq,
        step=lambda cur, run, prev: cp.if_(cp.eq(C[cur], C[prev]), run + 1, 1),
        emit=lambda cur, run, prev: cp.if_(run > paint_limit, 1, 0),
        init=0,
        boundary=n,
    )
    <= 0
)

solution = model.solve(time_limit=time_limit, verbose=args.verbose)

print(f"cars: {n}  options: {len(rules)}  status: {solution.status}")
if solution.lists is None:
    raise SystemExit(f"status: {solution.status} - no sequence within {time_limit}s")
order = solution.lists[0]
print(f"total option overflow: {solution.objective}")

# Replay the window penalties to confirm the reported objective.
total = 0
for (p, q), flag in zip(rules, options):
    for start in range(n - q + 1):
        cnt = sum(flag[order[start + idx]] for idx in range(q))
        total += max(0, cnt - p)
print(f"sequence: {order}")
assert total == solution.objective, "reported overflow matches the replayed windows"
assert sorted(order) == list(range(n)), "every car sequenced once"
# Paint batching is a hard constraint: no run of same colour exceeds the limit.
run = 1
for a, b in zip(order, order[1:]):
    run = run + 1 if colour[a] == colour[b] else 1
    assert run <= paint_limit, "paint run within the batch limit"
