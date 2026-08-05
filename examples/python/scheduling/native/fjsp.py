"""Flexible job-shop scheduling: like the job shop, but each operation may run on
one of several eligible machines, with a machine-dependent processing time. The
search decides both the machine (the op's mode) and the start, minimising the
makespan.

``model.alternatives`` declares one moded interval per op (its eligible
``(machine, duration)`` pairs); ``model.no_overlap_by_machine`` keeps ops that land
on the same machine from overlapping; ``model.precedence`` posts the job-order
chain. ``solution.machines`` reports the chosen machine per op.

Use ``--jobs``, ``--machines`` and ``--time-limit`` to control the example.
"""

import argparse
from random import Random

import qayd as cp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--machines", type=int, default=3)
parser.add_argument("--time-limit", type=int, default=10)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
jobs = args.jobs
machines = args.machines
ops_per_job = machines
time_limit = args.time_limit

rng = Random(0)
# Each op is eligible on a random non-empty subset of machines, each with its
# own duration. modes[op] = [(machine, duration), ...].
modes = []
for _ in range(jobs):
    for _ in range(ops_per_job):
        elig = rng.sample(range(machines), rng.randint(1, machines))
        modes.append([(mc, rng.randint(2, 9)) for mc in elig])

horizon = sum(max(d for _, d in op) for op in modes)
proc = [{mc: d for mc, d in op} for op in modes]  # per-op machine -> duration

model = cp.Model()
ivs = model.alternatives(modes, horizon)
for j in range(jobs):
    for k in range(1, ops_per_job):
        model.precedence(ivs[j * ops_per_job + k - 1], ivs[j * ops_per_job + k])
model.no_overlap_by_machine()
model.minimize_makespan(ivs)

solution = model.solve(time_limit=time_limit, verbose=args.verbose)

n = jobs * ops_per_job
print(f"jobs: {jobs}  machines: {machines}  ops: {n}  status: {solution.status}")
if not solution.starts:
    raise SystemExit(f"status: {solution.status} - no schedule within {time_limit}s")
starts, chosen = solution.starts, solution.machines
print(f"makespan: {solution.objective}")
# The chosen machine must be eligible; duration follows from it.
for op in range(n):
    assert chosen[op] in proc[op], f"op {op} on an eligible machine"
end = [starts[op] + proc[op][chosen[op]] for op in range(n)]
for j in range(jobs):
    for k in range(1, ops_per_job):
        assert end[j * ops_per_job + k - 1] <= starts[j * ops_per_job + k], "job order respected"
for mc in range(machines):
    ops = sorted((op for op in range(n) if chosen[op] == mc), key=lambda o: starts[o])
    for a, b in zip(ops, ops[1:]):
        assert end[a] <= starts[b], f"machine {mc} no overlap"
assert max(end) == solution.objective, "reported makespan matches the schedule"
