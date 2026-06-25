"""Flexible job-shop scheduling: like the job shop, but each operation may run on
one of several eligible machines, with a machine-dependent processing time. The
search decides both the machine (the op's mode) and the start, minimising the
makespan.

``model.interval_modes`` declares one moded interval per op (its eligible
``(machine, duration)`` pairs); ``model.machine_no_overlap`` keeps ops that land
on the same machine from overlapping; ``model.precedes`` posts the job-order
chain. ``solution.machines`` reports the chosen machine per op.

Tune via ``QAYD_FJSP_J`` / ``QAYD_FJSP_M`` / ``QAYD_FJSP_T``.
"""

import os
from random import Random

import qayd as cp

jobs = int(os.environ.get("QAYD_FJSP_J", "4"))
machines = int(os.environ.get("QAYD_FJSP_M", "3"))
ops_per_job = machines
time_limit = int(os.environ.get("QAYD_FJSP_T", "10"))

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
ivs = model.interval_modes(modes, horizon)
for j in range(jobs):
    for k in range(1, ops_per_job):
        model.precedes(ivs[j * ops_per_job + k - 1], ivs[j * ops_per_job + k])
model.machine_no_overlap()

solution = model.solve(time_limit=time_limit, verbose=os.environ.get("QAYD_VERBOSE") == "1")

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
