"""Enumerate the FULL infeasibility landscape with MARCO: model.enumerate_mus().

model.mus() returns ONE reason a model is infeasible. When there are several
independent conflicts it only shows the first - fix it and the model is still
infeasible for another reason. model.enumerate_mus() returns them ALL:

  - every MUS  (minimal unsatisfiable subset): each independent conflict;
  - every MSS  (maximal satisfiable subset): the largest sets of requirements
    you CAN keep together;
  - every MCS  (minimal correction set = the complement of an MSS): the smallest
    sets of requirements to drop to make the model solvable.

Scenario: scheduling a meeting from conflicting requests. Alice wants it early
(Mon/Tue) AND late (Thu/Fri); Bob wants it in the morning AND in the afternoon.
Two independent contradictions - so two MUSes, and four ways to keep a maximal
consistent set of requests.

Trace the solver with QAYD_VERBOSE=1.
"""

import os

import qayd as cp

MORNING, AFTERNOON = 0, 1

model = cp.Model()
day = model.int_var(1, 5, name="day")     # Mon..Fri
time = model.int_var(MORNING, AFTERNOON, name="time")

with model.soft("alice_early"):
    model.sum([day], "<=", 2)             # Mon or Tue
with model.soft("alice_late"):
    model.sum([day], ">=", 4)             # Thu or Fri
with model.soft("bob_morning"):
    model.sum([time], "<=", MORNING)
with model.soft("bob_afternoon"):
    model.sum([time], ">=", AFTERNOON)

time_limit = int(os.environ.get("QAYD_MUS_T", "8"))
requests = {"alice_early", "alice_late", "bob_morning", "bob_afternoon"}

# One reason (single-MUS) versus all of them (MARCO enumeration).
one = model.mus(time_limit=time_limit)
print(f"model.mus() finds ONE conflict:  {sorted(one)}")

muses, msses, complete = model.enumerate_mus(time_limit=time_limit)
if not complete:
    raise SystemExit("enumeration did not complete within the time limit")

muses = sorted(sorted(m) for m in muses)
msses = sorted(sorted(m) for m in msses)
mcses = sorted(sorted(requests - set(m)) for m in msses)

print(f"\nALL conflicts (MUSes), {len(muses)}:")
for m in muses:
    print(f"  {m}")
print(f"\nMaximal request sets you can keep (MSSes), {len(msses)}:")
for m in msses:
    print(f"  {m}")
print(f"\nMinimal sets to drop to make it schedulable (MCSes), {len(mcses)}:")
for m in mcses:
    print(f"  {m}")

# The two independent contradictions, both surfaced.
assert muses == [["alice_early", "alice_late"], ["bob_afternoon", "bob_morning"]], muses
# Four maximal consistent request sets: one choice from each conflict.
assert len(msses) == 4 and all(len(m) == 2 for m in msses), msses
# Each minimal fix relaxes exactly one request from each of the two conflicts.
assert all(len(c) == 2 for c in mcses), mcses
print("\nverified: two independent conflicts, four maximal keepable sets, four minimal fixes")
