"""CSPLib prob046: meeting scheduling.

Specification: https://www.csplib.org/Problems/prob046/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "horizon": 8,
    "travel_times": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    "meetings": [
        {"duration": 1, "location": 0, "agents": [0, 1]},
        {"duration": 2, "location": 1, "agents": [1, 2]},
        {"duration": 1, "location": 2, "agents": [0, 2]},
    ],
}


@dataclass(frozen=True)
class Meeting:
    duration: int
    location: int
    agents: frozenset[int]
    available: tuple[int, ...] | None


@dataclass(frozen=True)
class MeetingInstance:
    horizon: int
    travel_times: tuple[tuple[int, ...], ...]
    meetings: tuple[Meeting, ...]


@dataclass(frozen=True)
class MeetingModel:
    model: cp.Model
    instance: MeetingInstance
    starts: list[cp.IntVar]


def parse_instance(data: str | bytes) -> MeetingInstance:
    raw = json.loads(data)
    try:
        meetings = tuple(
            Meeting(
                int(item["duration"]),
                int(item["location"]),
                frozenset(int(agent) for agent in item["agents"]),
                None
                if "available" not in item
                else tuple(int(value) for value in item["available"]),
            )
            for item in raw["meetings"]
        )
        return MeetingInstance(
            int(raw["horizon"]),
            tuple(tuple(int(value) for value in row) for row in raw["travel_times"]),
            meetings,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid meeting-scheduling JSON instance") from error


def load_instance(path: str | Path) -> MeetingInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: MeetingInstance) -> MeetingModel:
    location_count = len(instance.travel_times)
    if instance.horizon < 1 or not instance.meetings or location_count < 1:
        raise ValueError("horizon, meetings, and locations must be non-empty")
    if any(
        len(row) != location_count or any(value < 0 for value in row)
        for row in instance.travel_times
    ):
        raise ValueError("travel_times must be a square non-negative matrix")
    if any(
        meeting.duration < 1
        or meeting.duration > instance.horizon
        or meeting.location < 0
        or meeting.location >= location_count
        or not meeting.agents
        for meeting in instance.meetings
    ):
        raise ValueError("a meeting is invalid")
    model = cp.Model()
    starts = [
        model.int_var(0, instance.horizon - meeting.duration, name=f"start_{index}")
        for index, meeting in enumerate(instance.meetings)
    ]
    for index, meeting in enumerate(instance.meetings):
        if meeting.available is not None:
            allowed = [
                value
                for value in meeting.available
                if 0 <= value <= instance.horizon - meeting.duration
            ]
            if allowed:
                model.table(
                    [starts[index]], [(value,) for value in sorted(set(allowed))]
                )
            else:
                model.add(starts[index] != starts[index])
    for first, left in enumerate(instance.meetings):
        for second in range(first + 1, len(instance.meetings)):
            right = instance.meetings[second]
            if not left.agents.intersection(right.agents):
                continue
            model.add(
                (
                    starts[first]
                    + left.duration
                    + instance.travel_times[left.location][right.location]
                    <= starts[second]
                )
                | (
                    starts[second]
                    + right.duration
                    + instance.travel_times[right.location][left.location]
                    <= starts[first]
                )
            )
    return MeetingModel(model, instance, starts)


def decode(built: MeetingModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.starts)


def validate(built: MeetingModel, starts: list[int]) -> None:
    if len(starts) != len(built.instance.meetings):
        raise AssertionError("the number of meeting start times is invalid")
    for index, (start, meeting) in enumerate(zip(starts, built.instance.meetings)):
        if start < 0 or start + meeting.duration > built.instance.horizon:
            raise AssertionError("a meeting lies outside the calendar")
        if meeting.available is not None and start not in meeting.available:
            raise AssertionError("a meeting starts outside its private availability")
        for second in range(index + 1, len(starts)):
            other = built.instance.meetings[second]
            if not meeting.agents.intersection(other.agents):
                continue
            separated = (
                start
                + meeting.duration
                + built.instance.travel_times[meeting.location][other.location]
                <= starts[second]
                or starts[second]
                + other.duration
                + built.instance.travel_times[other.location][meeting.location]
                <= start
            )
            if not separated:
                raise AssertionError("an attendee cannot travel between two meetings")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON meeting-scheduling instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob046 meetings={len(instance.meetings)} status={solution.status}")
    if not solution.is_sat():
        return 1
    starts = decode(built, solution)
    validate(built, starts)
    print(f"starts={starts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
