"""CSPLib prob011: ACC basketball schedule.

Specification: https://www.csplib.org/Problems/prob011/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

ACC_TEAMS = ("Clem", "Duke", "FSU", "GT", "UMD", "UNC", "NCSt", "UVA", "Wake")
DEFAULT_INSTANCE = {
    "teams": list(ACC_TEAMS),
    "dates": 18,
    "mirror_pairs": [
        [0, 7],
        [1, 8],
        [2, 11],
        [3, 12],
        [4, 13],
        [5, 14],
        [6, 15],
        [9, 16],
        [10, 17],
    ],
    "acc_rules": True,
    "fixed_schedule": [
        [
            [1, 1],
            [5, 0],
            [6, 1],
            [3, 0],
            [8, 0],
            [4, 1],
            [7, 0],
            [1, 0],
            [5, 1],
            [2, 1],
            [9, 0],
            [6, 0],
            [3, 1],
            [8, 1],
            [4, 0],
            [7, 1],
            [2, 0],
            [9, 0],
        ],
        [
            [0, 0],
            [2, 1],
            [4, 1],
            [6, 0],
            [3, 1],
            [8, 0],
            [9, 0],
            [0, 1],
            [2, 0],
            [7, 1],
            [5, 0],
            [4, 0],
            [6, 1],
            [3, 0],
            [8, 1],
            [9, 0],
            [7, 0],
            [5, 1],
        ],
        [
            [5, 1],
            [1, 0],
            [9, 0],
            [8, 1],
            [4, 0],
            [7, 1],
            [6, 0],
            [5, 0],
            [1, 1],
            [0, 0],
            [3, 1],
            [9, 0],
            [8, 0],
            [4, 1],
            [7, 0],
            [6, 1],
            [0, 1],
            [3, 0],
        ],
        [
            [7, 1],
            [6, 0],
            [5, 0],
            [0, 1],
            [1, 0],
            [9, 0],
            [8, 1],
            [7, 0],
            [6, 1],
            [4, 0],
            [2, 0],
            [5, 1],
            [0, 0],
            [1, 1],
            [9, 0],
            [8, 0],
            [4, 1],
            [2, 1],
        ],
        [
            [6, 1],
            [8, 1],
            [1, 0],
            [9, 0],
            [2, 1],
            [0, 0],
            [5, 1],
            [6, 0],
            [8, 0],
            [3, 1],
            [7, 0],
            [1, 1],
            [9, 0],
            [2, 0],
            [0, 1],
            [5, 0],
            [3, 0],
            [7, 1],
        ],
        [
            [2, 0],
            [0, 1],
            [3, 1],
            [7, 0],
            [9, 0],
            [6, 1],
            [4, 0],
            [2, 1],
            [0, 0],
            [8, 0],
            [1, 1],
            [3, 0],
            [7, 1],
            [9, 0],
            [6, 0],
            [4, 1],
            [8, 1],
            [1, 0],
        ],
        [
            [4, 0],
            [3, 1],
            [0, 0],
            [1, 1],
            [7, 1],
            [5, 0],
            [2, 1],
            [4, 1],
            [3, 0],
            [9, 0],
            [8, 1],
            [0, 1],
            [1, 0],
            [7, 0],
            [5, 1],
            [2, 0],
            [9, 0],
            [8, 0],
        ],
        [
            [3, 0],
            [9, 0],
            [8, 0],
            [5, 1],
            [6, 0],
            [2, 0],
            [0, 1],
            [3, 1],
            [9, 0],
            [1, 0],
            [4, 1],
            [8, 1],
            [5, 0],
            [6, 1],
            [2, 1],
            [0, 0],
            [1, 1],
            [4, 0],
        ],
        [
            [9, 0],
            [4, 0],
            [7, 1],
            [2, 0],
            [0, 1],
            [1, 1],
            [3, 0],
            [9, 0],
            [4, 1],
            [5, 1],
            [6, 0],
            [7, 0],
            [2, 1],
            [0, 0],
            [1, 0],
            [3, 1],
            [5, 0],
            [6, 1],
        ],
    ],
}


@dataclass(frozen=True)
class AccInstance:
    teams: tuple[str, ...]
    dates: int
    mirror_pairs: tuple[tuple[int, int], ...]
    acc_rules: bool
    fixed_schedule: tuple[tuple[tuple[int, int], ...], ...] | None


@dataclass(frozen=True)
class AccModel:
    model: cp.Model
    instance: AccInstance
    opponents: list[list[cp.IntVar]]
    home: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> AccInstance:
    raw = json.loads(data)
    try:
        return AccInstance(
            tuple(str(value) for value in raw["teams"]),
            int(raw["dates"]),
            tuple((int(pair[0]), int(pair[1])) for pair in raw["mirror_pairs"]),
            bool(raw.get("acc_rules", False)),
            None
            if "fixed_schedule" not in raw
            else tuple(
                tuple((int(game[0]), int(game[1])) for game in row)
                for row in raw["fixed_schedule"]
            ),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("invalid ACC-schedule JSON instance") from error


def load_instance(path: str | Path) -> AccInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: AccInstance) -> AccModel:
    team_count = len(instance.teams)
    if team_count < 2 or len(set(instance.teams)) != team_count:
        raise ValueError("team names must be unique")
    expected_dates = 2 * (team_count - 1) if team_count % 2 == 0 else 2 * team_count
    if instance.dates != expected_dates:
        raise ValueError("the date count is invalid for a double round robin")
    if sorted(date for pair in instance.mirror_pairs for date in pair) != list(
        range(instance.dates)
    ):
        raise ValueError("mirror pairs must partition all dates")
    if instance.acc_rules and instance.teams != ACC_TEAMS:
        raise ValueError("ACC-specific constraints require the canonical nine teams")
    bye = team_count
    bye_count = instance.dates - 2 * (team_count - 1)
    model = cp.Model()
    opponents = [
        model.int_vars(instance.dates, 0, bye, name=f"opponent_{team}")
        for team in range(team_count)
    ]
    home = [
        model.int_vars(instance.dates, 0, 1, name=f"home_{team}")
        for team in range(team_count)
    ]
    home_game = [[None for _ in range(instance.dates)] for _ in range(team_count)]
    away_game = [[None for _ in range(instance.dates)] for _ in range(team_count)]
    for team in range(team_count):
        counts = [0 if opponent == team else 2 for opponent in range(team_count)] + [
            bye_count
        ]
        model.cardinality(
            opponents[team], list(range(team_count + 1)), counts, counts, closed=True
        )
        allowed_team = [
            (bye, 0),
            *[
                (opponent, host)
                for opponent in range(team_count)
                if opponent != team
                for host in range(2)
            ],
        ]
        for date in range(instance.dates):
            model.table([opponents[team][date], home[team][date]], allowed_team)
            home_flag = model.bool_var(name=f"home_game_{team}_{date}")
            away_flag = model.bool_var(name=f"away_game_{team}_{date}")
            model.table(
                [opponents[team][date], home[team][date], home_flag, away_flag],
                [
                    (
                        opponent,
                        host,
                        int(opponent != bye and host == 1),
                        int(opponent != bye and host == 0),
                    )
                    for opponent, host in allowed_team
                ],
            )
            home_game[team][date] = home_flag
            away_game[team][date] = away_flag
        for opponent in range(team_count):
            if opponent == team:
                continue
            flags = []
            for date in range(instance.dates):
                flag = model.bool_var(name=f"home_{team}_against_{opponent}_{date}")
                model.table(
                    [opponents[team][date], home[team][date], flag],
                    [
                        (candidate, host, int(candidate == opponent and host == 1))
                        for candidate, host in allowed_team
                    ],
                )
                flags.append(flag)
            model.add(sum(flags) == 1)
        for first, second in instance.mirror_pairs:
            mirror_rows = [(bye, 0, bye, 0)] + [
                (opponent, host, opponent, 1 - host)
                for opponent in range(team_count)
                if opponent != team
                for host in range(2)
            ]
            model.table(
                [
                    opponents[team][first],
                    home[team][first],
                    opponents[team][second],
                    home[team][second],
                ],
                mirror_rows,
            )
    for date in range(instance.dates):
        for first in range(team_count):
            left_domain = [
                (bye, 0),
                *[
                    (opponent, host)
                    for opponent in range(team_count)
                    if opponent != first
                    for host in range(2)
                ],
            ]
            for second in range(first + 1, team_count):
                right_domain = [
                    (bye, 0),
                    *[
                        (opponent, host)
                        for opponent in range(team_count)
                        if opponent != second
                        for host in range(2)
                    ],
                ]
                rows = []
                for left_opponent, left_home in left_domain:
                    for right_opponent, right_home in right_domain:
                        meet = left_opponent == second
                        reciprocal = right_opponent == first
                        if meet == reciprocal and (
                            not meet or left_home + right_home == 1
                        ):
                            rows.append(
                                (left_opponent, left_home, right_opponent, right_home)
                            )
                model.table(
                    [
                        opponents[first][date],
                        home[first][date],
                        opponents[second][date],
                        home[second][date],
                    ],
                    rows,
                )
    if instance.acc_rules:
        _add_acc_rules(model, instance, opponents, home_game, away_game, bye)
    if instance.fixed_schedule is not None:
        if len(instance.fixed_schedule) != team_count or any(
            len(row) != instance.dates for row in instance.fixed_schedule
        ):
            raise ValueError("fixed_schedule dimensions are invalid")
        for team, row in enumerate(instance.fixed_schedule):
            for date, (opponent, host) in enumerate(row):
                model.add(opponents[team][date] == opponent)
                model.add(home[team][date] == host)
    return AccModel(model, instance, opponents, home)


def _match_flags(
    model: cp.Model,
    opponents: list[list[cp.IntVar]],
    team: int,
    opponent_set: set[int],
    dates: range,
    name: str,
) -> list[cp.IntVar]:
    team_count = len(opponents)
    flags = []
    for date in dates:
        flag = model.bool_var(name=f"{name}_{date}")
        model.table(
            [opponents[team][date], flag],
            [
                (opponent, int(opponent in opponent_set))
                for opponent in range(team_count + 1)
            ],
        )
        flags.append(flag)
    return flags


def _add_acc_rules(
    model: cp.Model,
    instance: AccInstance,
    opponents: list[list[cp.IntVar]],
    home_game: list[list[cp.IntVar]],
    away_game: list[list[cp.IntVar]],
    bye: int,
) -> None:
    index = {name: team for team, name in enumerate(instance.teams)}
    for team in range(len(instance.teams)):
        model.add(away_game[team][-2] + away_game[team][-1] <= 1)
        for start in range(instance.dates - 2):
            model.add(sum(away_game[team][start : start + 3]) <= 2)
            model.add(sum(home_game[team][start : start + 3]) <= 2)
        for start in range(instance.dates - 3):
            model.add(sum(home_game[team][start : start + 4]) >= 1)
        for start in range(instance.dates - 4):
            model.add(sum(away_game[team][start : start + 5]) >= 1)
        weekends = range(1, instance.dates, 2)
        model.add(sum(home_game[team][date] for date in weekends) == 4)
        model.add(sum(away_game[team][date] for date in weekends) == 4)
        model.add(sum(away_game[team][date] for date in list(weekends)[:5]) <= 3)
        targets = {index["UNC"], index["Duke"]}
        away_targets = []
        for date in range(instance.dates):
            flag = model.bool_var(name=f"away_unc_duke_{team}_{date}")
            model.table(
                [opponents[team][date], away_game[team][date], flag],
                [
                    (opponent, away, int(opponent in targets and away == 1))
                    for opponent in range(len(instance.teams) + 1)
                    for away in range(2)
                ],
            )
            away_targets.append(flag)
        for date in range(instance.dates - 1):
            model.add(away_targets[date] + away_targets[date + 1] <= 1)
        trio = _match_flags(
            model,
            opponents,
            team,
            {index["UNC"], index["Duke"], index["Wake"]},
            range(instance.dates),
            f"trio_{team}",
        )
        for date in range(instance.dates - 2):
            model.add(sum(trio[date : date + 3]) <= 2)
    rivals = {
        "Duke": "UNC",
        "UNC": "Duke",
        "Clem": "GT",
        "GT": "Clem",
        "NCSt": "Wake",
        "Wake": "NCSt",
        "UMD": "UVA",
        "UVA": "UMD",
    }
    for team, rival in rivals.items():
        allowed = {index[rival], index["FSU"], bye}
        model.table(
            [opponents[index[team]][-1]], [(value,) for value in sorted(allowed)]
        )
    for left, right in [
        ("Wake", "UNC"),
        ("Wake", "Duke"),
        ("GT", "UNC"),
        ("GT", "Duke"),
    ]:
        flags = _match_flags(
            model,
            opponents,
            index[left],
            {index[right]},
            range(10, 18),
            f"late_{left}_{right}",
        )
        model.add(sum(flags) >= 1)
    model.add(opponents[index["UNC"]][17] == index["Duke"])
    model.add(opponents[index["UNC"]][10] == index["Duke"])
    model.add(opponents[index["UNC"]][1] == index["Clem"])
    model.add(opponents[index["Duke"]][15] == bye)
    model.add(home_game[index["Wake"]][16] == 0)
    model.add(opponents[index["Wake"]][0] == bye)
    for team in ("Clem", "Duke", "UMD", "Wake"):
        model.add(away_game[index[team]][17] == 0)
    for team in ("Clem", "FSU", "GT", "Wake"):
        model.add(away_game[index[team]][0] == 0)
    for team in ("FSU", "NCSt"):
        model.add(opponents[index[team]][17] != bye)
    model.add(opponents[index["UNC"]][0] != bye)


def decode(
    built: AccModel, solution: cp.Solution
) -> list[list[tuple[int | None, bool | None]]]:
    bye = len(built.instance.teams)
    return [
        [
            (None, None)
            if solution.value(built.opponents[team][date]) == bye
            else (
                solution.value(built.opponents[team][date]),
                bool(solution.value(built.home[team][date])),
            )
            for date in range(built.instance.dates)
        ]
        for team in range(len(built.instance.teams))
    ]


def validate(
    built: AccModel, schedule: list[list[tuple[int | None, bool | None]]]
) -> None:
    team_count = len(schedule)
    for team, row in enumerate(schedule):
        for opponent in range(team_count):
            if opponent == team:
                continue
            games = [
                (date, home)
                for date, (other, home) in enumerate(row)
                if other == opponent
            ]
            if len(games) != 2 or {home for _, home in games} != {False, True}:
                raise AssertionError("the schedule is not a double round robin")
        for date, (opponent, home) in enumerate(row):
            if opponent is not None and schedule[opponent][date] != (team, not home):
                raise AssertionError("a match is not reciprocal")
        for first, second in built.instance.mirror_pairs:
            if row[first][0] != row[second][0] or (
                row[first][0] is not None and row[first][1] == row[second][1]
            ):
                raise AssertionError("the mirroring scheme is violated")
    if built.instance.acc_rules:
        _validate_acc_rules(built.instance, schedule)


def _validate_acc_rules(
    instance: AccInstance,
    schedule: list[list[tuple[int | None, bool | None]]],
) -> None:
    index = {name: team for team, name in enumerate(instance.teams)}

    def opponent(team: int, date: int) -> int | None:
        return schedule[team][date][0]

    def is_home(team: int, date: int) -> bool:
        return schedule[team][date][0] is not None and schedule[team][date][1] is True

    def is_away(team: int, date: int) -> bool:
        return schedule[team][date][0] is not None and schedule[team][date][1] is False

    for team in range(len(instance.teams)):
        if is_away(team, 16) and is_away(team, 17):
            raise AssertionError("a team has two final away matches")
        if any(
            sum(is_away(team, date) for date in range(start, start + 3)) > 2
            for start in range(16)
        ):
            raise AssertionError("a team has three consecutive away matches")
        if any(
            sum(is_home(team, date) for date in range(start, start + 3)) > 2
            for start in range(16)
        ):
            raise AssertionError("a team has three consecutive home matches")
        if any(
            not any(is_home(team, date) for date in range(start, start + 4))
            for start in range(15)
        ):
            raise AssertionError("a team has four consecutive away matches or byes")
        if any(
            not any(is_away(team, date) for date in range(start, start + 5))
            for start in range(14)
        ):
            raise AssertionError("a team has five consecutive home matches or byes")
        weekends = list(range(1, 18, 2))
        if (
            sum(is_home(team, date) for date in weekends) != 4
            or sum(is_away(team, date) for date in weekends) != 4
        ):
            raise AssertionError("a team violates its weekend pattern")
        if sum(is_away(team, date) for date in weekends[:5]) > 3:
            raise AssertionError("a team violates the first-weekend pattern")
        targets = {index["UNC"], index["Duke"]}
        if any(
            is_away(team, date)
            and opponent(team, date) in targets
            and is_away(team, date + 1)
            and opponent(team, date + 1) in targets
            for date in range(17)
        ):
            raise AssertionError("a team has consecutive away games at UNC and Duke")
        trio = {index["UNC"], index["Duke"], index["Wake"]}
        if any(
            sum(opponent(team, date) in trio for date in range(start, start + 3)) == 3
            for start in range(16)
        ):
            raise AssertionError("a team plays UNC, Duke, and Wake consecutively")
    rivals = {
        "Duke": "UNC",
        "UNC": "Duke",
        "Clem": "GT",
        "GT": "Clem",
        "NCSt": "Wake",
        "Wake": "NCSt",
        "UMD": "UVA",
        "UVA": "UMD",
    }
    for team, rival in rivals.items():
        if opponent(index[team], 17) not in {index[rival], index["FSU"], None}:
            raise AssertionError("a final rival-match constraint is violated")
    for left, right in [
        ("Wake", "UNC"),
        ("Wake", "Duke"),
        ("GT", "UNC"),
        ("GT", "Duke"),
    ]:
        if not any(
            opponent(index[left], date) == index[right] for date in range(10, 18)
        ):
            raise AssertionError("a constrained pairing is missing from dates 11 to 18")
    fixed_opponents = {
        ("UNC", 17): "Duke",
        ("UNC", 10): "Duke",
        ("UNC", 1): "Clem",
    }
    if any(
        opponent(index[team], date) != index[other]
        for (team, date), other in fixed_opponents.items()
    ):
        raise AssertionError("a fixed ACC match is missing")
    if (
        opponent(index["Duke"], 15) is not None
        or opponent(index["Wake"], 0) is not None
    ):
        raise AssertionError("a fixed ACC bye is missing")
    if is_home(index["Wake"], 16):
        raise AssertionError("Wake plays at home on date 17")
    if any(is_away(index[team], 17) for team in ("Clem", "Duke", "UMD", "Wake")):
        raise AssertionError("a final-date home or bye restriction is violated")
    if any(is_away(index[team], 0) for team in ("Clem", "FSU", "GT", "Wake")):
        raise AssertionError("a first-date home or bye restriction is violated")
    if any(opponent(index[team], 17) is None for team in ("FSU", "NCSt")):
        raise AssertionError("FSU or NCSt has a bye on the final date")
    if opponent(index["UNC"], 0) is None:
        raise AssertionError("UNC has a bye on the first date")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON ACC-schedule instance")
    add_solver_arguments(parser, time_limit=60)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob011 teams={len(instance.teams)} status={solution.status}")
    if not solution.is_sat():
        return 1
    schedule = decode(built, solution)
    validate(built, schedule)
    for team, row in zip(instance.teams, schedule):
        print(team, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
