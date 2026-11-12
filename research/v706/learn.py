"""Learning ways from its own failures (v706): each case it answered wrong is
searched for a way that answers it (`synth.solve`), and a way found is kept
only where it costs nothing it had right.

    python -m research.v706.learn m1-seen       from a measure's failures
    python -m research.v706.learn m1-seen --curriculum   and from its own cases

No model runs here and no code is written by one: the search enumerates, the
case's own check decides, and what is kept is a line of `ways.json`. Only the
cases the improving may look at are read (`suite.known`: never HELD).

**Kept where it loses nothing** (`costs`): every case the measure had right is
said to the ways again; a new way that takes one and answers it wrong is
dropped -- a way fitted to its case that hijacks another is no gain.
"""
from __future__ import annotations

import json
from pathlib import Path

from research.v706 import queries as Q
from research.v706 import synth, ways

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conversations-out"
FIXTURES = ROOT / "research" / "v704" / "fixtures"


def _cases(label: str) -> list:
    """(case, verdict) of a measure, for the cases that may be looked at and
    are asked of a project."""
    from research.v704.suite import known
    by_turns = {json.dumps(one.turns): one for one in known() if one.project}
    out = []
    path = OUT / f"{label}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        case = by_turns.get(json.dumps(row["turns"]))
        if case is not None and not getattr(case.check, "changes", False):
            out.append((case, row["verdict"]))
    return out


_HELD: dict = {}


def held_of(fixture: str):
    if fixture not in _HELD:
        _HELD[fixture] = ways.project_of(FIXTURES / fixture)
    return _HELD[fixture]


def world_of(case) -> dict:
    """The world a case's last turn is said in: its project, and what the
    turn before named of it."""
    held = held_of(case.fixture)
    world_ = Q.world(held)
    if len(case.turns) > 1:
        before = case.turns[-2]
        last = [row for row in world_["symbols"]
                if row["name"] in before.replace("?", " ").split()]
        if last:
            world_["last"] = last
    return world_


def _check(case):
    return lambda said: bool(case.check({"reply": {"text": said}}, None))


def costs(found: dict, right: list) -> list:
    """The cases it had right that the ways now take and answer wrong:
    [(case, the way's place)]."""
    out = []
    for case in right:
        world_ = world_of(case)
        chosen = ways.choose(case.turns[-1], world_, found)
        if chosen is None:
            continue
        rows, value = Q.run(chosen[1], world_)
        if Q.empty(chosen[1], value):
            continue
        if not _check(case)(Q.said(chosen[1], rows, value)):
            out.append((case, found["ways"].index(chosen[0])))
    return out


def learn(label: str, more: list = ()) -> dict:
    """Ways for the failures of a measure (and `more` cases of its own
    making), kept where they cost nothing: {learnt, dropped, unsolved}."""
    found = ways.load()
    cases = _cases(label)
    right = [case for case, verdict in cases if verdict == "right"]
    failed = [case for case, verdict in cases if verdict != "right"] + \
        list(more)
    report = {"learnt": [], "dropped": [], "unsolved": []}
    for case in failed:
        world_ = world_of(case)
        text = case.turns[-1]
        chosen = ways.choose(text, world_, found)
        if chosen is not None:
            rows, value = Q.run(chosen[1], world_)
            if not Q.empty(chosen[1], value) and \
                    _check(case)(Q.said(chosen[1], rows, value)):
                continue                    # a way it has answers it already
        way, said, tried = synth.solve(world_, text, _check(case),
                                       found["library"])
        if way is None:
            report["unsolved"].append({"case": text, "tried": tried})
            continue
        if any(one["steps"] == way["steps"] and one["give"] == way["give"]
               and one["from"] == way["from"] for one in found["ways"]) or \
                any(Q.expanded(one["steps"], found["library"]) == way["steps"]
                    and one["give"] == way["give"] and
                    one["from"] == way["from"] for one in found["ways"]):
            # it has this way, and another takes the message from it
            report["unsolved"].append({"case": text, "tried": tried,
                                       "has": "the way, outranked"})
            continue
        way = {**way, "learnt from": text, "fixture": case.fixture}
        found["ways"].append(way)
        lost = costs(found, right)
        added = 1
        # what it takes from a case it had right, it learns that case's own
        # way for (`does clean_code raise anything?` took `which exception
        # does hire raise?`, whose way -- shorter, saying as much -- then
        # takes it back): both kept where together they cost nothing
        for one, _ in lost:
            again, _, _ = synth.solve(world_of(one), one.turns[-1],
                                      _check(one), found["library"])
            if again is not None:
                found["ways"].append({**again, "learnt from": one.turns[-1],
                                      "fixture": one.fixture})
                added += 1
        if lost:
            lost = costs(found, right)
        if lost:
            del found["ways"][-added:]
            report["dropped"].append({"case": text, "way": way, "costs": [
                one.turns[-1] for one, _ in lost]})
            continue
        report["learnt"].append({"case": text, "tried": tried, "said": said,
                                 "way": ways.described(way, found["library"])})
    report["named"] = ways.abstract(found)
    ways.save(found)
    return report


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--curriculum", action="store_true")
    options = parser.parse_args(argv)
    more = []
    if options.curriculum:
        from research.v706 import curriculum
        more = [one for one in curriculum.failures()]
    print(json.dumps(learn(options.label, more), indent=1, ensure_ascii=False,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
