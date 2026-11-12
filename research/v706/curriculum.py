"""Its own questions (v706): every question its ways can be asked, of every
project it may look at, with the answer worked out from the project itself --
so that what it tests itself on is not the few cases someone wrote.

    python -m research.v706.curriculum          how many, how many it gets
    python -m research.v706.curriculum --show   each one it gets wrong

A way was learnt from one message (`list the classes in hires.py`). Each `?`
of it was filled by a word of that message; put another value of the same
field in that word's place, said as that word said its own (`classes` of
`class`: `functions` of `function`; `hires.py` of `rental/hires.py`:
`tools.py`), and it is another question -- whose right answer is the way's
query run on the project, which needs no one to write it. Asked of itself
(`ways.choose`, every way competing), the ones it gets wrong are cases to
learn from (`learn --curriculum`): a way taking a question another should, a
value said two ways.

The projects are the ones the improving may look at -- never HELD -- and its
own code (`own`: this package), which it models as it does any other.
"""
from __future__ import annotations

import itertools
import re

from research.v706 import queries as Q
from research.v706 import ways

#: the projects it asks itself about
PROJECTS = ("rental", "library", "garden", "own")
#: how many questions one way is asked of one project
ASKED = 40


def _held(fixture: str):
    from research.v706 import learn
    if fixture == "own":
        if "own" not in learn._HELD:
            learn._HELD["own"] = ways.project_of(
                learn.ROOT / "research" / "v706")
        return learn._HELD["own"]
    return learn.held_of(fixture)


def _holes(way_steps: list, bound_steps: list, source: str) -> list:
    """(collection, field, the value it was filled with) of each `?`."""
    out = []
    for step, bound in zip(way_steps, bound_steps):
        if step[0] == "is" and step[2] == "?":
            out.append((source, step[1], bound[2]))
        elif step[0] in ("linked", "unlinked"):
            out.extend(_holes(step[4], bound[4], step[1]))
    return out


def _form(value: str, word: str, was: str) -> str:
    """A value said as `word` said `was`."""
    if "/" in was and word == was.lower().rpartition("/")[2]:
        return value.rpartition("/")[2]
    if word == was.lower():
        return value
    return value + ("es" if value.endswith(("s", "x", "ch", "sh")) else "s")


def questions(found: dict | None = None, projects=PROJECTS) -> list:
    """[{turns, fixture, way, truth, give}]: its ways asked with other
    values, of each project, and what the project says is right."""
    found = found or ways.load()
    out = []
    for at, way in enumerate(found["ways"]):
        said = way.get("learnt from")
        if not said or way["from"] == "last" or not way.get("fixture"):
            continue
        home = Q.world(_held(way["fixture"]))
        bound = Q.bind(way, home, said, found["library"])
        if bound is None:
            continue
        steps = Q.expanded(way["steps"], found["library"])
        holes = _holes(steps, bound["steps"], bound["from"])
        words = [Q.said_as(said, value) for _, _, value in holes]
        if not holes or None in words:
            continue
        for fixture in projects:
            world_ = Q.world(_held(fixture))
            choices = []
            for collection, field, _ in holes:
                rows = world_.get(collection) or []
                choices.append(sorted({row.get(field) for row in rows
                                       if isinstance(row.get(field), str)
                                       and row.get(field)}))
            count = 0
            for values in itertools.product(*choices):
                if count >= ASKED:
                    break
                text = said
                for word, value, (_, _, was) in zip(words, values, holes):
                    text = re.sub(rf"(?<![\w./]){re.escape(word)}(?![\w/])",
                                  lambda _: _form(value, word, was), text,
                                  flags=re.I)
                again = Q.bind(way, world_, text, found["library"])
                if text == said and fixture == way["fixture"] or again is None:
                    continue
                meant = {"from": again["from"], "give": again["give"],
                         "steps": _filled(steps, list(values))}
                rows, truth = Q.run(meant, world_)
                if Q.empty(meant, truth):
                    continue
                count += 1
                out.append({"turns": [text], "fixture": fixture, "way": at,
                            "truth": truth, "give": again["give"][0]})
    return out


def _filled(steps: list, values: list) -> list:
    out = []
    for step in steps:
        if step[0] == "is" and step[2] == "?":
            out.append(["is", step[1], values.pop(0)])
        elif step[0] in ("linked", "unlinked"):
            out.append(step[:4] + [_filled(step[4], values)])
        else:
            out.append(step)
    return out


def right(one: dict, said: str | None) -> bool:
    """Whether an answer says what the project says is right."""
    if said is None:
        return False
    truth = one["truth"]
    if one["give"] == "any":
        return said.lower().startswith("yes" if truth else "no")
    if one["give"] == "count":
        return str(truth) in re.findall(r"\d+", said)
    return all(re.search(rf"(?<![\w.]){re.escape(str(value))}(?![\w])", said)
               for value in truth)


def asked(found: dict | None = None, projects=PROJECTS) -> list:
    """Each of its questions asked of its ways, every way competing:
    [(question, what it said or None)]."""
    found = found or ways.load()
    out = []
    for one in questions(found, projects):
        world_ = Q.world(_held(one["fixture"]))
        chosen = ways.choose(one["turns"][0], world_, found)
        said = None
        if chosen is not None:
            rows, value = Q.run(chosen[1], world_)
            if not Q.empty(chosen[1], value):
                said = Q.said(chosen[1], rows, value)
        out.append((one, said))
    return out


def failures(found: dict | None = None) -> list:
    """The questions of its own it gets wrong, as cases to learn from --
    of the projects that are fixtures (a case is said in a copy of one)."""
    from research.v704.suite import Conversation
    out = []
    for one, said in asked(found, [name for name in PROJECTS
                                   if name != "own"]):
        if right(one, said):
            continue
        check = (lambda turn, root, one=one: right(
            one, (turn.get("reply") or {}).get("text")))
        check.about = f"says {one['truth']!r}"
        out.append(Conversation("curriculum", one["turns"], check, True,
                                one["fixture"]))
    return out


def suite(projects=("rental", "library", "garden")) -> list:
    """Its questions as conversations, for the whole page to be asked them
    (`converse`): the reader, the dispatch and the ways together."""
    from research.v704.suite import Conversation
    out = []
    for one in questions(None, projects):
        check = (lambda turn, root, one=one: right(
            one, (turn.get("reply") or {}).get("text")
            if isinstance(turn.get("reply"), dict) else str(turn.get("reply"))))
        check.about = f"says {one['truth']!r}"
        out.append(Conversation("curriculum", one["turns"], check, True,
                                one["fixture"]))
    return out


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    options = parser.parse_args(argv)
    found = ways.load()
    got = asked(found)
    by: dict = {}
    for one, said in got:
        row = by.setdefault((one["fixture"], one["way"]), [0, 0])
        row[0] += 1
        row[1] += right(one, said)
        if options.show and not right(one, said):
            print(f"  WRONG {one['fixture']}: {one['turns'][0]} -> "
                  f"{said} (right: {one['truth']})")
    for (fixture, way), (count, good) in sorted(by.items()):
        print(f"{fixture:8} way {way}: {good} of {count}")
    print(f"all: {sum(right(one, said) for one, said in got)} of {len(got)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
