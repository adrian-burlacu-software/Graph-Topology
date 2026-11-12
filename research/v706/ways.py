"""The ways it has learnt to answer (v706): kept as data it can read, run,
compare and extend -- its model of this part of itself.

    python -m research.v706.ways show       each way, in words
    python -m research.v706.ways abstract   shared steps made a named part

`ways.json` is the architecture's own writing: `learn` adds a way where a
search found one that holds, and `abstract` gives steps several ways share a
name of their own, which later searches count as one step (`synth.size`). A
change to it is a diff to review, as any change it makes to itself.

At a turn (`answer`): the way that fits the message and says the most of it,
where it leaves nothing the message says of its collection unused -- `where
is due_date in loans.py?` names a function, and a way listing a file's
functions does not answer it.
"""
from __future__ import annotations

import json
from pathlib import Path

from research.v706 import queries as Q
from research.v706 import synth

PATH = Path(__file__).with_name("ways.json")
#: conversation -> (records, turn): what the last way answered with
LAST: dict = {}
#: how few ways must share steps for them to be given a name
SHARED_BY = 2


def load() -> dict:
    if not PATH.exists():
        return {"ways": [], "library": {}}
    return json.loads(PATH.read_text(encoding="utf-8"))


def save(found: dict) -> None:
    PATH.write_text(json.dumps(found, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def project_of(root) -> object:
    """A folder read as a project, with no server: its files as text."""
    from research.v698.project import Project
    root = Path(root)
    held = Project(root.name)
    for one in sorted(root.rglob("*")):
        if one.is_file() and "__pycache__" not in one.parts:
            try:
                held.files[one.relative_to(root).as_posix()] = \
                    one.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
    return held


def unused(bound: dict, world_: dict, text: str) -> list:
    """The fields of a way's collection the message says a value of that the
    way makes nothing of."""
    rows = world_.get(bound["from"]) or []
    made = {step[1] for step in bound["steps"] if step[0] in ("is", "has")}
    made |= {step[2] for step in bound["steps"]
             if step[0] in ("linked", "unlinked")}
    # (what a list gives is no use of a value said of it: a way listing
    # names does not answer for the one name a message says)
    if bound["give"][0] == "pick":
        made.add(bound["give"][1])
    # (a word the way has used says nothing more by being another field's
    # value too: `Tool`, a class's name and its methods' owner)
    return [field for field in synth._text_fields(rows)
            if field not in made and
            Q.said_value(text, rows, field) is not None and
            Q.said_as(text, Q.said_value(text, rows, field))
            not in bound.get("words", ())]


def choose(text: str, world_: dict, found: dict | None = None):
    """(way, bound) of the way that answers a message, else None."""
    found = found or load()
    best = None
    for at, way in enumerate(found["ways"]):
        bound = Q.bind(way, world_, text, found["library"])
        if bound is None or not bound["said"] or unused(bound, world_, text):
            continue
        rank = (-bound["said"], len(bound["steps"]), at)
        if best is None or rank < best[0]:
            best = (rank, way, bound)
    return best and best[1:]


def _last(held, key, turn: int) -> list:
    """The records the turn before was about: a way's, else its subject's."""
    kept = LAST.get(key)
    if kept and kept[1] == turn - 1:
        return kept[0]
    from research.v698 import asking
    focus = asking.FOCUS.get(key)
    if not focus or focus[1] != turn - 1:
        return []
    subject = focus[0]
    index = held.index()
    if subject.kind == "file":
        return [{"file": subject.file,
                 "lines": len(held.files.get(subject.file, "").splitlines())}]
    return [row for row in Q.world(held)["symbols"]
            if row["name"] == subject.name and
            (not subject.file or row["file"] == subject.file)] \
        if index.find(subject.name) else []


def answer(text: str, held, key=None, turn: int = 0) -> dict | None:
    """A message answered by a way it has learnt, as the page takes a reply;
    None where none fits or it finds nothing."""
    if held is None or not held.files:
        return None
    found = load()
    if not found["ways"]:
        return None
    world_ = Q.world(held, _last(held, key, turn))
    chosen = choose(text, world_, found)
    if chosen is None:
        return None
    way, bound = chosen
    rows, value = Q.run(bound, world_)
    if Q.empty(bound, value):
        return None
    from research.v698 import asking
    reply = asking._reply(
        "way", text, Q.said(bound, rows, value),
        {"way": {key_: way[key_] for key_ in ("from", "steps", "give")},
         "bound": bound, "learnt from": way.get("learnt from")},
        name=Q._named(rows[0]) if rows else "")
    LAST[key] = (rows, turn)
    return reply


def applies(text: str, held, key=None, turn: int = 0) -> bool:
    if held is None or not held.files:
        return False
    found = load()
    if not found["ways"]:
        return False
    world_ = Q.world(held, _last(held, key, turn))
    chosen = choose(text, world_, found)
    return chosen is not None and not Q.empty(
        chosen[1], Q.run(chosen[1], world_)[1])


def follows(text: str, held, key=None, turn: int = 0) -> bool:
    """Whether a message is answered by a way from what the turn before was
    about."""
    if held is None or not held.files or not _last(held, key, turn):
        return False
    found = load()
    world_ = Q.world(held, _last(held, key, turn))
    chosen = choose(text, world_, found)
    return chosen is not None and chosen[1]["from"] == "last"


def abstract(found: dict) -> list:
    """Steps that several ways share, given a name of their own and used by
    it: the vocabulary grows by what was found useful. The names made."""
    made = []
    while True:
        counts: dict = {}
        for at, way in enumerate(found["ways"]):
            steps = [step for step in way["steps"] if step[0] != "use"]
            seen = set()
            for size in range(2, len(steps) + 1):
                for start in range(0, len(steps) - size + 1):
                    part = steps[start:start + size]
                    key = json.dumps(part)
                    if key not in seen:
                        seen.add(key)
                        counts.setdefault(key, []).append(at)
        best = max(((key, where) for key, where in counts.items()
                    if len(where) >= SHARED_BY),
                   key=lambda one: (len(json.loads(one[0])) - 1) *
                   (len(one[1]) - 1), default=None)
        if best is None:
            return made
        part = json.loads(best[0])
        name = "_".join(step[1].replace("/", "_") for step in part)
        while name in found["library"]:
            name += "_"
        found["library"][name] = {"steps": part}
        for at in best[1]:
            way = found["ways"][at]
            keys = [json.dumps(step) for step in way["steps"]]
            start = next(i for i in range(len(keys)) if
                         keys[i:i + len(part)] ==
                         [json.dumps(step) for step in part])
            way["steps"] = way["steps"][:start] + [["use", name]] + \
                way["steps"][start + len(part):]
        made.append(name)


def described(way: dict, library: dict) -> str:
    steps = Q.expanded(way["steps"], library)
    how = Q._described([["is", step[1], "what is said"]
                        if step[0] == "is" and step[2] == "?" else step
                        for step in steps])
    give = way["give"]
    gives = "how many there are" if give[0] == "count" else \
        "whether there is one" if give[0] == "any" else \
        f"their {give[1]}" if give[1] != "?" else "the field the message names"
    parts = [step[1] for step in way["steps"] if step[0] == "use"]
    return (f"From {way['from']}" + (f", where {how}" if how else "")
            + f": {gives}." + (f" Uses {', '.join(parts)}." if parts else "")
            + (f" Learnt from “{way['learnt from']}”."
               if way.get("learnt from") else ""))


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("show", "abstract"))
    options = parser.parse_args(argv)
    found = load()
    if options.job == "abstract":
        print("named:", abstract(found))
        save(found)
    for name, one in found["library"].items():
        print(f"{name} = {Q._described(one['steps'])}")
    for way in found["ways"]:
        print("-", described(way, found["library"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
