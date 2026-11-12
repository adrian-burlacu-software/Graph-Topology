"""Ways found by search (v706): the queries a message could mean, enumerated
by type from what its words say of the project, each run and checked -- no
model writes one.

    candidates(world, text, library)    every way the message says whole
    solve(world, text, check, library)  the best that passes a check

Every condition of a candidate is one the message says (`queries.says`): a
value of a field, or a field's name. So a way kept answers the messages that
say as much, and no others; what a message says that no field holds (`the
dearest`) is beyond this search, and it finds nothing.
"""
from __future__ import annotations

import itertools

from research.v706 import queries as Q

#: how many conditions a way has at most, and how many ways are tried
CONDITIONS = 3
TRIED = 4000
#: how much of the smaller of two fields' values must be shared to link them
SHARED = 0.3


def _text_fields(rows: list) -> list:
    out = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, str) and value and key not in out:
                out.append(key)
    return out


def _options(rows: list, text: str) -> list:
    """The conditions on a collection that a message says."""
    out = [["is", field, "?"] for field in _text_fields(rows)
           if Q.said_value(text, rows, field) is not None]
    out += [["has", field] for field in rows[0] if Q.says(text, field)]
    return out


def _links(world: dict, source: str, text: str) -> list:
    rows = world[source]
    mine = {field: {row.get(field) for row in rows} - {None, ""}
            for field in _text_fields(rows)}
    out = []
    for other, others in world.items():
        if other == source or not others:
            continue
        said = _options(others, text)
        if not said:
            continue
        for theirs in _text_fields(others):
            values = {row.get(theirs) for row in others} - {None, ""}
            for field, own in mine.items():
                shared = len(own & values)
                if not shared or shared < SHARED * min(len(own), len(values)):
                    continue
                for step in said:
                    if step[1] == theirs and step[0] == "is":
                        continue
                    for kind in ("linked", "unlinked"):
                        out.append([kind, other, field, theirs, [step]])
    return out


def size(way: dict, library: dict | None = None) -> int:
    """How long a way is, in the vocabulary there is: a library's steps used
    whole count once."""
    keys = [repr(list(step)) for step in way["steps"]]
    count = len(keys)
    for one in (library or {}).values():
        part = [repr(list(step)) for step in one["steps"]]
        if all(step in keys for step in part):
            count -= len(part) - 1
    return count


def candidates(world: dict, text: str, library: dict | None = None) -> list:
    """Every way a message says whole, as (way, bound): most said first, then
    shortest."""
    out = []
    for source, rows in world.items():
        if not rows or len(out) >= TRIED:
            continue
        options = _options(rows, text)
        links = _links(world, source, text)
        fields = _text_fields(rows)
        named = [one for one in rows[0] if Q.says(text, one)]
        for count in range(0, CONDITIONS + 1):
            for chosen in itertools.combinations(options + links, count):
                if sum(1 for step in chosen if step[0] in
                       ("linked", "unlinked")) > 1:
                    continue
                if not chosen and source != "last":
                    continue
                used = {step[1] for step in chosen if step[0] in ("is", "has")}
                gives = [["list", field] for field in fields
                         if field not in used] + [["count"], ["any"]]
                if any(field not in used for field in named):
                    gives.insert(0, ["pick", "?"])
                for give in gives:
                    out.append({"from": "data" if Q.is_data(source)
                                else source,
                                "steps": [list(step) for step in chosen],
                                "give": give})
    library = library or {}
    scored = []
    for at, way in enumerate(out[:TRIED]):
        bound = Q.bind(way, world, text)
        if bound is not None:
            scored.append((-bound["said"], size(way, library), at, way, bound))
    scored.sort(key=lambda one: one[:3])
    return [(way, bound) for _, _, _, way, bound in scored]


def solve(world: dict, text: str, check, library: dict | None = None):
    """The first candidate whose answer passes `check(said)`: (way, said,
    how many were tried), else (None, None, tried)."""
    tried = 0
    for way, bound in candidates(world, text, library):
        tried += 1
        rows, value = Q.run(bound, world)
        if Q.empty(bound, value):
            continue
        answer = Q.said(bound, rows, value)
        try:
            if check(answer):
                return way, answer, tried
        except Exception:                           # noqa: BLE001
            continue
    return None, None, tried
