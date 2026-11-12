"""Answers as programs (v706): a small typed query language over a model of
a project, in place of one hand-written function an answer.

An answer written in Python (`asking._functions`) is nothing the architecture
can search: sixty changes were written for one question and none held. An
answer written here is a few steps over collections of records, each step of a
handful of kinds -- a space small enough to enumerate (`synth`), check by
running, keep as data (`ways.json`), and compress into its own vocabulary
(`ways.abstract`).

**The world** (`world`): what a project is, as collections of flat records --
`symbols` (name, kind, file, line, lines, doc, value, owner), `uses` (name,
file, line, user, use, test), `raises` (name, file, line, raise), `files`
(file, lines), each collection of its
data files (`data/tools.csv:`), and `last`, the records the turn before was
about.

**A way** is {from, steps, give}:

    from    a collection; `data` is whichever data collection the steps fit
    steps   ["is", field, "?"]      the field is the value the message says
            ["is", field, value]    the field is this value
            ["has", field]          the field is set (and the message names it)
            ["linked", collection, mine, theirs, [steps]]
                                    a record of the other collection, passing
                                    its steps, has `theirs` equal to `mine`
            ["unlinked", ...]       none has
            ["use", name]           steps of the library, by name
    give    ["list", field] | ["count"] | ["pick", "?"] | ["pick", field]
            | ["any"]               whether there is one: yes or no

A `?` is filled from the message by looking its words up in the collection
(`says`): no word of English is listed here, the project is. `bind` fills a
way for a message, `run` carries it out, `said` says the answer.
"""
from __future__ import annotations

import ast
import re

#: the collections every project has
CODE = ("symbols", "uses", "files")
#: how many values an answer lists
LISTED = 30


def _raises(files: dict) -> list:
    """What each function of the Python files raises, by name."""
    out = []

    def walk(node, path, owner):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, path, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{owner}.{child.name}" if owner else child.name
                for inner in ast.walk(child):
                    if isinstance(inner, ast.Raise) and inner.exc is not None:
                        what = inner.exc.func if isinstance(
                            inner.exc, ast.Call) else inner.exc
                        said = getattr(what, "id", None) or getattr(
                            what, "attr", None)
                        if said:
                            out.append({"name": name, "file": path,
                                        "line": inner.lineno, "raise": said})
                walk(child, path, owner)
    for path, text in sorted(files.items()):
        if path.endswith(".py"):
            try:
                walk(ast.parse(text), path, "")
            except SyntaxError:
                continue
    return out


def world(held, last=None) -> dict:
    """The project as collections of records. `last`: the records the turn
    before was about."""
    index = held.index()
    symbols = [{"name": one.name, "kind": one.kind, "file": one.file,
                "line": one.start, "lines": one.end - one.start + 1,
                "doc": one.doc, "value": one.value, "owner": one.owner}
               for one in index.symbols.values()]
    named = {f"{one.file}#{one.name}": one.name
             for one in index.symbols.values()}
    uses = [{"name": named.get(use.symbol, use.symbol.rpartition("#")[2]),
             "file": use.file, "line": use.line, "user": use.by,
             "use": use.how,
             "test": use.by if use.by and index.is_test(use.file) else ""}
            for use in index.uses]
    files = [{"file": path, "lines": len(text.splitlines())}
             for path, text in sorted(held.files.items())]
    out = {"symbols": symbols, "uses": uses, "files": files,
           "raises": _raises(held.files)}
    try:
        models = held.data()
    except Exception:                               # noqa: BLE001
        models = {}
    for path, model in models.items():
        try:
            collections = model.records()
        except Exception:                           # noqa: BLE001
            continue
        for place, rows in collections.items():
            rows = [{key: value for key, value in row.items()
                     if isinstance(key, str) and
                     isinstance(value, (str, int, float, bool))}
                    for row in rows if isinstance(row, dict)]
            if len(rows) >= 2:
                out[f"{path}:{place}"] = rows
    if last:
        out["last"] = list(last)
    return out


def is_data(collection: str) -> bool:
    return ":" in collection


def _words(text: str) -> list:
    return [one.strip(".,?!'\"`()") for one in text.split()]


def said_as(text: str, value) -> str | None:
    """The word of a message that says a value, looked up: the value itself,
    more than one of it (`classes` of `class`), or a path by its last part
    (`hires.py` of `rental/hires.py`). None where it does not say it."""
    if not isinstance(value, str) or len(value) < 2:
        return None
    written, value = value, value.lower()
    words = _words(text)
    if " " in value:
        return value if value in " ".join(words).lower() else None
    for at, word in enumerate(words):
        # a word written with capitals inside the sentence is a name as it
        # is written (`Hire` is not `hire`); the first word's is the
        # sentence's
        if word == written:
            return word.lower()
        if at and word != word.lower():
            continue
        word = word.lower()
        if word == value:
            return word
        if len(value) >= 3 and word.startswith(value) and \
                word[len(value):] in ("s", "es"):
            return word
        if "/" in value and value.rpartition("/")[2] == word:
            return word
    return None


def says(text: str, value) -> bool:
    return said_as(text, value) is not None


def said_value(text: str, rows: list, field: str):
    """The value of a field, among those the records have, that the message
    says -- the longest where it says several. None where it says none."""
    found = {row.get(field) for row in rows
             if isinstance(row.get(field), str) and says(text, row[field])}
    # as it is written, where two differ only by capitals (`Hire`, `hire`)
    written = set(text.replace("?", " ").replace(",", " ").split())
    return max(found, key=lambda one: (one in written, len(one), one)) \
        if found else None


def expanded(steps: list, library: dict) -> list:
    out = []
    for step in steps:
        if step[0] == "use":
            out.extend(expanded(library[step[1]]["steps"], library))
        else:
            out.append(step)
    return out


def _bound(steps: list, rows: list, found: dict, text: str):
    """The steps with each `?` filled from the message; None where the
    message does not say one. The words that said them go into
    found["said"]."""
    out = []
    for step in steps:
        kind = step[0]
        if kind == "is":
            value = step[2]
            if value == "?":
                value = said_value(text, rows, step[1])
                if value is None:
                    return None
                found["said"].add(said_as(text, value))
            out.append(["is", step[1], value])
        elif kind == "has":
            if not says(text, step[1]):
                return None
            found["said"].add(said_as(text, step[1]))
            out.append(step)
        elif kind in ("linked", "unlinked"):
            other = found["world"].get(step[1]) or []
            inner = _bound(step[4], other, found, text)
            if inner is None:
                return None
            out.append([kind, step[1], step[2], step[3], inner])
        else:
            return None
    return out


def bind(way: dict, world_: dict, text: str, library: dict | None = None):
    """A way filled for a message: {from, steps, give, said}, the count of
    what the message said of it -- None where it does not fit."""
    library = library or {}
    sources = [one for one in world_ if is_data(one)] \
        if way["from"] == "data" else [way["from"]]
    for source in sources:
        rows = world_.get(source)
        if not rows:
            continue
        found = {"said": set(), "world": world_}
        steps = _bound(expanded(way["steps"], library), rows, found, text)
        if steps is None:
            continue
        give = list(way["give"])
        if give[0] == "pick":
            used = {step[1] for step in steps if step[0] in ("is", "has")}
            if give[1] == "?":
                fields = [one for one in rows[0] if one not in used and
                          says(text, one)]
            else:
                fields = [give[1]] if says(text, give[1]) else []
            if not fields:
                continue
            give = ["pick", fields[0]]
            found["said"].add(said_as(text, fields[0]))
        # each word of the message counted once, however many steps it fills
        return {"from": source, "steps": steps, "give": give,
                "said": len(found["said"]), "words": sorted(found["said"])}
    return None


def _passes(row: dict, steps: list, world_: dict) -> bool:
    for step in steps:
        kind = step[0]
        if kind == "is":
            if row.get(step[1]) != step[2]:
                return False
        elif kind == "has":
            if row.get(step[1]) in (None, "", False):
                return False
        else:
            mine = row.get(step[2])
            linked = any(other.get(step[3]) == mine and
                         _passes(other, step[4], world_)
                         for other in world_.get(step[1]) or ())
            if linked != (kind == "linked"):
                return False
    return True


def empty(bound: dict, value) -> bool:
    """Whether a way found nothing to say: a yes-or-no says no."""
    return bound["give"][0] != "any" and (value == [] or value is None or
                                          value == 0)


def run(bound: dict, world_: dict) -> tuple:
    """A bound way carried out: (the records, what it gives)."""
    rows = [row for row in world_.get(bound["from"]) or ()
            if _passes(row, bound["steps"], world_)]
    give = bound["give"]
    if give[0] == "count":
        return rows, len(rows)
    if give[0] == "any":
        return rows, bool(rows)
    values = []
    for row in rows:
        value = row.get(give[1])
        if value not in (None, "") and value not in values:
            values.append(value)
    return rows, values


def _named(row: dict) -> str:
    for key in ("name", "file"):
        if row.get(key):
            return str(row[key])
    return str(next((value for value in row.values()
                     if isinstance(value, str)), "it"))


def _described(steps: list) -> str:
    parts = []
    for step in steps:
        if step[0] == "is":
            parts.append(f"{step[1]} is {step[2]}")
        elif step[0] == "has":
            parts.append(f"a {step[1]}")
        else:
            inner = _described(step[4])
            parts.append(("with" if step[0] == "linked" else "without")
                         + f" {step[1]} of that {step[2]}"
                         + (f" having {inner}" if inner else ""))
    return ", ".join(parts)


def said(bound: dict, rows: list, value) -> str:
    """The answer in words: what was found first, then how it was found."""
    source = bound["from"].rstrip(":") if is_data(bound["from"]) else \
        bound["from"]
    how = _described(bound["steps"])
    where = f"{source}" + (f" where {how}" if how else "")
    give = bound["give"]
    if give[0] == "count":
        return f"{value}: the number of {where}."
    if give[0] == "any":
        if not value:
            return f"No: there are no {where}."
        return f"Yes: {', '.join(_named(row) for row in rows[:5])} -- {where}."
    listed = ", ".join(str(one) for one in value[:LISTED]) + \
        (" …" if len(value) > LISTED else "")
    if give[0] == "pick":
        if not value:
            return f"Nothing: no {give[1]} of {where}."
        names = ", ".join(_named(row) for row in rows[:5])
        return f"{listed}: the {give[1]} of {names}."
    if not value:
        return f"None: there are no {where}."
    return f"{listed}: the {len(value)} {give[1]}s of {where}."
