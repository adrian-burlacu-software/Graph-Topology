"""Data said in a message -- a JSON object, a CSV's rows, a YAML block --
held as the conversation's data, and the question asked with it answered
from it.

    here is users.csv:
    name,role,age
    adrian,owner,41
    how many users are older than 30?

What marks data is that it parses as data, not a word of the message
(`asking.pasted` finds code the same way): a fenced block that reads as
JSON, YAML or CSV; unfenced, lines that read as a JSON object or list, or
as a CSV's header and rows (two columns or more, every row as wide as the
header). A file name said beside it (`users.csv`) names it; else it is
`pasted-N` of its format. It is put in the conversation's project
(`Project.put`), so what is asked of it is asked as of any data file.
"""
from __future__ import annotations

import json
import re

FENCE = re.compile(r"```([A-Za-z]*)[ \t]*\n(.*?)```", re.S)
NAMED = re.compile(r"[\w./-]+\.(json|ya?ml|csv|tsv)\b", re.I)


def _json(text: str):
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, (dict, list)) else None


def _csv_lines(lines: list) -> tuple | None:
    """(first, last) of the longest run of lines reading as a CSV header
    and rows: two columns or more, as wide as the header, two rows or
    more."""
    best = None
    at = 0
    while at < len(lines):
        width = lines[at].count(",")
        end = at + 1
        while end < len(lines) and width >= 1 and \
                lines[end].count(",") == width and lines[end].strip():
            end += 1
        if width >= 1 and end - at >= 3 and (best is None or
                                             end - at > best[1] - best[0]):
            best = (at, end)
        at = max(end, at + 1)
    return best


def found(text: str) -> dict | None:
    """{format, text, name, rest}: the data said in a message, what it is
    called, and the message without it -- or None."""
    named = NAMED.search(text)
    for match in FENCE.finditer(text):
        kind, inside = match.group(1).lower(), match.group(2)
        form = {"yml": "yaml"}.get(kind, kind)
        if form not in ("json", "yaml", "csv", "tsv", ""):
            continue
        if form in ("", "json") and _json(inside) is not None:
            form = "json"
        elif form == "":
            if _csv_lines(inside.splitlines()):
                form = "csv"
            else:
                continue
        rest = (text[:match.start()] + text[match.end():]).strip()
        return {"format": form, "text": inside, "rest": rest,
                "name": named.group(0) if named else None}
    lines = text.splitlines()
    # a JSON object or list, from a line that opens one to the line that
    # closes it
    for at, line in enumerate(lines):
        if line.lstrip().startswith(("{", "[")):
            for end in range(len(lines), at, -1):
                said = "\n".join(lines[at:end])
                if _json(said) is not None:
                    rest = "\n".join(lines[:at] + lines[end:]).strip()
                    return {"format": "json", "text": said, "rest": rest,
                            "name": named.group(0) if named else None}
    run = _csv_lines(lines)
    if run is not None:
        # a request's examples (`f(1, 2) == 3`) are as wide as each other
        # too: what the request reader reads as examples is not data
        from research.v697.coding import read as request
        if request("\n".join(lines[run[0]:run[1]])).get("examples"):
            return None
        said = "\n".join(lines[run[0]:run[1]]) + "\n"
        rest = "\n".join(lines[:run[0]] + lines[run[1]:]).strip()
        return {"format": "csv", "text": said, "rest": rest,
                "name": named.group(0) if named else None}
    return None


def hold(text: str, key) -> dict | None:
    """The data said in a message, put in the conversation's project:
    {path, model, rest} -- or None where there is none, or it does not
    read as data."""
    from research.v698 import project as Pj
    from research.v701 import datamodel
    said = found(text)
    if said is None:
        return None
    held = Pj.project(key)
    count = len(held.data()) if held is not None else 0
    path = said["name"] or f"pasted-{count + 1}.{said['format']}"
    if datamodel.format_of(path) is None:
        path += f".{said['format']}"
    model = datamodel.read(path, said["text"], said["format"])
    if model.error:
        return None
    if held is None:
        Pj.put(key, {path: said["text"]}, name="pasted")
    else:
        held.put({path: said["text"]})
    return {"path": path, "model": model, "rest": said["rest"]}
