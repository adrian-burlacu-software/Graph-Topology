"""Where in a data file a change is (v701).

The editor was taught changes of data with the lines around them
(`teach_data_edits.py`: a value's line, the last record of a collection,
`teach_editor.AROUND` lines each side), never a file whole: given all of
`service.yaml`, it added a user after `models.teacher.offline`; given the
lines around `users`, 7 of 7 answers added it after the last user.

So a statement is placed in a data file's own blocks -- what is written
under a key, by its indent -- by the words it says that the file has
(looked up in the file, a word or one of it: `user` is in `users`): from
the whole file down, into the one block that still holds all of them, until
none does or two do.
"""
from __future__ import annotations

import re

from research.v701.querying import _one

#: a line of only brackets and commas (`},`, `]`) closes what is above
CLOSING = re.compile(r"^\s*[\]\[{}(),]*\s*$")


def _words(text: str) -> set:
    """A text's names, each whole -- `llm/change-judge2` is one, not
    `change` -- but for `_` (`timeout_seconds` is `timeout seconds`)."""
    return {_one(one.lower().rstrip("./-"))
            for one in re.findall(r"[A-Za-z0-9][A-Za-z0-9./-]*", text)}


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _children(lines: list, first: int, last: int) -> list:
    """The blocks directly under lines [first, last] (0-based, inclusive):
    each from a line at their least indent to the line before the next."""
    starts = [at for at in range(first, last + 1)
              if lines[at].strip() and not CLOSING.match(lines[at])]
    if not starts:
        return []
    least = min(_indent(lines[at]) for at in starts)
    heads = [at for at in starts if _indent(lines[at]) == least]
    return [(head, (heads[at + 1] - 1) if at + 1 < len(heads) else last)
            for at, head in enumerate(heads)]


def block(lines: list, statement: str, rows: bool = False) -> tuple | None:
    """(first, last), 0-based and inclusive: the smallest block of a data
    file holding every word of the statement the file has; None where it
    has none of them. `rows`: a table, its first line the names of its
    columns -- a word only that line holds says which column, not which
    row (`set Dune's pages to 896`: `pages` is the header's, and the whole
    file was the block -- the editor shown its last rows changed Solaris)."""
    said = _words(statement)
    # a comment is no data (`# The page and the reader` holds `the`)
    held = [_words(re.sub(r"(^|\s)#.*$", "", line)) & said
            for line in lines]
    if rows and len(held) > 1:
        others = set().union(*held[1:])
        if others:
            held[0] = held[0] & others
    def has(first: int, last: int) -> set:
        return set().union(*held[first:last + 1]) if last >= first else set()
    first, last = 0, len(lines) - 1
    wanted = has(first, last)
    if not wanted:
        return None
    outer = True
    while True:
        inner = _children(lines, first + (0 if outer else 1), last)
        holding = [one for one in inner if has(*one) == wanted]
        if len(holding) != 1 or holding[0] == (first, last):
            break
        first, last = holding[0]
        outer = False
    # a closing line after it belongs to it
    while last + 1 < len(lines) and CLOSING.match(lines[last + 1]) and \
            lines[last + 1].strip():
        last += 1
    return first, last


def data_file(held, statement: str) -> str | None:
    """The data file a change naming no file is of -- where the code has no
    name for what it says (the code is changed where it has: `make loans
    last 21 days` is of loans.py's LOAN_DAYS, though config.yaml holds
    loan_days too). A data file is what is changed where the statement
    says, of it:

    - a value it holds, whole (`set Dune's pages to 896`, `remove Solaris`);
    - two fields or more of one of its collections: a record described
      (`add a member with id 5 named ..., email ...`);
    - a setting the code has no name of (`set the port to 9090`);
    - the file itself, where no code file is so named (`... in the
      config`).

    The one that says most; None where none does. Looked up in what the
    project holds, not read."""
    said = _words(statement)
    lowered = " " + re.sub(r"\s+", " ", statement.lower()) + " "
    code = _code_words(held)
    code_files = {_one(path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower())
                  for path in held.files if not _is_data(path)}
    best, score = None, 0
    for path, model in sorted(held.data().items()):
        if model.error:
            continue
        fit = 0
        leaves = [place for place in model.places.values()
                  if not set(place.types) & {"object", "array"}]
        # a value it holds, said whole
        if any(isinstance(value, str) and len(value) >= 3 and
               f" {value.lower()}" in lowered
               for place in leaves for value in list(place.distinct)[:500]):
            fit += 2
        # two fields of a collection or more: a record
        for collection, one in model.collections.items():
            named = {name for name in one.get("fields", ())
                     if _words(name) and _words(name) <= said}
            if len(named) >= 2:
                fit += 2
        # a setting the code has no name of
        for place in leaves:
            own = _words(place.path.rsplit(".", 1)[-1])
            if own and own <= said and not own <= code and \
                    "[]" not in place.path and not any(
                        place.path.startswith(f"{one}.") or place.path == one
                        for one in model.collections if one):
                fit += 1
        # the file named, where no code file is so named -- a file of
        # settings: one of records is named for what each is (`let members
        # hold up to 5 books` is of the code's limit, not members.json)
        base = _one(path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower())
        if (base in said and base not in code_files and
                not {"", "[]"} & set(model.collections)) or \
                path.lower() in lowered:
            fit += 1
        if fit > score:
            best, score = path, fit
    return best


def _shape(value) -> tuple:
    """What a value looks like: a number, or a string by what it has -- an
    `@`, a digit, a dot, a space."""
    if isinstance(value, bool):
        return ("bool",)
    if isinstance(value, (int, float)):
        return ("number",)
    text = str(value)
    return ("string", "@" in text, any(one.isdigit() for one in text),
            "." in text, " " in text)


def _as(token: str, old):
    """A word of the statement as a value of `old`'s kind, or None."""
    if isinstance(old, bool):
        return None
    if isinstance(old, int):
        return int(token) if re.fullmatch(r"-?\d+", token) else None
    if isinstance(old, float):
        return float(token) if re.fullmatch(r"-?\d+(\.\d+)?", token) \
            else None
    # a string only where its shape is the old one's and says more than a
    # word does (an address, a date): plain words are the statement's own
    if _shape(token) == _shape(old) and _shape(old)[1:3] != (False, False):
        return token
    return None


def settings(path: str, text: str, statement: str) -> list:
    """A value of the data set as the statement says, made by lookup, not
    written: [(what, the file's text so changed)]. A record named by a value
    it holds whole (`Dune`), a field of it named (`pages`), and a word of
    the statement of that field's kind (`896`); or a setting named (`port`)
    and a value of its kind (`9090`). Only where it is one line of the
    file, changed in place."""
    from research.v701 import datamodel
    import json
    model = datamodel.read(path, text)
    if model.error:
        return []
    said = _words(statement)
    lowered = " " + re.sub(r"\s+", " ", statement.lower()) + " "
    tokens = [one.rstrip(".,;:!?") for one in statement.split()]
    tokens = [re.sub(r"'s$", "", one) for one in tokens if one]
    lines = text.splitlines(keepends=True)
    out = []

    def named(name: str) -> bool:
        return bool(_words(name)) and _words(name) <= said

    def put(at: int, line: str, what: str):
        changed = "".join(lines[:at] + [line] + lines[at + 1:])
        if changed != text and (what, changed) not in out:
            out.append((what, changed))
    for collection, records in model.records().items():
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            by = [(name, value) for name, value in record.items()
                  if isinstance(value, str) and len(value) >= 3 and
                  f" {value.lower()}" in lowered]
            if not by:
                continue
            for field_, old in record.items():
                if not named(field_) or any(field_ == one for one, _ in by):
                    continue
                for token in tokens:
                    new = _as(token, old)
                    if new is None or new == old:
                        continue
                    what = f"{by[0][1]}'s {field_} set to {new}"
                    if tabular(path):
                        import csv
                        import io
                        # the header, then each record a line
                        at = index + 1
                        if at >= len(lines):
                            continue
                        cells = next(csv.reader([lines[at]]))
                        header = next(csv.reader([lines[0]]))
                        if field_ not in header or \
                                len(cells) != len(header):
                            continue
                        cells[header.index(field_)] = str(new)
                        buffer = io.StringIO()
                        csv.writer(buffer, lineterminator="").writerow(cells)
                        ending = lines[at][len(lines[at].rstrip("\r\n")):]
                        put(at, buffer.getvalue() + ending, what)
                    elif path.lower().endswith(".json"):
                        key = f'"{by[0][0]}": {json.dumps(by[0][1])}'
                        old_ = f'"{field_}": {json.dumps(old)}'
                        found = [at for at, line in enumerate(lines)
                                 if key in line and old_ in line]
                        if len(found) == 1:
                            put(found[0], lines[found[0]].replace(
                                old_, f'"{field_}": {json.dumps(new)}'), what)
    # a setting: a value of its own, named
    for place in model.places.values():
        if set(place.types) & {"object", "array"} or "[]" in place.path:
            continue
        own = place.path.rsplit(".", 1)[-1]
        if not named(own) or not place.examples:
            continue
        old = place.examples[0]
        for token in tokens:
            new = _as(token, old)
            if new is None or new == old:
                continue
            pattern = re.compile(rf"^(\s*{re.escape(own)}\s*:\s*)"
                                 rf"{re.escape(str(old))}(\s*(#.*)?)$")
            found = [at for at, line in enumerate(lines)
                     if pattern.match(line.rstrip("\r\n"))]
            if len(found) == 1:
                line = lines[found[0]]
                ending = line[len(line.rstrip("\r\n")):]
                put(found[0], pattern.sub(rf"\g<1>{new}\g<2>",
                                          line.rstrip("\r\n")) + ending,
                    f"{place.path} set to {new}")
    return out


def tabular(path: str) -> bool:
    """Whether a data file is a table of rows under a header (`block`)."""
    return path.lower().endswith((".csv", ".tsv"))


def _is_data(path: str) -> bool:
    from research.v698.project import is_data
    return is_data(path)


def _code_words(held) -> set:
    """Every word of every name the project's code uses (`LOAN_DAYS`: loan,
    days; `find_by_author`: find, by, author)."""
    import io
    import tokenize
    out = set()
    for path, text in held.files.items():
        if _is_data(path):
            continue
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            continue
        for token in tokens:
            if token.type == tokenize.NAME:
                for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+",
                                       token.string):
                    out.add(_one(part.lower()))
    return out


def window(text: str, statement: str, around: int,
           longest: int, rows: bool = False) -> tuple | None:
    """(first line, last line), 1-based and inclusive, of a data file to
    give the editor: its block (`block`) and `around` lines each side;
    None where the statement says nothing the file has, or the block is
    longer than `longest`."""
    lines = text.splitlines()
    found = block(lines, statement, rows)
    if found is None:
        return None
    first = max(0, found[0] - around)
    last = min(len(lines) - 1, found[1] + around)
    if last - first + 1 > longest:
        return None
    return first + 1, last + 1
