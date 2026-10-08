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


def block(lines: list, statement: str) -> tuple | None:
    """(first, last), 0-based and inclusive: the smallest block of a data
    file holding every word of the statement the file has; None where it
    has none of them."""
    said = _words(statement)
    # a comment is no data (`# The page and the reader` holds `the`)
    held = [_words(re.sub(r"(^|\s)#.*$", "", line)) & said
            for line in lines]
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


def window(text: str, statement: str, around: int,
           longest: int) -> tuple | None:
    """(first line, last line), 1-based and inclusive, of a data file to
    give the editor: its block (`block`) and `around` lines each side;
    None where the statement says nothing the file has, or the block is
    longer than `longest`."""
    lines = text.splitlines()
    found = block(lines, statement)
    if found is None:
        return None
    first = max(0, found[0] - around)
    last = min(len(lines) - 1, found[1] + around)
    if last - first + 1 > longest:
        return None
    return first + 1, last + 1
