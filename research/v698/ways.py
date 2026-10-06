"""Ways of writing code: what a change like *use a switch statement* or
*make it iterative* asks of the program, checked on how it is written.

An encoder reads which ways a message asks for, any number at once
(`asked_ways.py`, taught from `teach_code_talk.py`'s messages); this says
what each way *is*, in the code: a test on the program's shape (`shape`:
what it is written with, whether it calls itself, how its entry is bound,
its length) -- in each language, as that language writes it (v699): a
switch is TypeScript's `switch` and Python's `match`, an arrow function is
Python's `lambda`, a template literal is an f-string. A way a language has
not (TypeScript's `const`, Python's `with`) is never held against its code.
A way the code is not checked against is not here -- *memoize it* is read
as a change and said to the writers, but nothing holds an answer to it.

Each way belongs to a family; a later way replaces an earlier one of its
family (*make it recursive*, then *make it iterative*), others accumulate.
The `language` family is which language the code is asked in (*in
Python*): read by the same encoder, and what the request is written in.
"""
from __future__ import annotations

LOOPS = ("for", "for of", "for in", "while")
PY_LOOPS = ("for", "while")


def _has(*words):
    return lambda shape, before: any(one in shape["uses"] for one in words)


def _lacks(*words):
    return lambda shape, before: not any(one in shape["uses"]
                                         for one in words)


def _entry(kind):
    return lambda shape, before: shape["entry"] == kind


def _python_for_of(shape, before):
    """A loop over the items themselves, not a count."""
    return "for" in shape["uses"] and not any(
        one.startswith("for in range") for one in shape["uses"])


def _python_counted(shape, before):
    return any(one.startswith("for in range") for one in shape["uses"])


#: name: (family, TypeScript (said, test), Python (said, test)); a
#: language's pair is None where it has no such way. "none" first: a
#: message that asks for no way of writing.
TABLE = {
    "none": (None, None, None),
    "switch": ("branch", ("with a switch statement", _has("switch")),
               ("with a match statement", _has("match"))),
    "ifs": ("branch",
            ("with if/else and no switch",
             lambda s, b: "switch" not in s["uses"]
             and ("if" in s["uses"] or "?:" in s["uses"])),
            ("with if/elif/else and no match",
             lambda s, b: "match" not in s["uses"]
             and ("if" in s["uses"] or "?:" in s["uses"]))),
    "ternary": ("branch", ("with a conditional (?:) expression", _has("?:")),
                ("with a conditional expression (x if c else y)",
                 _has("?:"))),
    "recursive": ("recursion", ("recursively", lambda s, b: s["recursive"]),
                  ("recursively", lambda s, b: s["recursive"])),
    "iterative": ("recursion",
                  ("without recursion", lambda s, b: not s["recursive"]),
                  ("without recursion", lambda s, b: not s["recursive"])),
    "loop": ("loop", ("with a loop", _has(*LOOPS)),
             ("with a loop", _has(*PY_LOOPS))),
    "no-loop": ("loop", ("without a loop statement", _lacks(*LOOPS)),
                ("without a loop statement", _lacks(*PY_LOOPS))),
    "for": ("loop", ("with a for loop", _has("for")),
            ("with a for loop over a range", _python_counted)),
    "while": ("loop", ("with a while loop", _has("while")),
              ("with a while loop", _has("while"))),
    "for-of": ("loop", ("with a for...of loop", _has("for of")),
               ("with a for loop over the items", _python_for_of)),
    "map": ("map", ("with .map", _has(".map")),
            ("with map() or a comprehension",
             _has("map()", "list comprehension", "generator expression"))),
    "filter": ("filter", ("with .filter", _has(".filter")),
               ("with filter() or a comprehension with if",
                _has("filter()", "comprehension with if"))),
    "reduce": ("reduce", ("with .reduce", _has(".reduce")),
               ("with functools.reduce",
                _has("reduce()", "functools.reduce", ".reduce"))),
    "forEach": ("forEach", ("with .forEach", _has(".forEach")),
                None),
    "regex": ("regex", ("with a regular expression",
                        _has("regex", "new RegExp")),
              ("with a regular expression (re)",
               lambda s, b: any(one.startswith(("re.", "import re"))
                                for one in s["uses"]))),
    "no-regex": ("regex", ("without a regular expression",
                           _lacks("regex", "new RegExp")),
                 ("without a regular expression",
                  lambda s, b: not any(one.startswith(("re.", "import re"))
                                       for one in s["uses"]))),
    "set": ("set", ("with a Set", _has("new Set")),
            ("with a set", _has("set()", "set literal",
                                "set comprehension"))),
    # one family with .map: a message asking for both is rare, and `map`
    # read as both is what the reader confused
    "map-object": ("map", ("with a Map", _has("new Map")),
                   ("with a dict", _has("dict()", "dict literal",
                                        "dict comprehension"))),
    "async": ("async", ("with async/await", _has("async", "await")),
              ("with async/await", _has("async", "await"))),
    "arrow": ("form", ("as an arrow function", _entry("arrow")),
              ("as a lambda", _entry("lambda"))),
    "declaration": ("form", ("as a function declaration",
                             _entry("function")),
                    ("as a def", _entry("function"))),
    "class": ("form", ("with a class", _has("class")),
              ("with a class", _has("class"))),
    "one-liner": ("length", ("in one line",
                             lambda s, b: s["statements"] <= 1),
                  ("in one line", lambda s, b: s["statements"] <= 1)),
    "shorter": ("length", ("shorter than before",
                           lambda s, b: b is None or s["lines"] < b["lines"]),
                ("shorter than before",
                 lambda s, b: b is None or s["lines"] < b["lines"])),
    "template": ("template", ("with a template literal", _has("template")),
                 ("with an f-string", _has("f-string"))),
    "spread": ("spread", ("with the spread operator", _has("...")),
               ("with * unpacking", _has("*"))),
    "destructuring": ("destructuring", ("with destructuring",
                                        _has("destructuring")),
                      ("with tuple unpacking", _has("unpacking"))),
    "const": ("const", ("with const and no let or var",
                        _lacks("let", "var")), None),
    # Python's own
    "comprehension": ("comprehension", None,
                      ("with a list comprehension",
                       _has("list comprehension", "set comprehension",
                            "dict comprehension"))),
    "no-comprehension": ("comprehension", None,
                         # as people say it: the teacher, asked for code
                         # `without a comprehension`, wrote `fix the bugs`
                         ("without a list comprehension",
                          _lacks("list comprehension", "set comprehension",
                                 "dict comprehension",
                                 "generator expression"))),
    "generator": ("generator", None,
                  ("with a generator", _has("generator expression",
                                            "yield"))),
    "enumerate": ("enumerate", None,
                  ("with enumerate", _has("enumerate()"))),
    "zip": ("zip", None, ("with zip", _has("zip()"))),
    "annotations": ("annotations", None,
                    ("with type hints", _has("annotations"))),
    "with": ("with", None, ("with a with statement", _has("with"))),
    "decorator": ("decorator", None, ("with a decorator",
                                      _has("decorator"))),
    # which language it is written in
    "python": ("language", ("in Python", lambda s, b: False),
               ("in Python", lambda s, b: True)),
    "typescript": ("language", ("in TypeScript", lambda s, b: True),
                   ("in TypeScript", lambda s, b: False)),
}

LANGUAGES = ("typescript", "python")


def _language_of(shape) -> str:
    return (shape or {}).get("language", "typescript")


def _pair(way: str, language: str):
    family, ts, py = TABLE[way]
    return py if language == "python" else ts


#: name: (family, how TypeScript says it, its TypeScript test) -- as before
#: v699, for what reads one language
WAYS = {name: (family, (ts or ("", None))[0] if ts else
               (py[0] if py else ""), ts[1] if ts else None)
        for name, (family, ts, py) in TABLE.items()}

NAMES = tuple(TABLE)


def merged(before, way: str | None) -> list:
    """The ways asked so far, with `way` added: it replaces an earlier way
    of its family."""
    out = list(before or ())
    if not way or way == "none" or way not in TABLE:
        return out
    family = TABLE[way][0]
    out = [one for one in out if TABLE[one][0] != family]
    return out + [way]


def said(ways, language: str = "typescript") -> str:
    """The ways as words, as the language says them: `with a switch
    statement and recursively`; `with a match statement and ...`."""
    parts = []
    for one in ways or ():
        if one not in TABLE:
            continue
        pair = _pair(one, language) or _pair(one, "typescript") or \
            _pair(one, "python")
        if pair:
            parts.append(pair[0])
    return " and ".join(parts)


def held(ways, language: str = "typescript") -> list:
    """The ways a program in this language can be held to: the language
    family is what it is written in, not a way it is written; a way the
    language has not is not held against it."""
    return [one for one in ways or () if one in TABLE
            and TABLE[one][0] != "language"
            and _pair(one, language) is not None]


def asked_language(ways) -> str | None:
    """The language the ways ask the code in, if they ask one."""
    for one in ways or ():
        if one in LANGUAGES:
            return one
    return None


def fits(shape: dict | None, ways, before: dict | None = None) -> bool:
    """Whether code of this shape is written each of these ways."""
    language = _language_of(shape)
    ways = held(ways, language)
    if not ways:
        return True
    if shape is None:
        return False
    return all(_pair(one, language)[1](shape, before) for one in ways)


def missing(shape: dict | None, ways, before: dict | None = None) -> list:
    """The ways this code is not written."""
    language = _language_of(shape)
    ways = held(ways, language)
    if shape is None:
        return list(ways)
    return [one for one in ways
            if not _pair(one, language)[1](shape, before)]
