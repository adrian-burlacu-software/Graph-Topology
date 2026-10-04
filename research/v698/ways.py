"""Ways of writing code: what a change like *use a switch statement* or
*make it iterative* asks of the program, checked on how it is written.

An encoder reads which ways a message asks for, any number at once
(`asked_ways.py`, taught from `teach_code_talk.py`'s messages); this says
what each way *is*, in the code: a test
on the program's shape (`tscheck.js` `shape`: what it is written with,
whether it calls itself, how its entry is bound, its length). A way the
code is not checked against is not here -- *memoize it* is read as a
change and said to the writers, but nothing holds an answer to it.

Each way belongs to a family; a later way replaces an earlier one of its
family (*make it recursive*, then *make it iterative*), others accumulate.
"""
from __future__ import annotations

LOOPS = ("for", "for of", "for in", "while")


def _has(*words):
    return lambda shape, before: any(one in shape["uses"] for one in words)


def _lacks(*words):
    return lambda shape, before: not any(one in shape["uses"]
                                         for one in words)


#: name: (family, what it is said as, its test on (shape, shape before)).
#: "none" first: a message that asks for no way of writing.
WAYS = {
    "none": (None, "", None),
    "switch": ("branch", "with a switch statement", _has("switch")),
    "ifs": ("branch", "with if/else and no switch",
            lambda s, b: "switch" not in s["uses"]
            and ("if" in s["uses"] or "?:" in s["uses"])),
    "ternary": ("branch", "with a conditional (?:) expression", _has("?:")),
    "recursive": ("recursion", "recursively",
                  lambda s, b: s["recursive"]),
    "iterative": ("recursion", "without recursion",
                  lambda s, b: not s["recursive"]),
    "loop": ("loop", "with a loop", _has(*LOOPS)),
    "no-loop": ("loop", "without a loop statement", _lacks(*LOOPS)),
    "for": ("loop", "with a for loop", _has("for")),
    "while": ("loop", "with a while loop", _has("while")),
    "for-of": ("loop", "with a for...of loop", _has("for of")),
    "map": ("map", "with .map", _has(".map")),
    "filter": ("filter", "with .filter", _has(".filter")),
    "reduce": ("reduce", "with .reduce", _has(".reduce")),
    "forEach": ("forEach", "with .forEach", _has(".forEach")),
    "regex": ("regex", "with a regular expression",
              _has("regex", "new RegExp")),
    "no-regex": ("regex", "without a regular expression",
                 _lacks("regex", "new RegExp")),
    "set": ("set", "with a Set", _has("new Set")),
    # one family with .map: a message asking for both is rare, and `map`
    # read as both is what the reader confused
    "map-object": ("map", "with a Map", _has("new Map")),
    "async": ("async", "with async/await", _has("async", "await")),
    "arrow": ("form", "as an arrow function",
              lambda s, b: s["entry"] == "arrow"),
    "declaration": ("form", "as a function declaration",
                    lambda s, b: s["entry"] == "function"),
    "class": ("form", "with a class", _has("class")),
    "one-liner": ("length", "in one line",
                  lambda s, b: s["statements"] <= 1),
    "shorter": ("length", "shorter than before",
                lambda s, b: b is None or s["lines"] < b["lines"]),
    "template": ("template", "with a template literal", _has("template")),
    "spread": ("spread", "with the spread operator", _has("...")),
    "destructuring": ("destructuring", "with destructuring",
                      _has("destructuring")),
    "const": ("const", "with const and no let or var",
              _lacks("let", "var")),
}

NAMES = tuple(WAYS)


def merged(before, way: str | None) -> list:
    """The ways asked so far, with `way` added: it replaces an earlier way
    of its family."""
    out = list(before or ())
    if not way or way == "none" or way not in WAYS:
        return out
    family = WAYS[way][0]
    out = [one for one in out if WAYS[one][0] != family]
    return out + [way]


def said(ways) -> str:
    """The ways as words: `with a switch statement and recursively`."""
    parts = [WAYS[one][1] for one in ways or () if one in WAYS]
    return " and ".join(parts)


def fits(shape: dict | None, ways, before: dict | None = None) -> bool:
    """Whether code of this shape is written each of these ways."""
    if not ways:
        return True
    if shape is None:
        return False
    return all(WAYS[one][2](shape, before) for one in ways if one in WAYS
               and WAYS[one][2] is not None)


def missing(shape: dict | None, ways, before: dict | None = None) -> list:
    """The ways this code is not written."""
    if shape is None:
        return list(ways or ())
    return [one for one in ways or () if one in WAYS
            and WAYS[one][2] is not None
            and not WAYS[one][2](shape, before)]
