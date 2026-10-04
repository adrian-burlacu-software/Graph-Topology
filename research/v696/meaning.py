"""What a function means: one structure, whatever it was read from.

English and code are read together into this (`PLAN.md`, "Reading code
and English together"), and it is what the search, recognition and the
checker use:

    takes, returns   the types
    behaviour        predicates of how the output stands to the inputs,
                     read off input/output pairs -- by running, for code
    uses             what the program is made of: members named by the
                     interface that declares them (`String.split`), the
                     operators, the forms and statements (`for of`, `if`,
                     `recursion`)
    root             what gives its result

The exact readings here -- `behaviour` by values, `structure` by the
compiler -- are the teachers and the checkers of the learned reader
(`reader.py`), never a second reader beside it.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from research.v696 import program as P


@dataclass(frozen=True)
class Meaning:
    takes: tuple
    returns: str
    #: None where no input/output pairs are known
    behaviour: frozenset | None = frozenset()
    #: None where nothing verified says what it is made of
    uses: frozenset | None = None
    root: str | None = None

    def json(self) -> dict:
        return {"takes": list(self.takes), "returns": self.returns,
                "behaviour": None if self.behaviour is None
                else sorted(self.behaviour),
                "uses": None if self.uses is None else sorted(self.uses),
                "root": self.root}

    @classmethod
    def of(cls, row: dict) -> "Meaning":
        return cls(tuple(row["takes"]), row["returns"],
                   None if row["behaviour"] is None
                   else frozenset(row["behaviour"]),
                   None if row["uses"] is None else frozenset(row["uses"]),
                   row["root"])


# -- behaviour, read off values ---------------------------------------------

def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _length(value):
    return len(value) if isinstance(value, (str, list)) else None


def _key(value) -> str:
    return json.dumps(value, sort_keys=True)


def _all(pairs, test) -> bool:
    try:
        return all(test(a, b) for a, b in pairs)
    except (TypeError, ValueError, OverflowError):
        return False


def _some(pairs, test) -> bool:
    try:
        return any(test(a, b) for a, b in pairs)
    except (TypeError, ValueError, OverflowError):
        return False


def _sorted(values, reverse=False) -> bool:
    try:
        return values == sorted(values, reverse=reverse)
    except TypeError:
        return False


def _close(a, b) -> bool:
    return _number(a) and _number(b) and math.isclose(a, b, rel_tol=1e-9,
                                                      abs_tol=1e-9)


def _numbers(value) -> bool:
    return isinstance(value, list) and value and all(map(_number, value))


def behaviour(pairs) -> frozenset:
    """What every pair shows of how the output stands to the inputs:
    `pairs` is [(args, output)]. A predicate is only said where a pair
    could have shown otherwise (a sorted output of length one says
    nothing). Inputs are named by position: `input 1`."""
    pairs = [(list(args), out) for args, out in pairs]
    if not pairs:
        return frozenset()
    outputs = [out for _, out in pairs]
    said = set()
    if len({_key(one) for one in outputs}) == 1 and len(pairs) > 1:
        said.add("the same output always")
    # -- the output on its own
    if all(isinstance(one, bool) for one in outputs):
        if len(set(outputs)) > 1:
            said.add("true or false")
    elif all(_number(one) for one in outputs):
        if all(float(one).is_integer() for one in outputs):
            said.add("a whole number")
        else:
            said.add("a fraction")
        if all(one >= 0 for one in outputs) and any(one == 0
                                                    for one in outputs):
            said.add("zero or more")
        if any(one < 0 for one in outputs):
            said.add("can be negative")
    elif all(isinstance(one, str) for one in outputs):
        letters = [one for one in outputs if any(c.isalpha() for c in one)]
        if letters and all(one == one.upper() for one in letters):
            said.add("upper case")
        if letters and all(one == one.lower() for one in letters):
            said.add("lower case")
        if any(one == "" for one in outputs):
            said.add("can be empty")
    elif all(isinstance(one, list) for one in outputs):
        long = [one for one in outputs if len(one) > 1]
        if long and all(_sorted(one) for one in long) \
                and any(len(set(map(_key, one))) > 1 for one in long):
            said.add("sorted")
        if long and all(_sorted(one, True) for one in long) \
                and any(len(set(map(_key, one))) > 1 for one in long):
            said.add("sorted down")
        if long and all(len(set(map(_key, one))) == len(one)
                        for one in outputs):
            said.add("no repeats")
        if any(one == [] for one in outputs):
            said.add("can be empty")
    # -- the output against each input
    for at in range(len(pairs[0][0])):
        name = f"input {at + 1}"
        each = [(args[at], out) for args, out in pairs if len(args) > at]
        if not each:
            continue
        if _all(each, lambda a, b: _key(a) == _key(b)):
            said.add(f"is {name}")
            continue
        if _all(each, lambda a, b: _length(a) is not None
                and _length(a) == _length(b)):
            said.add(f"as long as {name}")
        elif _all(each, lambda a, b: _length(a) is not None
                  and _length(b) is not None and _length(b) <= _length(a)) \
                and _some(each, lambda a, b: _length(b) < _length(a)):
            said.add(f"shorter than {name}")
        elif _all(each, lambda a, b: _length(a) is not None
                  and _length(b) is not None and _length(b) >= _length(a)) \
                and _some(each, lambda a, b: _length(b) > _length(a)):
            said.add(f"longer than {name}")
        if _all(each, lambda a, b: _length(a) is not None and _number(b)
                and b == _length(a)):
            said.add(f"the length of {name}")
        elif _all(each, lambda a, b: _length(a) is not None and _number(b)
                  and 0 <= b <= _length(a) and float(b).is_integer()):
            said.add(f"a count within {name}")
        if _all(each, lambda a, b: isinstance(a, list) and not isinstance(
                b, list) and _key(b) in map(_key, a)):
            said.add(f"one of {name}")
        if _all(each, lambda a, b: isinstance(a, str) and isinstance(b, str)
                and b in a) and _some(each, lambda a, b: b != a):
            said.add(f"a part of {name}")
        if _all(each, lambda a, b: isinstance(a, list) and isinstance(b, list)
                and all(_key(one) in set(map(_key, a)) for one in b)) \
                and _some(each, lambda a, b: _key(a) != _key(b)):
            said.add(f"taken from {name}")
        if _all(each, lambda a, b: isinstance(a, (list, str))
                and isinstance(b, type(a))
                and sorted(map(_key, a)) == sorted(map(_key, b))) \
                and _some(each, lambda a, b: _key(a) != _key(b)):
            said.add(f"reordered {name}")
        if _all(each, lambda a, b: isinstance(a, (list, str))
                and isinstance(b, type(a)) and b == a[::-1]) \
                and _some(each, lambda a, b: a != a[::-1]):
            said.add(f"reversed {name}")
        if _all(each, lambda a, b: isinstance(a, str) and isinstance(b, str)
                and a.lower() == b.lower() and a != b) :
            said.add(f"the letters of {name} recased")
        if _all(each, lambda a, b: _numbers(a) and _close(b, sum(a))):
            said.add(f"the sum of {name}")
        if _all(each, lambda a, b: _numbers(a) and _close(b, max(a))):
            said.add(f"the largest of {name}")
        if _all(each, lambda a, b: _numbers(a) and _close(b, min(a))):
            said.add(f"the smallest of {name}")
        if _all(each, lambda a, b: _numbers(a) and _number(b)
                and _close(b, math.prod(a))):
            said.add(f"the product of {name}")
        if _all(each, lambda a, b: _number(a) and _number(b)):
            if _all(each, lambda a, b: b > a):
                said.add(f"more than {name}")
            elif _all(each, lambda a, b: b < a):
                said.add(f"less than {name}")
            elif _all(each, lambda a, b: abs(b) <= abs(a)) \
                    and _some(each, lambda a, b: abs(b) < abs(a)):
                said.add(f"no bigger than {name}")
    return frozenset(said)


# -- what code is made of ---------------------------------------------------

#: A receiver type as the interface declaring its members.
INTERFACES = {"string": "String", "number": "Number", "boolean": "Boolean"}


def interface(type_: str) -> str:
    if type_.endswith("[]"):
        return "Array"
    return INTERFACES.get(type_, type_)


def word(op) -> str:
    """A library operator (`program.Op`) or form (`program.Form`) said as
    the compiler's structure reading says it, so what the reader expects
    and what the search grows are named alike."""
    if isinstance(op, P.Form):
        return f"{interface(op.receiver)}.{op.name}"
    if op.kind == "range":
        return "Array.from"
    if op.kind == "index":
        return "[i]"
    if op.kind == "append":
        return "[...]"
    if op.kind == "opaque":
        return "opaque"
    if op.kind in ("method", "property"):
        return f"{interface(op.needs[0])}.{op.name}"
    if op.kind == "form":
        return f"{interface(op.needs[0])}.{op.name}"
    return op.name


def structure(source: str, entry: str) -> tuple:
    """(uses, root) of `entry` in verified source, by the compiler."""
    from research.v696.checker import checker
    read = checker().structure(source, entry)
    return frozenset(read["uses"]), read["root"]


# -- MultiPL-E's tests as pairs ---------------------------------------------

CALL = re.compile(r"assert\.(?:deepEqual|equal|strictEqual|deepStrictEqual)"
                  r"\(\s*candidate\(")


def _balanced(text: str, start: int) -> int:
    """The index of the `)` closing the call opened just before `start`."""
    depth, quote, at = 1, None, start
    while at < len(text):
        char = text[at]
        if quote:
            if char == "\\":
                at += 1
            elif char == quote:
                quote = None
        elif char in "\"'`":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return at
        at += 1
    return -1


def test_pairs(tests: str) -> list:
    """Each `assert.deepEqual(candidate(ARGS), EXPECTED)` in a MultiPL-E
    test file as (ARGS text, EXPECTED text)."""
    out = []
    for found in CALL.finditer(tests):
        close = _balanced(tests, found.end())
        if close < 0:
            continue
        # the assert's own `(`, and the `)` that closes it
        end = _balanced(tests, found.start() + found.group(0).index("(") + 1)
        rest = tests[close + 1:end].strip()
        if end < 0 or not rest.startswith(","):
            continue
        out.append((tests[found.end():close], rest[1:].strip()))
    return out


def values_of(pairs_text: list) -> list:
    """The texts of `test_pairs` as values, as Node makes them; pairs it
    cannot read are left out."""
    from research.v696.checker import checker
    if not pairs_text:
        return []
    texts = []
    for args, expected in pairs_text:
        texts += [f"[{args}]", expected]
    rows = checker().values([], [[]], texts)
    out = []
    for at in range(0, len(rows), 2):
        args, expected = rows[at][0], rows[at + 1][0]
        if "error" in args or "error" in expected:
            continue
        out.append((args["value"], expected["value"]))
    return out


# -- behaviour of a candidate beyond its examples ----------------------------

#: What an output is like on its own -- its kind of number, its case, that
#: it can be empty, that it has no repeats -- is true of the inputs a task
#: was tried on, not of the function: varied inputs break it for the right
#: program too. Only how the output stands to the inputs, and its order,
#: are the function's own, and only they are checked beyond the examples.
RANGE = frozenset({"a whole number", "a fraction", "zero or more",
                   "can be negative", "can be empty", "no repeats",
                   "upper case", "lower case", "true or false",
                   "the same output always"})


def checkable(predicate: str) -> bool:
    return predicate not in RANGE


def _variants(value) -> list:
    """Values near one an example gave: the same kind, the same domain
    as far as can be kept -- reordered, shortened, lengthened, nudged."""
    if isinstance(value, bool):
        return [not value]
    if _number(value):
        out = [value + 1, value - 1, value * 2]
        return [one for one in out if not isinstance(value, int)
                or one >= 0 or value < 0]
    if isinstance(value, str):
        out = [value[::-1], value[:-1], value + value[:1],
               value[1:] + value[:1]]
        return [one for one in out if one != value]
    if isinstance(value, list):
        out = [value[::-1], value[:-1], value + value[:1],
               value[1:] + value[:1]]
        if value and all(_number(one) for one in value):
            out.append([one + 1 for one in value])
        return [one for one in out if one != value]
    return []


def probes(examples: list, most: int = 16) -> list:
    """Fresh inputs for a spec: each example's arguments with one of them
    varied. A candidate's behaviour is read over these as well as the
    examples, so a program that only fits what it was shown shows it."""
    out, seen = [], {_key(list(args)) for args, _ in examples}
    for args, _ in examples:
        for at, value in enumerate(args):
            for other in _variants(value):
                fresh = list(args)
                fresh[at] = other
                if _key(fresh) not in seen:
                    seen.add(_key(fresh))
                    out.append(fresh)
    # spread over every example, not the first one's variants only
    return _spread(out, most)


def _spread(out: list, most: int) -> list:
    return out if len(out) <= most else [
        out[at * len(out) // most] for at in range(most)]


def _edges(value) -> list:
    """A value's edges, in its own domain: nothing, one, the least -- an
    empty list or string, one element, zero and one (and minus one where
    the example is negative)."""
    if isinstance(value, bool):
        return [not value]
    if _number(value):
        out = [0, 1] + ([-1] if value < 0 else [])
        return [one for one in out if one != value]
    if isinstance(value, str):
        return [one for one in ("", value[:1]) if one != value]
    if isinstance(value, list):
        return [one for one in ([], value[:1]) if one != value]
    return []


def _fresh(examples: list, seen: set, out: list, fresh: list) -> None:
    if _key(fresh) not in seen:
        seen.add(_key(fresh))
        out.append(fresh)


#: values of each type far from any example: what is varied from `day(0)` is
#: `day(1)`, and `day(7)`, `day(-1)` are where it may give nothing
WIDE = {"number": [0, 1, 5, -3, 100, -1, 7],
        "string": ["", "a", "hello world", "Abc", "  x  "],
        "boolean": [True, False],
        "number[]": [[], [1, 2, 3], [5, -1, 0], [7]],
        "string[]": [[], ["a", "b"], ["hello", "world"], [""]],
        "boolean[]": [[], [True, False]]}


def wide_probes(kinds: list) -> list:
    """Inputs of the parameters' types far from the examples, position by
    position (`WIDE`); none where a type is not one of them."""
    values = [WIDE.get(kind) for kind in kinds]
    if not kinds or any(one is None for one in values):
        return []
    return [[one[at % len(one)] for one in values]
            for at in range(max(len(one) for one in values))]


def edge_probes(examples: list, most: int = 12) -> list:
    """Inputs at their edges (`risk.py`, the S move: examples first): each
    example's arguments with one of them at an edge of its domain."""
    out, seen = [], {_key(list(args)) for args, _ in examples}
    for args, _ in examples:
        for at, value in enumerate(args):
            for other in _edges(value):
                fresh = list(args)
                fresh[at] = other
                _fresh(examples, seen, out, fresh)
    return _spread(out, most)


def pair_probes(examples: list, most: int = 12, edges: bool = False
                ) -> list:
    """Inputs with two arguments varied at once (the P move: a pairwise
    matrix) -- each at its edges too, with `edges` (P×S: a decision table).
    One argument varied alone does not show how two stand to each other."""
    out, seen = [], {_key(list(args)) for args, _ in examples}
    vary = _edges if edges else _variants
    for args, _ in examples:
        for one in range(len(args)):
            for two in range(one + 1, len(args)):
                for first in vary(args[one])[:2]:
                    for second in vary(args[two])[:2]:
                        fresh = list(args)
                        fresh[one], fresh[two] = first, second
                        _fresh(examples, seen, out, fresh)
    return _spread(out, most)
