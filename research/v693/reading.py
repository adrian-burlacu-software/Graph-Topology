"""A design goal read back from what the encoder said of each word.

The encoder reads *find a quadratic whose roots are 2 and minus 3 and whose
value at 0 is 12* word by word (`v692.reading`, act `design`): each word's
part -- `DKIND`, `CLAUSE`, `AT`, `IS`, `MULT`, or none -- and the symbols
or label it stands for (`stating.py` is what taught it). Nothing here looks
at a word. What is left is to **put the parts together**:

- a run of `AT`, `IS` or `MULT` words is one value, handed to sympy;
- a `CLAUSE` word's label opens a clause, and the values after it, up to
  the next clause, are its values -- except an `AT` or `MULT` right before
  a clause word, which is that clause's (*the 5th term*, *a double root*);
- each clause turns its values into specification clauses (`BUILD`): one
  root for each `IS`, a value for each `AT` paired with its `IS`, a term
  for each number a sequence *starts* with.

`spec_of` returns None when no kind was read: a design goal names what is
to be designed.
"""
from __future__ import annotations

import sympy as S

from research.v692.reading import _symbols
from research.v692.symbols import parsed
from research.v693.spec import C, Spec

ROLES = ("DKIND", "CLAUSE", "AT", "IS", "MULT")
#: A kind as its label names it, and the degree the word says, if any.
KINDS = {"polynomial": ("polynomial", None), "sequence": ("sequence", None),
         "function": ("function", None),
         "line": ("polynomial", 1), "quadratic": ("polynomial", 2),
         "cubic": ("polynomial", 3), "quartic": ("polynomial", 4)}


def _events(said, roles, labels) -> list:
    """(role, value, next to a clause) in order: a kind or clause label,
    or a value's symbols -- one for each run of words with the same part
    -- and whether the next word is a clause's: *5th* in *5th term*."""
    out = []
    run_role, run = None, []
    rows = list(zip(said, roles, labels))
    for index, (word, role, label) in enumerate(rows):
        if role in ("AT", "IS", "MULT"):
            if role != run_role and run:
                out.append((run_role, " ".join(run), False))
                run = []
            run_role = role
            symbols = _symbols(word, label)
            if symbols:
                run.append(symbols)
            following = rows[index + 1][1] if index + 1 < len(rows) else ""
            if following != role and run:
                out.append((run_role, " ".join(run),
                            following == "CLAUSE"))
                run, run_role = [], None
            continue
        if role in ("DKIND", "CLAUSE") and label not in ("DROP", "KEEP"):
            out.append((role, label, False))
    return out


def _value(symbols: str):
    symbols += " )" * max(0, symbols.count("(") - symbols.count(")"))
    return parsed(symbols)


def _pairs(items: list) -> list:
    """(at, is) pairs from a clause's values, whichever comes first: `value
    12 at 0` and `value at 0 is 12` are both (0, 12), and `at 3 and at 4
    is 8` is two."""
    out, waiting, last_is = [], [], None
    for role, value in items:
        if role == "AT":
            if last_is is not None and not waiting:
                out.append((value, last_is))
            else:
                waiting.append(value)
        elif role == "IS":
            if waiting:
                out += [(at, value) for at in waiting]
                waiting, last_is = [], None
            else:
                last_is = value
    return out


def _times(items) -> object:
    """How many times itself: the MULT before the clause, or once."""
    found = [value for role, value in items if role == "MULT"]
    return found[-1] if found else S.Integer(1)


def _roots(items):
    out, times = [], 1
    for role, value in items:
        if role == "MULT":
            times = int(value)
        elif role == "IS":
            out.append(C("root", value, times) if times != 1
                       else C("root", value))
            times = 1
    return out


BUILD = {
    "root": _roots,
    "value": lambda items: [C("value", at, is_) for at, is_ in
                            _pairs(items)],
    "slope": lambda items: [C("slope", at, is_) for at, is_ in
                            _pairs(items)],
    "term": lambda items: [C("term", int(at), is_) for at, is_ in
                           _pairs(items)],
    "stationary": lambda items: [C("stationary", value)
                                 for role, value in items if role in
                                 ("AT", "IS")],
    "leading": lambda items: [C("leading", value) for role, value in items
                              if role == "IS"][:1],
    "monic": lambda items: [C("leading", 1)],
    "degree": lambda items: [C("degree", int(value)) for role, value in
                             items if role == "IS"][:1],
    "integer": lambda items: [C("integer")],
    "total": lambda items: [C("total", int(at), is_) for at, is_ in
                            _pairs(items)],
    "start": lambda items: [C("term", place + 1, value) for place, value in
                            enumerate(value for role, value in items
                                      if role == "IS")],
    "derivative": lambda items: [C("derivative", value)
                                 for role, value in items if role == "IS"][:1],
    # `whose derivative is 3 times itself`: f' = 3f; the second, f'' = 3f
    "rate": lambda items: [C("ode", 0, 1, -_times(items))],
    "second-rate": lambda items: [C("ode", 1, 0, -_times(items))],
    "through": lambda items: [C("value", point[0], point[1])
                              for role, point in items if role == "IS"
                              and isinstance(point, S.Tuple)
                              and len(point) == 2],
}


def spec_of(said, roles, labels) -> Spec | None:
    """The specification the parts say, or None where no kind was read.
    Raises `symbols.Unreadable` where a value cannot be read."""
    events = _events(said, roles, labels)
    kind, degree = None, None
    clauses: list = []
    pending: list = []
    for role, value, beside in events:
        if role == "DKIND":
            if value in KINDS:
                kind, degree = KINDS[value]
            continue
        if role == "CLAUSE":
            clauses.append([value, pending])
            pending = []
            continue
        item = (role, _value(value))
        if role in ("AT", "MULT") and beside:
            # *the 5th term*, *a double root*: the clause after it.
            pending.append(item)
        elif clauses:
            clauses[-1][1].append(item)
        else:
            pending.append(item)
    if kind is None:
        return None
    out = []
    for name, items in clauses:
        make = BUILD.get(name)
        if make is not None:
            out += make(items)
    if degree is not None and not any(one.kind == "degree" for one in out):
        out.append(C("degree", degree))
    return Spec(kind, out)
