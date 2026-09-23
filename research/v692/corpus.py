"""The records that teach the encoder mathematics (`task: math`).

    python -m research.v692.corpus                 # llm/math-data
    python -m research.v692.corpus --count 80000

Each record is one utterance from a template of the curriculum
(`curriculum.ACTS`), its slots filled with objects drawn for the act and said
by `saying.py`. Three things are taught about it:

    math_act      what it asks: value, solve, factor, is kind, ... or none
    math_role     each word's part: EXPR, VAR, A, B, LIST, KIND, OTHER, O
    math_symbol   each word's symbols: `^2`, `5`, `sqrt(`, KEEP, DROP

**A record is kept only when it reads back.** Every part's symbols are
handed to sympy (`symbols.parsed`) and must come back as the object that
filled the slot -- the round trip every other corpus here is held to, and
here it is exact.

**And the encoder is taught when it is not its turn.** A tenth of the
records are utterances with no mathematics to do -- the reader's own
corpus, and the near misses that belong to another layer: *what is a prime
number* is a definition (v687's R26), *mary has 3 apples* a count in a
world (v691), *is a square a rectangle* the taxonomy. Their act is `none`.

The records live in their own folder, `llm/math-data`, which nothing else
writes to; `teach_reader` reads them beside its own corpus.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

import sympy as S

from research.v692 import curriculum as C, measuring as M
from research.v692.saying import DROP, KEEP, Said, Speaker, ordinal
from research.v692.symbols import Unreadable, parsed, same

LLM = Path(__file__).resolve().parents[2] / "llm"
OUT = LLM / "math-data"

#: Each slot's part in the utterance.
ROLES = {"E": "EXPR", "EQ": "EXPR", "R": "EXPR", "M": "EXPR", "S": "EXPR",
         "V": "VAR", "F": "VAR", "A": "A", "N": "A", "B": "B", "L": "LIST",
         "K": "KIND", "T": "OTHER", "U": "OTHER", "H": "SHAPE",
         "W": "WANTED"}
#: A measure's parts besides: what each given is (`NAME`, its label the
#: quantity) and how much (`VALUE`), what is wanted, and of what shape.
ROLE_LABELS = ("O", "EXPR", "VAR", "A", "B", "LIST", "KIND", "OTHER",
               "NAME", "VALUE", "WANTED", "SHAPE")
NONE = "none"

#: How a sentence is split into words, here and at run time
#: (`reading.words`): punctuation that parts words is a word of its own.
PARTING = re.compile(r"\s*([,?:;])\s*")


def words(text: str) -> list[str]:
    """The words of a sentence: split on spaces, and on punctuation that
    parts words -- but not inside brackets, where `[[1,2],[3,4]]` and
    `{1,2,3}` are one written thing."""
    plain, depth = [], 0
    for char in (text or "").strip().lower():
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if depth == 0 and char in ",?:;%":
            plain.append(f" {char} ")
        else:
            plain.append(char)
    out = []
    for one in "".join(plain).split():
        # A full stop that ends a sentence is a word of its own, as it is
        # in what was taught; the one inside `3.14` is not. Without this a
        # reply the decoder wrote as `That's 166.03.` hands sympy
        # `166.03.`, which is nothing.
        stops = 0
        while len(one) > 1 and one.endswith("."):
            one, stops = one[:-1], stops + 1
        if one not in ("?",):
            out.append(one)
        out += ["."] * stops
    return out


# -- filling slots ------------------------------------------------------------

def _function(rng, var):
    choice = rng.random()
    if choice < 0.6:
        return C.polynomial(rng, var)
    base = rng.choice((S.sin, S.cos, S.exp, S.log))
    inner = var if rng.random() < 0.7 else rng.randint(2, 5) * var
    return base(inner) * (rng.choice((1, 1, 2, 3)))


def _rational_function(rng, var):
    """`(x^2 - x - 6) / (x - 3)`: a limit worth asking, the removable point
    kept -- sympy would cancel it on construction."""
    root = rng.randint(-5, 5)
    other = rng.randint(-5, 5)
    return S.Mul(S.expand((var - root) * (var - other)),
                 S.Pow(var - root, -1), evaluate=False)


def fill(act: str, slot: str, rng, filled: dict):
    """What fills a slot of an act, consistent with what was filled before:
    the variable a derivative is taken in is the one its expression has."""
    var = filled.get("V")
    if slot == "V" and act == "let":
        return C.variable(rng)
    if slot == "F":
        return S.Function(rng.choice(("f", "f", "g", "h")))(
            rng.choice((C.x, C.x, C.t, C.n)))
    if act == "series" and slot == "V":
        return filled["E"].free_symbols.pop()
    if slot == "V":
        expr = filled.get("E") or filled.get("EQ")
        free = sorted(getattr(expr, "free_symbols", ()), key=str)
        return free[0] if free else C.variable(rng)
    if act == "part of":
        if slot == "E":
            bottom = rng.choice((2, 3, 4, 5, 6, 8, 10))
            return S.Rational(rng.randint(1, bottom - 1), bottom)
        # An amount the fraction goes into, mostly whole.
        bottom = int(S.fraction(filled["E"])[1])
        return S.Integer(bottom * rng.randint(1, 30) if rng.random() < 0.8
                         else rng.randint(2, 200))
    if act == "value":
        choice = rng.random()
        if choice < 0.06:
            return C.variable(rng)
        if choice < 0.14:
            # An expression in a variable: `what is a squared plus 1`.
            return C.polynomial(rng, C.variable(rng),
                                rng.choice((1, 2, 2, 3)))
        if choice < 0.1:
            return S.Function(rng.choice(("f", "g")))(C.integer(rng, -5, 10))
        if choice < 0.16:
            return S.Mul(C.complex_number(rng), C.complex_number(rng),
                         evaluate=False) if rng.random() < 0.5 else S.Add(
                C.complex_number(rng), C.complex_number(rng), evaluate=False)
        if choice < 0.22:
            return S.Add(C.decimal(rng, 2), C.decimal(rng, 1),
                         evaluate=False)
        return C.BY_ACT["value"].fills["E"](rng)
    if act == "simplify":
        var = C.variable(rng)
        return S.Add(C.polynomial(rng, var), C.polynomial(rng, var),
                     evaluate=False)
    if act == "expand":
        return C.factored(rng)
    if act == "factor":
        if rng.random() < 0.2:
            return C.natural(rng, 500)
        return S.expand(C.factored(rng))
    if act == "solve":
        if rng.random() < 0.25:
            return C.inequality(rng, var)
        return C.equation(rng, var)
    if act == "solve system":
        filled.setdefault("__system", C.system(rng))
        return filled["__system"][0 if slot == "EQ" else 1]
    if act == "convert":
        if slot == "A":
            return C.amount(rng)
        return C.unit_like(rng, filled["A"])
    if act in ("next term", "nth term", "term"):
        if slot == "L":
            return C.sequence(rng)
        return S.Integer(rng.randint(5, 15))
    if act == "series":
        var = rng.choice((C.n, C.n, S.Symbol("k"), C.x))
        if slot == "A":
            return S.Integer(rng.choice((0, 1, 1, 1)))
        if slot == "B":
            return S.oo if rng.random() < 0.3 else S.Integer(
                rng.randint(4, 20))
        if filled.get("B") is S.oo and filled.get("A") == 0:
            return rng.choice((S.Rational(1, 2) ** var,
                               S.Rational(1, 3) ** var))
        if filled.get("B") is S.oo:
            return rng.choice((1 / var ** 2, S.Rational(1, 2) ** var,
                               1 / (var * (var + 1)), S.Rational(1, 3) ** var))
        return C.summand(rng, var)
    if act == "equivalent":
        if slot == "E":
            return C.proposition(rng)
        other = filled["E"]
        return rng.choice((S.to_dnf(other), S.to_cnf(other),
                           S.simplify_logic(other), C.proposition(rng)))
    if act == "dice":
        if slot == "A":
            return S.Integer(rng.choice((1, 2, 2, 3)))
        dice = int(filled.get("A", 1))
        return S.Integer(rng.randint(dice, 6 * dice))
    if act == "coins":
        if slot == "A":
            return S.Integer(rng.randint(1, 6))
        return S.Integer(rng.randint(0, int(filled["A"])))
    if act == "round places":
        if slot == "A":
            return C.decimal(rng, rng.choice((3, 4, 5)))
        return S.Integer(rng.choice((1, 2, 2, 3)))
    if act == "round nearest":
        if slot == "A":
            return (C.decimal(rng) if rng.random() < 0.5
                    else S.Integer(rng.randint(11, 99999)))
        return S.Integer(rng.choice((10, 100, 1000)))
    if act == "percent of":
        if slot == "A":
            return S.Integer(rng.choice((5, 10, 12, 15, 20, 25, 30, 40, 50,
                                         60, 75, 80, 90, rng.randint(1, 100))))
        return S.Integer(rng.choice((10, 20, 40, 50, 60, 80, 120, 150, 200,
                                     240, 300, 500, 1000,
                                     rng.randint(10, 999))))
    if act == "as percent":
        if slot == "A":
            return S.Integer(rng.randint(1, 80))
        return S.Integer(int(filled["A"]) + rng.randint(1, 200))
    if act == "percent change":
        return S.Integer(rng.randint(10, 500))
    if act in ("dot", "cross", "magnitude"):
        size = 3 if act == "cross" else (len(filled["S"]) if "S" in filled
                                         else None)
        return C.vector(rng, size)
    if act in ("modulus", "conjugate"):
        return C.complex_number(rng)
    if act in ("derivative", "integral"):
        if slot == "E":
            return _function(rng, var or C.variable(rng))
        low = rng.randint(-3, 3)
        return S.Integer(low if slot == "A" else low + rng.randint(1, 5))
    if act == "limit":
        if slot == "E":
            return _rational_function(rng, var or C.variable(rng))
        return S.Integer(rng.randint(-5, 5))
    if act == "at":
        if slot == "E":
            return C.polynomial(rng, var or C.variable(rng))
        return S.Integer(rng.randint(-6, 10))
    if act in ("gcd", "lcm"):
        base = rng.choice((2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15))
        return [S.Integer(base * rng.randint(1, 12))
                for _ in range(rng.choice((2, 2, 2, 3)))]
    if act in ("divisors", "prime factors"):
        return C.natural(rng, 360)
    if act == "is kind":
        if slot == "K":
            return filled["__kind"]
        kind = filled["__kind"]
        if kind.of == "proposition":
            return C.proposition(rng)
        if kind.of == "sequence":
            return C.sequence(rng)
        if kind.of == "expression":
            return C.polynomial(rng) if rng.random() < 0.7 else S.sin(
                C.variable(rng))
        if kind.of == "equation":
            return C.equation(rng)
        return C.number_like(rng)
    if act == "divides":
        if slot == "B":
            return S.Integer(rng.randint(2, 12))
        return C.natural(rng, 300)
    if act == "check":
        left = C.arithmetic(rng)
        value = S.sympify(left)
        right = value if rng.random() < 0.5 else value + rng.choice(
            (-2, -1, 1, 2, 10))
        relation = rng.choice((S.Eq, S.Eq, S.Gt, S.Lt))
        return relation(left, right, evaluate=False)
    if act == "mode":
        # A list with a mode: one value said more often than the rest.
        items = C.numbers(rng)
        items += [rng.choice(items)] * rng.choice((1, 1, 2))
        rng.shuffle(items)
        return items
    if act in ("mean", "median", "range", "sum"):
        return C.numbers(rng)
    if act == "choose":
        if slot == "A":
            return S.Integer(rng.randint(3, 20))
        return S.Integer(rng.randint(1, max(1, int(filled["A"]) - 1)))
    if act == "arrange":
        return S.Integer(rng.randint(2, 10))
    if act in ("determinant", "inverse", "transpose"):
        return C.matrix(rng)
    if act in ("union", "intersection", "subset"):
        return C.finite_set(rng)
    if act == "let":
        if slot == "A":
            return C.number_like(rng)
        if "F" in filled:
            # `let f(x) = x^2 + 1`: the body is in the function's variable.
            return C.polynomial(rng, filled["F"].args[0])
        # What a variable is let be is said in others, never itself.
        others = [one for one in C.VARIABLES[:4] if one != filled.get("V")]
        return C.polynomial(rng, rng.choice(others), 1)
    raise KeyError((act, slot))


def said_list(items, speaker: Speaker) -> Said:
    """`12 , 18 and 24`: items with what parts them."""
    out = Said()
    for index, item in enumerate(items):
        if index:
            last = index == len(items) - 1
            joiner = speaker.rng.choice(("and", ",")) if last else ","
            out.add(joiner, ",")
        out.then(speaker.say(item))
    return out


def said_kind(kind, template: str, rng) -> Said:
    """A kind as the curriculum says it -- `prime`, `a prime number` --
    each word a part of the kind, standing for no symbols."""
    phrase = rng.choice(kind.said)
    if "a {K}" in template:
        # `is 28 a {K}`: the template has the article.
        phrase = rng.choice([one.split(" ", 1)[1] if one.split()[0] in (
            "a", "an", "the") else one for one in kind.said])
    out = Said()
    for word in phrase.split():
        out.add(word, DROP)
    return out


#: Slots filled before the others, where what fills one depends on it:
#: the amount a fraction is taken of is drawn to suit the fraction.
FIRST = {"part of": ("E",)}
#: What each slot of `is kind` asks about.
KIND_SLOTS = {"A": ("number",), "E": ("expression", "equation",
                                      "proposition"), "L": ("sequence",)}


def record(act, template: str, rng) -> dict | None:
    if act.name in ("measure", "given"):
        return measure_record(act, template, rng)
    speaker = Speaker(rng)
    filled: dict = {}
    if act.name == "is kind":
        slot = re.search(r"\{([AEL])\}", template).group(1)
        kinds = [one for one in C.KINDS if one.of in KIND_SLOTS[slot]]
        filled["__kind"] = rng.choice(kinds)
    slots = re.findall(r"\{(\w+)\}", template)
    # The variable is filled after the expression it is the variable of
    # (before it, for `let`, and a function before its body), and A
    # before B.
    for slot in sorted(set(slots), key=lambda one: (
            (one not in ("V", "F")) if act.name == "let"
            else (one not in FIRST.get(act.name, ())) if act.name in FIRST
            else (one == "V"), one)):
        filled[slot] = fill(act.name, slot, rng, filled)
    out_words, out_roles, out_labels = [], [], []
    parts: dict = collections.defaultdict(list)
    for piece in re.split(r"(\{\w+\})", template):
        slot = re.fullmatch(r"\{(\w+)\}", piece)
        if slot is None:
            for word in words(piece):
                out_words.append(word)
                out_roles.append("O")
                out_labels.append(DROP)
            continue
        name = slot.group(1)
        value = filled[name]
        if name == "K":
            said = said_kind(value, template, rng)
        elif name == "N":
            said = ordinal(int(value), rng)
        elif isinstance(value, list):
            said = said_list(value, speaker)
        else:
            before = re.split(r"\{\w+\}", template)[
                re.findall(r"\{\w+\}", template).index(piece)]
            speaker.relating = (
                "after is" if before.strip().endswith("is")
                else "free" if template.strip() == piece
                else "embedded")
            said = speaker.say(value)
        role = ROLES[name]
        # A said part split into this corpus's words (`words`): a comma is
        # its own word.
        for word, label in zip(said.words, said.labels):
            for one in words(word):
                out_words.append(one)
                out_roles.append(role)
                out_labels.append(label if one == words(word)[-1] else
                                  DROP)
        parts[role].append((name, value))
    if len(out_words) > 60:
        return None
    # The round trip: every part's symbols read back as what filled it.
    for role, filled_parts in parts.items():
        if role == "KIND":
            continue
        symbols = " ".join(
            (word if label == KEEP else
             word + label[len(KEEP):] if label.startswith(KEEP + " ")
             else label)
            for word, part, label in zip(out_words, out_roles, out_labels)
            if part == role and label != DROP)
        wanted = filled_parts[0][1]
        try:
            back = parsed(symbols)
        except Unreadable:
            return None
        if isinstance(wanted, list):
            wanted = S.Tuple(*wanted)
            back = back if isinstance(back, S.Tuple) else S.Tuple(back)
        if not same(back, wanted):
            return None
    return {"task": "math", "words": out_words, "act": act.name,
            "roles": out_roles, "symbols": out_labels,
            "tags": [""] * len(out_words), "deps": [""] * len(out_words),
            "names": [], "said": " ".join(out_words),
            # What filled each part, for `speaking.py`; never written out.
            "_parts": parts_of(filled)}


def parts_of(filled: dict) -> dict:
    """What filled each slot, by the part it is read as: what `doing.do`
    is handed."""
    parts = {}
    for slot, value in filled.items():
        if slot == "__kind":
            parts["KIND"] = value
        elif slot.startswith("__") or slot == "K":
            continue
        elif isinstance(value, list):
            parts["LIST"] = S.Tuple(*value)
        else:
            parts[ROLES[slot]] = value
    return parts


# -- measures: shapes, what is given, what is wanted -----------------------

#: How each shape is said; the label is the shape, on the last word.
SHAPE_WORDS = {"right triangle": ("right triangle", "right-angled triangle",
                                  "right angled triangle"),
               "sphere": ("sphere", "ball"), "cube": ("cube",),
               "trapezoid": ("trapezoid", "trapezium")}
#: How each quantity is said, as given or as wanted.
QUANTITY_WORDS = {
    "side": ("side", "side length", "length of a side", "edge"),
    "surface-area": ("surface area",), "angle-sum": (
        "sum of the interior angles", "angle sum",
        "sum of the angles", "total of the interior angles"),
    "interior-angle": ("interior angle", "each interior angle",
                       "each angle", "size of each interior angle"),
    "exterior-angle": ("exterior angle", "each exterior angle"),
    "slant": ("slant height",), "leg2": ("other leg",),
    "base2": ("other base", "second base"), "diagonal": ("diagonal",),
    "hypotenuse": ("hypotenuse", "longest side")}
TRIPLES = ((3, 4), (5, 12), (6, 8), (8, 15), (9, 12), (7, 24), (20, 21))


def _phrase(words: str, label: str, role: str) -> tuple:
    """A phrase of one part: its words, the label on the last."""
    split = words.split()
    return ([(word, role, label if index == len(split) - 1 else DROP)
             for index, word in enumerate(split)])


def _shape_said(shape: str, rng) -> list:
    phrase = rng.choice(SHAPE_WORDS.get(shape, (shape,)))
    return _phrase(phrase, shape.replace(" ", "-"), "SHAPE")


def _quantity_said(name: str, rng, role: str) -> list:
    from research.v692 import measuring as M
    phrase = rng.choice(QUANTITY_WORDS.get(name, (M.said(name),)))
    return _phrase(phrase, name, role)


def _value_said(value, speaker: Speaker) -> list:
    said = speaker.say(value)
    out = []
    for word, label in zip(said.words, said.labels):
        pieces = words(word)
        for one in pieces:
            out.append((one, "VALUE", label if one == pieces[-1] else DROP))
    return out


def _worked(shape: str, rng) -> dict:
    """Every measure of a shape, from base values drawn for it."""
    if shape in M.BASES:
        names = M.BASES[shape]
        if shape in ("right triangle", "cone") or (
                shape == "rectangle" and rng.random() < 0.3):
            pair = rng.choice(TRIPLES)
            scale = rng.choice((1, 1, 2))
            values = [S.Integer(one * scale) for one in pair]
            values += [S.Integer(rng.randint(2, 12))
                       for _ in names[len(values):]]
        else:
            values = [S.Integer(rng.randint(1, 20)) for _ in names]
        base = dict(zip(names, values))
    else:
        base = {"side": S.Integer(rng.randint(1, 20))}
    out = dict(base)
    for name in M.quantities(shape):
        if name not in out:
            found = M.measure(shape, name, base)
            if found.value is not None:
                out[name] = found.value
    return out


def _nice(value) -> bool:
    """A value someone would give: short, and without roots."""
    return value.is_number and not value.has(S.Pow) or (
        value.is_number and value.is_Rational)


def _givens(shape: str, wanted: str, values: dict, rng,
            complete: bool) -> dict:
    """Givens a question could state: enough to work the wanted measure
    out from (`complete`), or one short of that."""
    from research.v692 import measuring as M
    names = [one for one in values if one not in (wanted, "sides")
             and _nice(values[one])]
    size = len(M.BASES.get(shape, ("side",)))
    for _ in range(12):
        rng.shuffle(names)
        chosen = names[:size]
        if M._plan(shape, set(chosen) | ({"sides"} if M.sides(shape)
                                         else set()), wanted):
            if complete:
                return {one: values[one] for one in chosen}
            if len(chosen) > 1:
                return {one: values[one] for one in chosen[:-1]}
    return {}


def measure_record(act, template: str, rng) -> dict | None:
    """A question about a shape's measure, or a given said on its own
    (`the width is 6`), each word with its part."""
    from research.v692 import measuring as M
    speaker = Speaker(rng, "spoken" if rng.random() < 0.8 else "written")
    polygons = [one for one in M.POLYGONS if M.sides(one)]
    shape = rng.choice(sorted(M.SHAPES) + polygons
                       if "{G}" in template else polygons)
    if "regular" in template:
        shape = rng.choice(polygons)
    values = _worked(shape, rng)
    wanted = rng.choice([one for one in values if one != "sides"])
    if "{G}" not in template:
        wanted = rng.choice(("angle-sum", "interior-angle",
                             "exterior-angle"))
    givens = {}
    if "{G}" in template:
        if act.name == "given":
            name = rng.choice([one for one in values if one != "sides"
                               and _nice(values[one])] or [None])
            if name is None:
                return None
            givens = {name: values[name]}
        else:
            givens = _givens(shape, wanted, values, rng,
                             complete=rng.random() < 0.85)
        if not givens:
            return None
    if givens and rng.random() < 0.3:
        # In a unit: lengths in it, areas and volumes in its powers.
        from research.v692.curriculum import UNIT_KINDS, unit
        chosen = unit(rng.choice(UNIT_KINDS["length"][:4]))
        givens = {name: value * chosen ** M.DIMENSIONS[name]
                  if isinstance(M.DIMENSIONS[name], int) and
                  M.DIMENSIONS[name] else value
                  for name, value in givens.items()}
    out: list = []
    whose = "whose {G}" in template or (act.name == "given"
                                        and "has" not in template)
    for piece in re.split(r"(\{\w+\})", template):
        if piece == "{H}":
            out += _shape_said(shape, rng)
        elif piece == "{W}":
            out += _quantity_said(wanted, rng, "WANTED")
        elif piece == "{G}" and {"leg", "leg2"} <= set(givens) and                 rng.random() < 0.5:
            # `legs 6 and 8`, `whose legs are 6 and 8`: one name, two values
            out += _phrase("legs", "leg", "NAME")
            if whose:
                out.append(("are", "O", DROP))
            out += _value_said(givens["leg"], speaker)
            out.append(("and", "O", DROP))
            out += _value_said(givens["leg2"], speaker)
            givens = {"leg": givens["leg"], "leg2": givens["leg2"]}
        elif piece == "{G}":
            items = list(givens.items())
            for index, (name, value) in enumerate(items):
                if index:
                    out.append(("and" if index == len(items) - 1
                                else ",", "O", DROP))
                form = rng.random()
                if whose:
                    out += _quantity_said(name, rng, "NAME")
                    out.append(("is", "O", DROP))
                elif form < 0.4:
                    out.append(("a", "O", DROP))
                    out += _quantity_said(name, rng, "NAME")
                    out.append(("of", "O", DROP))
                else:
                    out += _quantity_said(name, rng, "NAME")
                out += _value_said(value, speaker)
        else:
            out += [(word, "O", DROP) for word in words(piece)]
    if len(out) > 60:
        return None
    # The round trip: each value's symbols read back as the value.
    runs, current = [], []
    for word, role, label in out:
        if role == "VALUE":
            current.append((word, label))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    for run, wanted_value in zip(runs, givens.values()):
        symbols = " ".join(word if label == KEEP else
                           word + label[len(KEEP):]
                           if label.startswith(KEEP + " ") else label
                           for word, label in run if label != DROP)
        try:
            if not same(parsed(symbols), wanted_value):
                return None
        except Unreadable:
            return None
    if len(runs) != len(givens):
        return None
    said = [word for word, _, _ in out]
    parts = {"SHAPE": shape, "GIVENS": givens}
    if "{W}" in template:
        parts["WANTED"] = wanted
    return {"task": "math", "words": said, "act": act.name,
            "roles": [role for _, role, _ in out],
            "symbols": [label for _, _, label in out],
            "tags": [""] * len(said), "deps": [""] * len(said),
            "names": [], "said": " ".join(said), "_parts": parts}


#: Near misses: mathematics' words, and another layer's question.
NEAR = (
    "what is a prime number", "what is an integer", "what is a triangle",
    "define a polynomial", "what does divisible mean", "what is algebra",
    "what is a square", "is a square a rectangle",
    "is a triangle a polygon", "is a circle a shape",
    "how many legs does a spider have", "how many sides does a hexagon have",
    "mary has 3 apples", "i gave 2 apples to mary",
    "how many apples does mary have", "give mary 2 apples",
    "there are 12 eggs in the basket", "john ate 2 cookies",
    "how many kinds of dog are there", "what is a number",
    "can a number be negative", "who invented calculus",
    "why is maths hard", "i like maths", "what is the capital of france",
    "is a whale a mammal", "what is a dog", "can a pig fly",
    "the book is in the kitchen", "what time is it", "hello",
    "what is your name", "thank you", "tell me about prime numbers",
    "what are numbers used for", "mary is 5 years old",
    "the train leaves at 7", "i have two brothers",
    # the later branches' words, in another layer's questions
    "what is a hexagon", "what is a tautology", "what is a vector",
    "what is probability", "what is a sequence", "is a cube a solid",
    "i rolled a die yesterday", "the recipe needs 2 cups of flour",
    "i walked 5 km this morning", "the sequence of events was strange",
    "it is a complex problem", "the circle of friends met at 6",
    "a mosquito is a vector of disease", "what is the area of expertise",
    "the room is square", "she flipped a coin", "he is 6 feet tall",
    "the box weighs 3 kilograms", "what does percent mean",
    "who invented the metric system", "is a square a polygon",
    "the price went up 10 percent", "the pentagon is in washington",
)


def negatives(count: int, rng) -> list[dict]:
    """Utterances with no mathematics to do: the reader's own corpus, and
    the near misses above."""
    out = []
    pool = []
    path = LLM / "reader-data" / "train.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index % 13:
                    continue
                one = json.loads(line)
                if one.get("task", "read") in ("read", "ask"):
                    pool.append(" ".join(one["words"]))
    pool = sorted(set(pool))
    texts = [rng.choice(NEAR) if rng.random() < 0.35 or not pool
             else rng.choice(pool) for _ in range(count)]
    for text in texts:
        said = words(text)
        if not said:
            continue
        out.append({"task": "math", "words": said, "act": NONE,
                    "roles": ["O"] * len(said),
                    "symbols": [DROP] * len(said),
                    "tags": [""] * len(said), "deps": [""] * len(said),
                    "names": [], "said": " ".join(said)})
    return out


def build(count: int, seed: int = 692, out: Path = OUT) -> dict:
    rng = random.Random(seed)
    records, tried = [], collections.Counter()
    kept = collections.Counter()
    per_act = count * 7 // 10 // len(C.ACTS)
    for act in C.ACTS:
        attempts = 0
        while kept[act.name] < per_act and attempts < per_act * 6:
            attempts += 1
            template = rng.choice(act.templates)
            tried[act.name] += 1
            try:
                one = record(act, template, rng)
            except (ValueError, TypeError, ZeroDivisionError,
                    RecursionError, KeyError, AttributeError):
                one = None
            if one is None:
                continue
            one.pop("_parts", None)
            one["source"] = f"math-{act.name}-{template}"
            one["how"] = "math"
            records.append(one)
            kept[act.name] += 1
    records += negatives(count - len(records), rng)
    rng.shuffle(records)
    out.mkdir(parents=True, exist_ok=True)
    train = valid = 0
    with open(out / "train-math.jsonl", "w", encoding="utf-8") as trained, \
            open(out / "valid-math.jsonl", "w", encoding="utf-8") as held:
        for index, one in enumerate(records):
            if index % 10 == 0:
                held.write(json.dumps(one) + "\n")
                valid += 1
            else:
                trained.write(json.dumps(one) + "\n")
                train += 1
    stats = {"train": train, "valid": valid,
             "acts": dict(collections.Counter(one["act"] for one in records)),
             "kept of tried": {name: f"{kept[name]}/{tried[name]}"
                               for name in tried}}
    (out / "stats.json").write_text(json.dumps(stats, indent=1),
                                    encoding="utf-8")
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=692)
    parser.add_argument("--out", default=str(OUT))
    options = parser.parse_args(argv)
    stats = build(options.count, options.seed, Path(options.out))
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
