"""Shapes and their measures, worked out by planning: formulas as actions.

*What is the perimeter of a square with area 49?* Nothing here knows that
answer, or the way to it. What is known is a textbook's formulas for each
shape (`SHAPES`), and each formula is an **action**: from the quantities it
relates, all but one known, the last becomes known. `area = side^2` is two
actions -- *side from area* and *area from side* -- and `perimeter = 4
side` two more. The question is a problem for v691's agent (`acting.agent`):
the start is what was given, the goal is knowing what was asked, and the
plan it finds is the derivation -- *side from area, then perimeter from
side*. Doing each step is sympy's (`Sheet.do`): the formula solved for the
unknown, the knowns put in, units and all.

**What cannot be planned is asked for.** *The area of a rectangle with
length 4* has no plan: every way to the area needs the width, or something
that gives it. Which quantities would give one is found the same way -- the
problem planned again with each unknown quantity supposed known -- and the
answer names them: *I need its width (or its perimeter, or its diagonal)*.
The page keeps the question open, so *the width is 6* finishes it
(`page.py`).

**How many sides a polygon has is read from WordNet**, not written here:
*hexagon* is `a six-sided polygon`, *decagon* `a polygon with 10 sides and
10 angles` (`sides`). That is what lets the angles of any polygon WordNet
names be worked out.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field

import sympy as S

from research.v691 import acting, world as W

#: Each quantity's dimension: a length, an area, a volume, an angle (in
#: degrees), or a count.
DIMENSIONS = {
    "radius": 1, "diameter": 1, "circumference": 1, "perimeter": 1,
    "side": 1, "length": 1, "width": 1, "height": 1, "base": 1, "base2": 1,
    "leg": 1, "leg2": 1, "hypotenuse": 1, "diagonal": 1, "slant": 1,
    "area": 2, "surface-area": 2, "volume": 3, "angle-sum": "degrees",
    "interior-angle": "degrees", "exterior-angle": "degrees", "sides": 0}

_Q = {name: S.Symbol(name.replace("-", "_"), positive=True)
      for name in DIMENSIONS}


def q(name: str) -> S.Symbol:
    return _Q[name]


def _name(symbol) -> str:
    return str(symbol).replace("_", "-")


def _f(left: str, right) -> S.Equality:
    return S.Eq(q(left), right)


_POLYGON = (
    _f("angle-sum", (q("sides") - 2) * 180),
    _f("interior-angle", q("angle-sum") / q("sides")),
    _f("exterior-angle", S.Integer(360) / q("sides")),
    _f("perimeter", q("sides") * q("side")),
)

#: Each shape's formulas, as a textbook states them.
SHAPES = {
    "circle": (_f("diameter", 2 * q("radius")),
               _f("circumference", 2 * S.pi * q("radius")),
               _f("area", S.pi * q("radius") ** 2)),
    "square": (_f("perimeter", 4 * q("side")), _f("area", q("side") ** 2),
               _f("diagonal", S.sqrt(2) * q("side"))),
    "rectangle": (_f("perimeter", 2 * (q("length") + q("width"))),
                  _f("area", q("length") * q("width")),
                  _f("diagonal", S.sqrt(q("length") ** 2
                                        + q("width") ** 2))),
    "triangle": (_f("area", q("base") * q("height") / 2),),
    "right triangle": (_f("hypotenuse", S.sqrt(q("leg") ** 2
                                                + q("leg2") ** 2)),
                       _f("area", q("leg") * q("leg2") / 2),
                       _f("perimeter", q("leg") + q("leg2")
                          + q("hypotenuse"))),
    "parallelogram": (_f("area", q("base") * q("height")),),
    "trapezoid": (_f("area", (q("base") + q("base2")) * q("height") / 2),),
    "cube": (_f("volume", q("side") ** 3),
             _f("surface-area", 6 * q("side") ** 2),
             _f("diagonal", S.sqrt(3) * q("side"))),
    "sphere": (_f("diameter", 2 * q("radius")),
               _f("volume", S.Rational(4, 3) * S.pi * q("radius") ** 3),
               _f("surface-area", 4 * S.pi * q("radius") ** 2)),
    "cylinder": (_f("diameter", 2 * q("radius")),
                 _f("volume", S.pi * q("radius") ** 2 * q("height")),
                 _f("surface-area", 2 * S.pi * q("radius") ** 2
                    + 2 * S.pi * q("radius") * q("height"))),
    "cone": (_f("diameter", 2 * q("radius")),
             _f("volume", S.pi * q("radius") ** 2 * q("height") / 3),
             _f("slant", S.sqrt(q("radius") ** 2 + q("height") ** 2))),
}
#: What each shape's measures are worked out from: what is asked for
#: first when not enough is given.
BASES = {"circle": ("radius",), "square": ("side",),
         "rectangle": ("length", "width"), "triangle": ("base", "height"),
         "right triangle": ("leg", "leg2"),
         "parallelogram": ("base", "height"),
         "trapezoid": ("base", "base2", "height"), "cube": ("side",),
         "sphere": ("radius",), "cylinder": ("radius", "height"),
         "cone": ("radius", "height")}
#: Polygons WordNet names, whose sides it says (`sides`).
POLYGONS = ("triangle", "quadrilateral", "pentagon", "hexagon", "heptagon",
            "octagon", "nonagon", "decagon", "dodecagon")

_NUMBER_WORDS = {"three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                 "twelve": 12}


@functools.lru_cache(maxsize=None)
def sides(shape: str) -> int | None:
    """How many sides WordNet says a polygon has: `a six-sided polygon`,
    `a polygon with 10 sides`. None for what WordNet does not call a
    polygon."""
    try:
        from nltk.corpus import wordnet
        polygon = wordnet.synset("polygon.n.01")
        found = wordnet.synsets(shape.replace(" ", "_"), "n")
    except Exception:                                # noqa: BLE001
        return None
    for synset in found:
        if polygon not in synset.closure(lambda one: one.hypernyms()):
            continue
        gloss = synset.definition()
        named = re.search(r"\b(\w+)-sided\b", gloss)
        if named and named.group(1) in _NUMBER_WORDS:
            return _NUMBER_WORDS[named.group(1)]
        counted = re.search(r"\bwith (\d+) sides\b", gloss)
        if counted:
            return int(counted.group(1))
    return None


def formulas(shape: str) -> tuple:
    """The formulas of a shape; a polygon WordNet knows has the polygon's
    too, so a triangle has both."""
    own = SHAPES.get(shape, ())
    if sides(shape) is not None:
        own = own + _POLYGON
    return own


def shapes() -> list:
    return sorted(set(SHAPES) | {one for one in POLYGONS
                                 if sides(one) is not None})


def quantities(shape: str) -> list:
    return sorted({_name(symbol) for formula in formulas(shape)
                   for symbol in formula.free_symbols})


# -- the problem: formulas as actions ------------------------------------------

def _known(name: str) -> str:
    return f"known {name}"


def actions(shape: str) -> tuple:
    """Each formula once for each quantity in it: that quantity, from the
    others."""
    out = []
    for index, formula in enumerate(formulas(shape)):
        names = sorted(_name(one) for one in formula.free_symbols)
        for name in names:
            others = [one for one in names if one != name]
            out.append(W.Action(
                f"find {name} by formula-{index}",
                frozenset(_known(one) for one in others),
                frozenset({_known(name)}), frozenset()))
    return tuple(out)


class Sheet(W.Imagined):
    """The page the working is done on: what is known, and its value.

    An `Imagined` world, not a real one: working a measure out moves
    nothing, so there is no change to announce and no store for the act to
    declare (`v687.executive.effect`). A real world here would refuse to
    be acted on from the page at all."""

    def __init__(self, shape: str, givens: dict) -> None:
        self.shape = shape
        self.values = dict(givens)
        if sides(shape) is not None:
            self.values.setdefault("sides", S.Integer(sides(shape)))
        #: (quantity, the formula used, its value)
        self.steps: list = []
        super().__init__({_known(one) for one in self.values})

    def do(self, action: W.Action) -> bool:
        if not self.can(action):
            return False
        name = action.name.split()[1]
        formula = formulas(self.shape)[int(action.name.rsplit("-", 1)[1])]
        known = {q(one): value for one, value in self.values.items()
                 if one != name}
        found = [one for one in S.solve(formula.subs(known), q(name))
                 if _positive(one)]
        if not found:
            return False
        before = self.facts
        value = S.simplify(found[0])
        self.values[name] = value
        self.steps.append((name, formula, value))
        self.facts = self.facts | {_known(name)}
        self.did.append(action)
        self.announce(action, before)
        return True


def _positive(value) -> bool:
    number = value
    try:
        from sympy.physics.units import Quantity
        number = value.subs({one: 1 for one in value.atoms(Quantity)})
    except Exception:                                # noqa: BLE001
        pass
    return not (number.is_number and number.is_extended_nonpositive)


@dataclass
class Measured:
    shape: str
    wanted: str
    value: object = None
    #: (quantity, formula, value), in the order worked out
    steps: list = field(default_factory=list)
    #: what would let it be worked out, where it could not be
    needs: list = field(default_factory=list)
    trouble: str = ""


def _plan(shape: str, known, wanted: str) -> tuple:
    found = acting.think(actions(shape), frozenset(_known(one)
                                                   for one in known),
                         {_known(wanted)})
    return found.plan


def _same_units(givens: dict) -> dict:
    """Lengths given in different units, put in the first one's: `2 m` and
    `50 cm` are worked out as `200 cm` and `50 cm`."""
    from sympy.physics.units import Quantity, convert_to
    first = None
    for name, value in givens.items():
        units = sorted(getattr(value, "atoms", lambda _: set())(Quantity),
                       key=str)
        if units and DIMENSIONS.get(name) == 1:
            first = units[0]
            break
    if first is None:
        return givens
    out = {}
    for name, value in givens.items():
        dimension = DIMENSIONS.get(name)
        if isinstance(dimension, int) and dimension and hasattr(
                value, "atoms") and value.atoms(Quantity):
            value = convert_to(value, first ** dimension)
        out[name] = value
    return out


def measure(shape: str, wanted: str, givens: dict,
            learner=None) -> Measured:
    """Work out one measure of a shape from what was given: planned, done,
    watched, by v691's agent."""
    from research.v687.executive import Working
    out = Measured(shape, wanted)
    if not formulas(shape):
        out.trouble = f"I do not know the formulas of a {shape}"
        return out
    if wanted not in quantities(shape):
        out.trouble = f"a {shape} has no {said(wanted)} I know of"
        return out
    givens = _same_units({name: value for name, value in givens.items()
                          if name in quantities(shape)})
    sheet = Sheet(shape, givens)
    if wanted in sheet.values:
        out.value = sheet.values[wanted]
        return out
    goal = frozenset({_known(wanted)})
    problem = W.Problem(f"{wanted} of a {shape}", sheet.facts, goal,
                        actions(shape))
    report = acting.Attempt(name=problem.name)
    agent = acting.agent(problem, sheet, None, report, tries=3,
                         learner=learner)
    agent.run(Working(goal=f"find the {wanted} of a {shape}"))
    out.steps = sheet.steps
    if sheet.solved(goal):
        out.value = sheet.values[wanted]
        return out
    # No plan: which one more thing known would give one.
    known = set(sheet.values)
    bases = BASES.get(shape, ("side",))
    out.needs = sorted((one for one in quantities(shape)
                        if one not in known and one != wanted
                        and _plan(shape, known | {one}, wanted)),
                       key=lambda one: (one not in bases, one))
    return out


def said(name: str) -> str:
    """A quantity as it is said: `surface-area` as `surface area`."""
    return {"leg2": "other leg", "base2": "other base",
            "sides": "number of sides",
            "angle-sum": "sum of the interior angles",
            "slant": "slant height"}.get(name, name.replace("-", " "))


def unit_of(name: str) -> str:
    return "degrees" if DIMENSIONS.get(name) == "degrees" else ""
