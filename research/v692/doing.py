"""What each act asks, done exactly.

The encoder says what an utterance asks and what it is about
(`reading.py`); this does it, with sympy, and says what came of it in the
shape the rest of the system speaks in: a **value** (`68`, `x = 2 or x =
3`) or a **verdict** (yes or no, with the reason -- `91 is not prime: 7 times
13`). Nothing here reads words, and nothing here is learned.

A result that cannot be had -- a limit that does not exist, a matrix with no
inverse, a kind the object is not the sort of thing to be -- is said as not
known or not so, never guessed.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from fractions import Fraction

import sympy as S
from sympy.core.function import AppliedUndef

from sympy.logic.inference import satisfiable
from sympy.physics.units import Quantity, convert_to

from research.v692 import curriculum as C
from research.v692.saying import decimal, tight


@dataclass
class Result:
    """What doing an act came to."""

    act: str
    #: value | yes | no | unknown
    stance: str
    #: the value, as an object, where there is one
    value: object = None
    #: the value or verdict written down, as a person writes it
    text: str = ""
    #: why, where a verdict has a reason: `7 times 13`
    because: str = ""
    #: the object it was about, written down
    about: str = ""
    steps: list = field(default_factory=list)
    #: what would let it be worked out, where it could not be: a measure
    #: short of a given
    needs: list = field(default_factory=list)


def written(obj) -> str:
    """An object as a person writes it: `x^2 - 5x + 6`, `3/4`, `{1, 2}`."""
    if isinstance(obj, (list, tuple)) and not isinstance(obj, S.Basic):
        return ", ".join(written(one) for one in obj)
    if isinstance(obj, S.MatrixBase):
        return tight(obj)
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, S.Rel):
        symbol = {S.Eq: "=", S.Ne: "!=", S.Lt: "<", S.Gt: ">", S.Le: "<=",
                  S.Ge: ">="}[type(obj)]
        return f"{written(obj.lhs)} {symbol} {written(obj.rhs)}"
    if isinstance(obj, (S.And, S.Or, S.Not, S.Implies)):
        return S.sstr(obj)
    if getattr(obj, "is_Float", False):
        return decimal(obj)
    if hasattr(obj, "atoms") and obj.atoms(Quantity):
        return with_units(obj)
    text = S.sstr(obj, order="grlex").replace("**", "^")
    text = re.sub(r"\bexp\(([^()]*)\)", r"e^(\1)", text)
    if getattr(obj, "has", None) and obj.has(S.I):
        text = re.sub(r"\bI\b", "i", text)
    return text.replace("*", "")  if _implicit(obj) else text.replace(
        "*", " * ")


def _implicit(obj) -> bool:
    """Whether a product can be written by juxtaposition, `2x`: only where
    no two numbers would end up side by side."""
    text = S.sstr(obj)
    return not re.search(r"\d\*\d|\)\*\d", text)


def _number(obj):
    """A number as it is best said: exact, and an integer where it is one."""
    value = S.nsimplify(obj) if obj.is_number else obj
    return value


def _items(value) -> list:
    if isinstance(value, S.Tuple):
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _variable(expr, var=None):
    if var is not None:
        return var
    free = sorted(getattr(expr, "free_symbols", ()), key=lambda one: one.name)
    return free[0] if free else S.Symbol("x")


def contributes(act: str, handler) -> None:
    """Register a later layer's act: `handler(get, parts)` -> `Result`.
    v693 adds `design` this way."""
    HANDLERS[act] = handler


def do(act: str, parts: dict, bound: dict | None = None) -> Result:
    """Do an act on its parts (`EXPR`, `VAR`, `A`, `B`, `LIST`, `KIND`,
    `OTHER`), with the variables said to be something (`bound`) put in."""
    bound = bound or {}

    def get(role, default=None):
        value = parts.get(role, default)
        if bound and hasattr(value, "subs") and act != "let":
            # A variable said to be something is that; a function, what
            # it was defined as (`f(3)` with `f` a `Lambda`).
            value = value.subs(bound)
            if value.has(S.Lambda):
                value = value.doit()
        return value

    handler = HANDLERS.get(act)
    if handler is None:
        return Result(act, "unknown", text=f"I do not know how to {act}")
    try:
        return handler(get, parts)
    except (ValueError, TypeError, NotImplementedError, ZeroDivisionError,
            AttributeError, KeyError, IndexError, RecursionError,
            S.PolynomialError) as trouble:
        return Result(act, "unknown", text=f"I could not work that out "
                                           f"({type(trouble).__name__})")


def _value(get, parts):
    expr = get("EXPR")
    if isinstance(expr, S.Rel):
        return _check(get, parts)
    value = S.simplify(expr) if not expr.is_number else _number(
        S.simplify(expr))
    if value.free_symbols:
        return Result("value", "value", value, written(value),
                      about=written(expr))
    exact = written(value)
    text = exact
    if not value.is_Rational and value.is_real:
        text = f"{exact}, about {float(value):.4g}"
    return Result("value", "value", value, text, about=written(expr))


def _simplify(get, parts):
    value = S.simplify(get("EXPR"))
    return Result("simplify", "value", value, written(value))


def _expand(get, parts):
    value = S.expand(get("EXPR"))
    return Result("expand", "value", value, written(value))


def _factor(get, parts):
    expr = get("EXPR")
    if expr.is_Integer:
        return _prime_factors(get, parts)
    value = S.factor(expr)
    return Result("factor", "value", value, written(value))


def _solve(get, parts):
    equation = get("EXPR")
    if not isinstance(equation, S.Rel):
        equation = S.Eq(equation, 0)
    var = _variable(equation, get("VAR"))
    found = S.solveset(equation, var, domain=S.S.Reals)
    if not isinstance(equation, S.Equality):
        # An inequality: what it holds for, said as one.
        if found == S.S.EmptySet:
            return Result("solve", "no", found, f"no real {var} makes "
                          f"{written(equation)} true",
                          about=written(equation))
        if found == S.S.Reals:
            return Result("solve", "value", found, f"every real {var}",
                          about=written(equation))
        relation = _interval(found, var)
        return Result("solve", "value", relation, written(relation),
                      about=written(equation))
    if found == S.S.EmptySet:
        return Result("solve", "no", found,
                      f"no real {var} makes {written(equation)} true",
                      about=written(equation))
    if isinstance(found, S.FiniteSet):
        values = sorted(found, key=S.default_sort_key)
        text = " or ".join(f"{var} = {written(one)}" for one in values)
        return Result("solve", "value", values, text,
                      about=written(equation), steps=_worked(equation, var))
    return Result("solve", "value", found, f"{var} is in {found}",
                  about=written(equation))


def _derivative(get, parts):
    expr = get("EXPR")
    var = _variable(expr, get("VAR"))
    value = S.diff(expr, var)
    return Result("derivative", "value", value, written(S.simplify(value)),
                  about=written(expr))


def _integral(get, parts):
    expr = get("EXPR")
    var = _variable(expr, get("VAR"))
    low, high = get("A"), get("B")
    if low is not None and high is not None:
        value = S.integrate(expr, (var, low, high))
        return Result("integral", "value", value, written(value),
                      about=written(expr))
    value = S.integrate(expr, var)
    return Result("integral", "value", value,
                  f"{written(value)} + C", about=written(expr))


def _limit(get, parts):
    expr = get("EXPR")
    var = _variable(expr, get("VAR"))
    try:
        # Both sides: sympy's default is from the right, which gives 1/x a
        # limit of infinity at 0 where there is none.
        value = S.limit(expr, var, get("A"), dir="+-")
    except ValueError:
        return Result("limit", "unknown", None, "the limit does not exist: "
                                                "it is not the same from "
                                                "the left and the right")
    if value in (S.zoo, S.nan) or isinstance(value, S.AccumBounds):
        return Result("limit", "unknown", value, "the limit does not exist")
    return Result("limit", "value", value, written(value),
                  about=written(expr))


def _at(get, parts):
    expr = parts.get("EXPR")
    var = _variable(expr, parts.get("VAR"))
    value = S.simplify(expr.subs(var, get("A")))
    return Result("at", "value", value, written(value), about=written(expr))


def _gcd(get, parts):
    items = [S.Integer(one) for one in _items(get("LIST"))]
    value = S.gcd_list(items)
    return Result("gcd", "value", value, written(value))


def _lcm(get, parts):
    items = [S.Integer(one) for one in _items(get("LIST"))]
    value = S.lcm_list(items)
    return Result("lcm", "value", value, written(value))


def _divisors(get, parts):
    number = int(_first(get, "A", "EXPR"))
    value = S.divisors(number)
    return Result("divisors", "value", value, ", ".join(map(str, value)))


def _prime_factors(get, parts):
    number = int(_first(get, "A", "EXPR"))
    found = S.factorint(number)
    said = " times ".join(
        f"{prime}" if power == 1 else f"{prime}^{power}"
        for prime, power in sorted(found.items()))
    return Result("prime factors", "value", found, said or str(number))


def _first(get, *roles):
    for role in roles:
        value = get(role)
        if role == "LIST" and value is not None:
            return list(_items(value))
        if value is not None:
            return value
    raise ValueError("nothing to do it to")


def _is_kind(get, parts):
    obj = _first(get, "A", "EXPR", "LIST")
    kind = parts.get("KIND")
    if kind is None:
        return Result("is kind", "unknown", text="I did not catch what kind")
    holds = kind.holds(obj)
    about = written(obj)
    because = _why_kind(kind, obj, holds)
    stance = "yes" if holds else "no"
    text = (f"{about} is {kind.said[0]}" if holds
            else f"{about} is not {kind.said[0]}")
    return Result("is kind", stance, holds, text, because, about)


def _why_kind(kind, obj, holds) -> str:
    """The reason, where one can be shown: the factors of a number that is
    not prime, what a number that is not a square lies between."""
    try:
        if kind.name == "prime number" and not holds and obj.is_Integer \
                and obj > 1:
            smallest = min(S.primefactors(int(obj)))
            return f"{obj} = {smallest} times {int(obj) // smallest}"
        if kind.name in ("even number", "odd number") and obj.is_Integer:
            return f"{obj} = 2 times {int(obj) // 2}" + (
                " plus 1" if int(obj) % 2 else "")
        if kind.name == "perfect square" and obj.is_Integer and obj >= 0:
            root = int(S.sqrt(obj)) if holds else None
            return f"{root} squared is {obj}" if holds else ""
        if kind.name == "tautology" and not holds:
            return f"it is false when {_counterexample(S.Not(obj))}"
        if kind.name == "contradiction" and not holds:
            return f"it is true when {_counterexample(obj)}"
        if kind.name == "satisfiable" and holds:
            return f"it is true when {_counterexample(obj)}"
        if kind.name == "composite number" and holds:
            smallest = min(S.primefactors(int(obj)))
            return f"{obj} = {smallest} times {int(obj) // smallest}"
    except Exception:                                # noqa: BLE001
        return ""
    return ""


def _divides(get, parts):
    number, by = int(get("A")), int(get("B"))
    quotient, remainder = divmod(number, by)
    holds = remainder == 0
    text = (f"yes: {number} = {by} times {quotient}" if holds else
            f"no: {number} divided by {by} is {quotient} remainder "
            f"{remainder}")
    return Result("divides", "yes" if holds else "no", holds, text,
                  about=f"{number} and {by}")


def _check(get, parts):
    relation = get("EXPR")
    if not isinstance(relation, S.Rel):
        return Result("check", "unknown", text="there is nothing to check")
    left, right = S.simplify(relation.lhs), S.simplify(relation.rhs)
    holds = relation.func(left, right)
    if holds not in (S.true, S.false):
        return Result("check", "unknown", text="that depends on the "
                                               "variables")
    written_relation = written(relation)
    text = (f"yes, {written_relation}" if holds == S.true else
            f"no: {written(relation.lhs)} is {written(left)}")
    return Result("check", "yes" if holds == S.true else "no",
                  holds == S.true, text, about=written_relation)


def _stat(name):
    def run(get, parts):
        items = [Fraction(str(S.nsimplify(one))) for one in
                 _items(get("LIST"))]
        if name == "mean":
            value = sum(items, Fraction(0)) / len(items)
        elif name == "median":
            value = Fraction(statistics.median(items))
        elif name == "mode":
            found = statistics.multimode(items)
            if len(found) == len(set(items)) and len(found) > 1:
                return Result("mode", "unknown", text="every value is as "
                                                      "common as any other")
            value = found if len(found) > 1 else found[0]
        elif name == "range":
            value = max(items) - min(items)
        else:
            value = sum(items, Fraction(0))
        if isinstance(value, list):
            return Result(name, "value", value, " and ".join(
                written(S.Rational(one.numerator, one.denominator))
                for one in value))
        value = S.Rational(value.numerator, value.denominator)
        return Result(name, "value", value, written(value))
    return run


def _choose(get, parts):
    value = S.binomial(get("A"), get("B"))
    return Result("choose", "value", value, written(value))


def _arrange(get, parts):
    value = S.factorial(_first(get, "A", "EXPR"))
    return Result("arrange", "value", value, written(value))


def _matrix(name):
    def run(get, parts):
        matrix = get("EXPR")
        if not isinstance(matrix, S.MatrixBase):
            return Result(name, "unknown", text="that is not a matrix")
        if name == "determinant":
            value = matrix.det()
        elif name == "inverse":
            if matrix.det() == 0:
                return Result(name, "no", None, "it has no inverse: its "
                                                "determinant is 0")
            value = matrix.inv()
        else:
            value = matrix.T
        return Result(name, "value", value, written(value))
    return run


def _sets(name):
    def run(get, parts):
        first, other = get("EXPR"), get("OTHER")
        if name == "union":
            value = S.Union(first, other)
        elif name == "intersection":
            value = S.Intersection(first, other)
        else:
            holds = first.is_subset(other)
            return Result(name, "yes" if holds else "no", holds,
                          f"{written(first)} is {'' if holds else 'not '}"
                          f"a subset of {written(other)}")
        return Result(name, "value", value, written(value))
    return run


def _let(get, parts):
    var = parts.get("VAR")
    value = parts.get("A") if parts.get("A") is not None else parts.get(
        "EXPR")
    if isinstance(value, S.Equality) and var is None:
        var, value = value.lhs, value.rhs
    if isinstance(var, AppliedUndef):
        # `let f(x) = x^2 + 1`: f is that rule, whatever it is applied to.
        return Result("let", "noted", {var.func: S.Lambda(var.args, value)},
                      f"{var} = {written(value)}")
    if not isinstance(var, S.Symbol):
        return Result("let", "unknown", text="I did not catch which "
                                             "variable")
    return Result("let", "noted", {var: value}, f"{var} = {written(value)}")


def _worked(equation, var) -> list:
    """How the equation is solved by moves (`solving.solve`, v691's agent),
    where it is one the moves can solve: each move and what it left."""
    from research.v692 import solving
    try:
        if S.degree(S.expand(equation.lhs - equation.rhs), var) > 2:
            return []
        found = solving.solve(S.Eq(equation.lhs, equation.rhs,
                                   evaluate=False), var)
    except Exception:                                # noqa: BLE001
        return []
    if not found.solved:
        return []
    return [(move, " or ".join(written(one) for one in after))
            for move, after in found.steps]


def _interval(found, var):
    """Where an inequality holds, as the relation a person writes: `x > 2`,
    not `(2 < x) & (x < oo)`."""
    if isinstance(found, S.Interval):
        low, high = found.start, found.end
        if high == S.oo and low != -S.oo:
            return (S.Gt if found.left_open else S.Ge)(var, low)
        if low == -S.oo and high != S.oo:
            return (S.Lt if found.right_open else S.Le)(var, high)
    return found.as_relational(var)


def with_units(obj, exact: bool = False) -> str:
    """An amount in units, as written: `25pi cm^2`, `3.107 miles`."""
    from research.v692.saying import UNIT_WORDS
    units = [one for one in S.Mul.make_args(obj)
             if isinstance(one, Quantity) or (isinstance(one, S.Pow)
                                              and isinstance(one.base,
                                                             Quantity))]
    amount = S.Mul(*[one for one in S.Mul.make_args(obj)
                     if one not in units])
    about = ""
    if exact and amount.is_number and not amount.is_Rational:
        # `25pi square centimeters, about 78.54`
        shown = written(amount)
        about = f", about {float(amount):.4g}"
    elif amount.is_number and not amount.is_Integer:
        shown = f"{float(amount):.4g}"
    else:
        shown = written(amount)
    names = []
    for one in units:
        base, power = (one.base, int(one.exp)) if isinstance(
            one, S.Pow) else (one, 1)
        word = UNIT_WORDS[str(base.name)][0]
        plural = word + ("es" if word.endswith("h") else "s")
        if word == "foot":
            plural = "feet"
        word = word if amount == 1 else plural
        names.append(word if power == 1 else
                     f"{('square', 'cubic')[power - 2]} {word}")
    return f"{shown} {' '.join(names)}".strip() + about


def _solve_system(get, parts):
    equations = [one for one in (get("EXPR"), get("OTHER"))
                 if isinstance(one, S.Rel)]
    if len(equations) < 2:
        return Result("solve system", "unknown", text="I need two "
                                                      "equations")
    free = sorted(set().union(*[one.free_symbols for one in equations]),
                  key=lambda one: one.name)
    found = S.solve(equations, free, dict=True)
    if not found:
        return Result("solve system", "no", None, "no values make both "
                                                  "true")
    if len(found) > 1 or len(found[0]) < len(free):
        return Result("solve system", "unknown", text="there are many "
                                                      "solutions")
    values = [S.Eq(var, found[0][var]) for var in free]
    return Result("solve system", "value", values,
                  " and ".join(written(one) for one in values))


def _convert(get, parts):
    amount, target = get("A"), get("OTHER")
    if amount is None or target is None or not isinstance(
            target, Quantity):
        return Result("convert", "unknown", text="I did not catch the "
                                                 "units")
    value = convert_to(amount, target)
    number = value / target
    if number.is_number and not number.is_Integer:
        # `0.456 liters`, not `57/125 liters`.
        value = S.Float(f"{float(number):.4g}") * target
    if value.atoms(Quantity) != {target}:
        return Result("convert", "no", None, f"{with_units(amount)} is not "
                      f"something measured in {with_units(1 * target)}")
    return Result("convert", "value", value, with_units(value),
                  about=with_units(amount))


def _rule(items: list):
    """The rule of a sequence, in n from 1: a constant ratio, else the
    polynomial of least degree through all the terms but the last that the
    last confirms."""
    n = S.Symbol("n")
    if len(items) >= 3 and 0 not in items:
        ratios = {items[i + 1] / items[i] for i in range(len(items) - 1)}
        if len(ratios) == 1 and ratios != {1}:
            ratio = ratios.pop()
            return items[0] * ratio ** (n - 1)
    for degree in range(0, len(items) - 1):
        rule = S.expand(S.interpolate(items[:degree + 1], n))
        if all(rule.subs(n, k + 1) == items[k] for k in range(len(items))):
            # `(n + 3)^3` where it is one, not its expansion.
            factored = S.factor(rule)
            return factored if S.count_ops(factored) < S.count_ops(
                rule) else rule
    return None


def _sequence(name):
    def run(get, parts):
        items = [S.nsimplify(one) for one in _items(get("LIST"))]
        rule = _rule(items)
        if rule is None:
            return Result(name, "unknown", text="I cannot see the pattern")
        n = S.Symbol("n")
        if name == "nth term":
            return Result(name, "value", rule, written(rule),
                          about=written(items))
        place = len(items) + 1 if name == "next term" else get("A")
        if place is None:
            return Result(name, "unknown", text="I did not catch which term")
        value = rule.subs(n, place)
        return Result(name, "value", value, written(value),
                      because=f"the nth term is {written(rule)}",
                      about=written(items))
    return run


def _series(get, parts):
    expr = get("EXPR")
    var = _variable(expr, get("VAR"))
    low, high = get("A"), get("B")
    value = S.summation(expr, (var, low, high))
    if value in (S.oo, -S.oo, S.zoo, S.nan) or isinstance(value, S.Sum):
        return Result("series", "unknown", value, "it does not add up to a "
                                                  "number: the series "
                                                  "diverges")
    value = S.simplify(value)
    return Result("series", "value", value, written(value),
                  about=written(expr))


def _counterexample(proposition) -> str:
    found = satisfiable(proposition)
    if not found:
        return ""
    return " and ".join(f"{var} is {'true' if value else 'false'}"
                        for var, value in sorted(found.items(), key=str))


def _equivalent(get, parts):
    one, other = get("EXPR"), get("OTHER")
    differs = S.Xor(one, other)
    holds = not satisfiable(differs)
    text = (f"{written(one)} and {written(other)} are equivalent" if holds
            else f"{written(one)} and {written(other)} are not equivalent")
    because = "" if holds else f"they differ when {_counterexample(differs)}"
    return Result("equivalent", "yes" if holds else "no", holds, text,
                  because)


def faces() -> int:
    """How many faces a die has, as WordNet says: `a small cube with 1 to
    6 spots on the six faces`."""
    try:
        from nltk.corpus import wordnet
        gloss = wordnet.synset("die.n.01").definition()
        found = re.search(r"\b(\w+) faces\b", gloss)
        words = {"four": 4, "six": 6, "eight": 8, "twelve": 12, "twenty": 20}
        return words.get(found.group(1), 6) if found else 6
    except Exception:                                # noqa: BLE001
        return 6


def _dice(get, parts):
    count = int(get("A") or 1)
    total = int(_first(get, "B", "EXPR"))
    sides = faces()
    if not 1 <= count <= 6:
        return Result("dice", "unknown", text="that is too many dice")
    ways = [1]
    for _ in range(count):
        ways = [sum(ways[i - k] for k in range(1, sides + 1)
                    if 0 <= i - k < len(ways)) for i in range(len(ways)
                                                             + sides)]
    favourable = ways[total] if 0 <= total < len(ways) else 0
    value = S.Rational(favourable, sides ** count)
    return Result("dice", "value", value, written(value),
                  because=f"{favourable} of the {sides ** count} ways the "
                          f"dice can land")


def _coins(get, parts):
    count, heads = int(get("A")), int(get("B"))
    value = S.binomial(count, heads) / S.Integer(2) ** count
    return Result("coins", "value", value, written(value),
                  because=f"{S.binomial(count, heads)} of the {2 ** count} "
                          f"ways the coins can land")


def _decimal(value):
    """A number as a decimal, exactly as written: rounding is about the
    digits, so it is not done in binary."""
    from decimal import Decimal, InvalidOperation
    if value is None or not getattr(value, "is_number", False):
        raise ValueError("there is no number to round")
    try:
        return Decimal(decimal(value) if value.is_Float else str(value))
    except InvalidOperation:
        return Decimal(repr(float(value)))


def _round_places(get, parts):
    from decimal import ROUND_HALF_UP, Decimal
    number, places = _decimal(get("A")), int(get("B") or 0)
    found = number.quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP)
    return Result("round places", "value", S.Float(str(found)), str(found))


def _round_nearest(get, parts):
    from decimal import ROUND_HALF_UP, Decimal
    number = _decimal(get("A"))
    step = int(get("B") or 1)
    found = (number / step).quantize(Decimal(1), ROUND_HALF_UP) * step
    value = S.Integer(int(found))
    return Result("round nearest", "value", value, written(value))


def _part_of(get, parts):
    part, whole = get("EXPR"), _first(get, "A", "B")
    value = S.simplify(S.nsimplify(part) * S.nsimplify(whole))
    return Result("part of", "value", value, written(value),
                  about=f"{written(part)} of {written(whole)}")


def _percent(name):
    def run(get, parts):
        first, second = get("A"), get("B")
        if name == "percent of":
            value = S.nsimplify(first) * S.nsimplify(second) / 100
            return Result(name, "value", value, _plain(value))
        if name == "as percent":
            value = S.nsimplify(first) / S.nsimplify(second) * 100
            return Result(name, "value", value, f"{_plain(value)}%")
        if first == 0:
            return Result(name, "unknown", text="there is no change from 0 "
                                                "as a percentage")
        value = (S.nsimplify(second) - S.nsimplify(first)) / S.nsimplify(
            first) * 100
        way = "an increase" if value > 0 else "a decrease" if value < 0             else "no change"
        return Result(name, "value", value, f"{_plain(value)}%",
                      because=f"{way} of {_plain(abs(value))}%")
    return run


def _plain(value) -> str:
    """A number as a person writes one: whole, or to two places."""
    if value.is_Integer:
        return str(value)
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _vectors(name):
    def run(get, parts):
        first = get("EXPR")
        if not isinstance(first, S.Tuple):
            return Result(name, "unknown", text="that is not a vector")
        one = S.Matrix(list(first))
        if name == "magnitude":
            value = S.sqrt(sum(part ** 2 for part in first))
            return Result(name, "value", value, written(value))
        other = get("OTHER")
        if not isinstance(other, S.Tuple) or len(other) != len(first):
            return Result(name, "unknown", text="the vectors are not the "
                                                "same size")
        two = S.Matrix(list(other))
        if name == "dot":
            value = one.dot(two)
            return Result(name, "value", value, written(value))
        if len(first) != 3:
            return Result(name, "unknown", text="a cross product needs "
                                                "vectors of three parts")
        value = S.Tuple(*one.cross(two))
        return Result(name, "value", value, written(value))
    return run


def _complex(name):
    def run(get, parts):
        number = S.expand(get("EXPR"))
        value = (S.Abs(number) if name == "modulus" else S.conjugate(number))
        value = S.simplify(value)
        return Result(name, "value", value, written(value),
                      about=written(number))
    return run


def _measure(get, parts):
    from research.v692 import measuring as M
    shape, wanted = parts.get("SHAPE"), parts.get("WANTED")
    if shape is None or wanted is None:
        return Result("measure", "unknown", text="I did not catch what to "
                                                 "work out")
    found = M.measure(shape, wanted, parts.get("GIVENS") or {})
    if found.trouble:
        return Result("measure", "unknown", text=found.trouble)
    steps = [(name, f"{M.said(name)} = {measured(name, value)}")
             for name, _, value in found.steps]
    if found.value is None:
        needs = found.needs
        if not needs:
            return Result("measure", "unknown", text=f"I cannot work out "
                          f"the {M.said(wanted)} of a {shape} from that")
        others = ", or ".join(f"its {M.said(one)}" for one in needs[1:])
        text = (f"I need the {M.said(needs[0])} of the {shape}"
                + (f" (or {others})" if others else ""))
        return Result("measure", "unknown", None, text, steps=steps,
                      needs=needs)
    value = found.value
    because = "; ".join(said for _, said in steps[:-1])
    return Result("measure", "value", value, measured(wanted, value),
                  because=because, about=f"the {M.said(wanted)} of the "
                                         f"{shape}", steps=steps)


def measured(name: str, value) -> str:
    from research.v692 import measuring as M
    if value.atoms(Quantity):
        return with_units(value, exact=True)
    text = written(value)
    if M.unit_of(name):
        text += f" {M.unit_of(name)}"
    if value.is_number and not value.is_Rational:
        text += f", about {float(value):.4g}"
    return text


def continued(pending: dict, given: dict) -> dict:
    """A measure that was short of a given, with it: `what is the area of a
    rectangle with length 4` and then `the width is 6`."""
    out = dict(pending)
    out["GIVENS"] = {**(pending.get("GIVENS") or {}),
                     **(given.get("GIVENS") or {})}
    return out


def _given(get, parts):
    givens = parts.get("GIVENS") or {}
    if not givens:
        return Result("given", "unknown", text="I did not catch what that "
                                               "was")
    return Result("given", "noted", dict(givens), ", ".join(
        f"{name} = {written(value)}" for name, value in givens.items()))


HANDLERS = {
    "value": _value, "simplify": _simplify, "expand": _expand,
    "factor": _factor, "solve": _solve, "derivative": _derivative,
    "integral": _integral, "limit": _limit, "at": _at, "gcd": _gcd,
    "lcm": _lcm, "divisors": _divisors, "prime factors": _prime_factors,
    "is kind": _is_kind, "divides": _divides, "check": _check,
    "mean": _stat("mean"), "median": _stat("median"), "mode": _stat("mode"),
    "range": _stat("range"), "sum": _stat("sum"), "choose": _choose,
    "arrange": _arrange, "determinant": _matrix("determinant"),
    "inverse": _matrix("inverse"), "transpose": _matrix("transpose"),
    "union": _sets("union"), "intersection": _sets("intersection"),
    "subset": _sets("subset"), "let": _let,
    "solve system": _solve_system, "convert": _convert,
    "next term": _sequence("next term"), "nth term": _sequence("nth term"),
    "term": _sequence("term"), "series": _series,
    "equivalent": _equivalent, "dice": _dice, "coins": _coins,
    "round places": _round_places, "round nearest": _round_nearest,
    "part of": _part_of, "percent of": _percent("percent of"),
    "as percent": _percent("as percent"),
    "percent change": _percent("percent change"),
    "dot": _vectors("dot"), "cross": _vectors("cross"),
    "magnitude": _vectors("magnitude"), "modulus": _complex("modulus"),
    "conjugate": _complex("conjugate"), "measure": _measure,
    "given": _given,
}

#: Every act the curriculum asks is one that can be done.
assert set(HANDLERS) >= {one.name for one in C.ACTS}
