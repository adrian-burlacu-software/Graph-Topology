"""What a design is for: a specification, clause by clause, checked exactly.

v692 answers questions whose answer is a computation: *what is the
derivative of x cubed*. A design goal is the other way round -- *a
polynomial with roots 2 and -3 whose value at 0 is 12* -- and asks for an
object nobody has yet, of which only properties are known. So the goal is
a **specification**: a kind of object and the clauses it must satisfy.

Every clause is two things at once:

    residuals   expressions that are zero exactly when it holds, in
                whatever unknowns the object has -- what turns a proposed
                form with holes into equations sympy can solve
    holds       the exact check on a finished object

A clause with no residuals (`integer`: whole coefficients) is only
checked. Nothing a designer hands back is unchecked: a design is right
when `Spec.holds` says so, whatever it looks like, and there is no
expected answer to compare against -- any object that meets the
specification is one.

`features` describes a specification the way `solving.features` describes
an equation: what it gives (roots, values, terms), how much, and the shape
of its data (terms of one sign, evenly stepped, evenly multiplied). The
designer's learner reads which of these a design move turns out to need.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sympy as S

x, n = S.symbols("x n")

KINDS = {"polynomial": x, "sequence": n, "function": x}

#: How many derivatives at 0 a clause that must hold for every x is
#: turned into. The check is exact all the same (`CHECKS`): these only
#: propose.
SAMPLES = tuple(range(5))


def _identity(expression, var) -> list:
    """Something that is zero for every x, as equations: it and its first
    derivatives are zero at 0. Taken at 0 rather than at points, because
    e^(kx) and sin(wx) there are polynomials in k and w -- equations sympy
    solves -- where at x = 1 they are e^k and sin(w), which it may not."""
    return [S.diff(expression, var, order).subs(var, 0)
            for order in SAMPLES]


def _ode(obj, var, second, first, zeroth, rhs=0):
    """second f'' + first f' + zeroth f - rhs: zero for every x."""
    return (second * S.diff(obj, var, 2) + first * S.diff(obj, var)
            + zeroth * obj - rhs)


def _poly(obj, var):
    return S.Poly(S.expand(obj), var)


def _residual_root(obj, var, at, times=1):
    return [S.diff(obj, var, k).subs(var, at) for k in range(times)]


def _residual_degree(obj, var, degree):
    found = _poly(obj, var)
    return [found.coeff_monomial(var ** k)
            for k in range(degree + 1, found.degree() + 1)]


RESIDUALS = {
    "root": _residual_root,
    "value": lambda obj, var, at, is_: [obj.subs(var, at) - is_],
    "slope": lambda obj, var, at, is_: [S.diff(obj, var).subs(var, at) - is_],
    "stationary": lambda obj, var, at: [S.diff(obj, var).subs(var, at)],
    "leading": lambda obj, var, is_: [_poly(obj, var).LC() - is_],
    "degree": _residual_degree,
    "term": lambda obj, var, at, is_: [obj.subs(var, at) - is_],
    "total": lambda obj, var, count, is_: [
        sum(obj.subs(var, k) for k in range(1, count + 1)) - is_],
    "integer": lambda obj, var: [],
    "derivative": lambda obj, var, is_: _identity(S.diff(obj, var) - is_,
                                                   var),
    "ode": lambda obj, var, *args: _identity(_ode(obj, var, *args), var),
}


def _is_integer_coefficients(obj, var) -> bool:
    return all(one.is_integer for one in _poly(obj, var).all_coeffs())


def _vanishes(expression) -> bool:
    return S.simplify(expression) == 0 or S.simplify(
        S.expand(S.expand_trig(expression))) == 0


CHECKS = {
    "degree": lambda obj, var, degree: _poly(obj, var).degree() == degree,
    "integer": _is_integer_coefficients,
    "derivative": lambda obj, var, is_: _vanishes(S.diff(obj, var) - is_),
    "ode": lambda obj, var, *args: _vanishes(_ode(obj, var, *args)),
}

#: How each clause is said, for explanations.
SAID = {
    "root": lambda at, times=1: (f"a root at {at}" if times == 1 else
                                 f"a root of multiplicity {times} at {at}"),
    "value": lambda at, is_: f"value {is_} at {at}",
    "slope": lambda at, is_: f"slope {is_} at {at}",
    "stationary": lambda at: f"a turning point at {at}",
    "leading": lambda is_: f"leading coefficient {is_}",
    "degree": lambda degree: f"degree {degree}",
    "term": lambda at, is_: f"term {at} equal to {is_}",
    "total": lambda count, is_: f"the first {count} terms adding to {is_}",
    "integer": lambda: "whole-number coefficients",
    "derivative": lambda is_: f"derivative {is_}",
    "ode": lambda second, first, zeroth, rhs=0: (
        f"{second} f'' + {first} f' + {zeroth} f = {rhs}"),
}


@dataclass(frozen=True)
class Clause:
    kind: str
    args: tuple = ()

    def residuals(self, obj, var) -> list:
        return RESIDUALS[self.kind](obj, var, *self.args)

    def holds(self, obj, var) -> bool:
        try:
            check = CHECKS.get(self.kind)
            if check is not None and not check(obj, var, *self.args):
                return False
            return all(S.simplify(one) == 0
                       for one in self.residuals(obj, var))
        except (S.PolynomialError, TypeError, ValueError):
            return False

    def conditions(self) -> int:
        """How many equations the clause puts on an object."""
        if self.kind == "root":
            return self.args[1] if len(self.args) > 1 else 1
        if self.kind in ("derivative", "ode"):
            # A condition at every x: as many as the samples it is
            # solved from.
            return len(SAMPLES)
        return 0 if self.kind in ("integer", "degree") else 1

    def said(self) -> str:
        return SAID[self.kind](*self.args)


def C(kind: str, *args) -> Clause:
    return Clause(kind, tuple(S.sympify(one) if not isinstance(one, int)
                              else one for one in args))


@dataclass
class Spec:
    """A kind of object, and what it must satisfy."""

    kind: str
    clauses: tuple
    #: how it was put, for reports
    text: str = ""
    var: S.Symbol = field(default=None)

    def __post_init__(self) -> None:
        if self.var is None:
            self.var = KINDS[self.kind]
        self.clauses = tuple(self.clauses)

    def of(self, kind: str) -> list:
        return [one for one in self.clauses if one.kind == kind]

    def residuals(self, obj) -> list:
        return [one for clause in self.clauses
                for one in clause.residuals(obj, self.var)]

    def conditions(self) -> int:
        return sum(one.conditions() for one in self.clauses)

    def holds(self, obj) -> bool:
        """Whether `obj` meets the specification: an object of the kind,
        with real coefficients, satisfying every clause."""
        if obj is None:
            return False
        obj = S.sympify(obj)
        if obj == 0:
            # Nothing is not a design: the zero polynomial has every root.
            return False
        if obj.free_symbols - {self.var}:
            return False
        if self.kind == "polynomial" and not obj.is_polynomial(self.var):
            return False
        try:
            if self.kind == "polynomial" and not all(
                    one.is_real for one in _poly(obj, self.var).all_coeffs()):
                return False
        except S.PolynomialError:
            return False
        return all(one.holds(obj, self.var) for one in self.clauses)

    def said(self) -> str:
        return self.text or (f"a {self.kind} with " + ", ".join(
            one.said() for one in self.clauses))

    def features(self) -> set:
        """What the specification is like, for the learner to read."""
        out = {self.kind}
        roots = self.of("root")
        others = [one for one in self.clauses
                  if one.kind not in ("root", "integer", "degree")]
        if roots:
            out.add("roots-given")
            out.add("few-others" if len(others) <= 1 else "many-others")
        if any(one.kind in ("value", "slope", "stationary")
               for one in self.clauses):
            out.add("values-given")
        for kind, name in (("degree", "degree-given"),
                           ("leading", "leading-given"),
                           ("integer", "integer-wanted"),
                           ("total", "sum-given"),
                           ("derivative", "derivative-given"),
                           ("ode", "equation-given")):
            if self.of(kind):
                out.add(name)
        terms = sorted((one.args for one in self.of("term")),
                       key=lambda one: one[0])
        if terms:
            out.add("terms-given")
            out.add("two-terms" if len(terms) <= 2 else "three-or-more-terms")
            values = [value for _, value in terms]
            if all(value != 0 for value in values) and (
                    all(value > 0 for value in values)
                    or all(value < 0 for value in values)):
                out.add("same-sign-terms")
            out |= _shape(terms)
            if len(terms) >= 3:
                if _collinear(terms):
                    out.add("even-steps")
                if all(value != 0 for value in values) and _even_ratios(
                        terms):
                    out.add("even-ratios")
        return out


def _shape(terms) -> set:
    """What a run of consecutive terms is like, beyond a line or a ratio:
    steps that are one ratio apart (2^n + 1: steps 2, 4, 8, 16), steps
    that step evenly (a quadratic), and each term following from the two
    before by one rule (a sum of two geometric sequences: Fibonacci).
    Descriptions of the data, as a discriminant describes an equation --
    which form they call for is the learner's and the proposer's to find."""
    out = set()
    places = [int(place) for place, _ in terms]
    values = [S.Rational(value, 1) for _, value in terms]
    if len(values) < 4 or places != list(range(places[0],
                                               places[0] + len(places))):
        return out
    steps = [b - a for a, b in zip(values, values[1:])]
    if all(step != 0 for step in steps):
        ratios = {b / a for a, b in zip(steps, steps[1:])}
        if len(ratios) == 1 and ratios != {1}:
            out.add("geometric-steps")
    second = [b - a for a, b in zip(steps, steps[1:])]
    if len(set(second)) == 1 and second[0] != 0:
        out.add("even-second-steps")
    if len(values) >= 5:
        p, q = S.symbols("p q")
        found = S.solve([values[2] - p * values[1] - q * values[0],
                         values[3] - p * values[2] - q * values[1]],
                        [p, q], dict=True)
        if found and {p, q} <= set(found[0]) and all(
                values[k] == found[0][p] * values[k - 1]
                + found[0][q] * values[k - 2]
                for k in range(4, len(values))):
            out.add("two-term-rule")
    return out


def _collinear(points) -> bool:
    (k0, v0), (k1, v1) = points[0], points[1]
    slope = S.Rational(v1 - v0, 1) / (k1 - k0)
    return all(S.Rational(v - v0, 1) == slope * (k - k0)
               for k, v in points[2:])


def _even_ratios(points) -> bool:
    """Whether the terms could be one ratio apart: for every three,
    (v_j / v_i)^(k_l - k_i) = (v_l / v_i)^(k_j - k_i), exactly."""
    (k0, v0) = points[0]
    for (kj, vj), (kl, vl) in zip(points[1:], points[2:]):
        if (S.Rational(vj, 1) / v0) ** (kl - k0) != (
                S.Rational(vl, 1) / v0) ** (kj - k0):
            return False
    return True
