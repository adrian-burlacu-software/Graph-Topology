"""The shapes a design can take: forms with holes.

A designer does not search the space of all objects. It **proposes a
form** -- *something times (x - 2)(x + 3)*, *a first term times a ratio to
the n - 1* -- with unknowns where the specification will decide, and lets
the specification turn those unknowns into equations (`designing.fit`).
That is the move a mathematician calls an ansatz, and it is the part of
design an LLM does from its prior. Here each form is a row: what it is
built from, how many unknowns it has, and what an unknown nothing decides
is left as (`defaults`: a scale of 1, anything else 0).

A form's `needs` are what it is **built from** -- the roots form cannot
be built without roots -- not when it will work. When it works is not
written anywhere: it is what the designer's learner finds out by trying
(`lessons.Learner`), the way v692's solver found out when factoring works.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import sympy as S

from research.v693.spec import Spec


@dataclass(frozen=True)
class Form:
    name: str
    kind: str
    #: the features it is built from (`Spec.features`)
    needs: tuple
    #: (spec, extra degree) -> (object with unknowns, the unknowns, what
    #: each defaults to)
    build: Callable
    #: how it is said, for explanations
    said: str
    #: whether it can be made a degree bigger when the least is not
    #: enough: a condition on the slope says nothing of the constant, so
    #: two slopes need a quadratic, not the line two conditions suggest
    grows: bool = False
    #: built from other forms (`composed`): solved from as many conditions
    #: as it has unknowns, the rest only checked -- sympy will not solve
    #: an overdetermined system of powers in reasonable time
    parts: tuple = ()
    #: built out of the specification's own data -- its roots, the
    #: derivative it names -- rather than being a shape of its own: not a
    #: part other forms are made from
    from_data: bool = False


def _holes(prefix: str, count: int) -> list:
    return list(S.symbols(f"{prefix}0:{count}")) if count else []


def _root_product(spec: Spec):
    out = S.Integer(1)
    for clause in spec.of("root"):
        at = clause.args[0]
        times = clause.args[1] if len(clause.args) > 1 else 1
        out *= (spec.var - at) ** times
    return out


def _degree(spec: Spec) -> int | None:
    found = spec.of("degree")
    return int(found[0].args[0]) if found else None


def _general(spec: Spec, var, degree: int, prefix: str = "c"):
    """c0 + c1 x + ... + cd x^d, its top coefficient defaulting to 1."""
    holes = _holes(prefix, degree + 1)
    obj = sum((hole * var ** k for k, hole in enumerate(holes)),
              S.Integer(0))
    defaults = {hole: 0 for hole in holes}
    if holes:
        defaults[holes[-1]] = 1
    return obj, holes, defaults


def _sized(spec: Spec, taken: int = 0) -> int:
    """The least degree that the conditions not yet taken can fix: one
    unknown for each."""
    return max(spec.conditions() - taken - 1, 0)


def build_roots(spec: Spec, extra: int = 0):
    scale = S.Symbol("a")
    return scale * _root_product(spec), [scale], {scale: 1}


def build_general(spec: Spec, extra: int = 0):
    degree = _degree(spec)
    return _general(spec, spec.var, degree if degree is not None
                    else max(_sized(spec) + extra, 0))


def build_roots_times(spec: Spec, extra: int = 0):
    product = _root_product(spec)
    taken = sum(one.conditions() for one in spec.of("root"))
    degree = _degree(spec)
    rest = (degree - S.degree(product, spec.var) if degree is not None
            else max(_sized(spec, taken) + extra, 0))
    if rest < 0:
        rest = 0
    obj, holes, defaults = _general(spec, spec.var, rest, prefix="b")
    return product * obj, holes, defaults


def build_exponential(spec: Spec, extra: int = 0):
    scale, rate = S.symbols("a k")
    return (scale * S.exp(rate * spec.var), [scale, rate],
            {scale: 1, rate: 1})


def build_oscillation(spec: Spec, extra: int = 0):
    sine, cosine, frequency = S.symbols("a b w")
    return (sine * S.sin(frequency * spec.var)
            + cosine * S.cos(frequency * spec.var),
            [sine, cosine, frequency], {sine: 1, cosine: 0, frequency: 1})


def build_antiderivative(spec: Spec, extra: int = 0):
    constant = S.Symbol("c")
    wanted = spec.of("derivative")[0].args[0]
    return (S.integrate(wanted, spec.var) + constant, [constant],
            {constant: 0})


def build_arithmetic(spec: Spec, extra: int = 0):
    first, step = S.symbols("a d")
    return (first + (spec.var - 1) * step, [first, step],
            {first: 0, step: 1})


def build_geometric(spec: Spec, extra: int = 0):
    first, ratio = S.symbols("a r")
    return (first * ratio ** (spec.var - 1), [first, ratio],
            {first: 1, ratio: 2})


BASE = (
    Form("roots", "polynomial", ("roots-given",), build_roots,
         "a number times a factor for each root", from_data=True),
    Form("general", "polynomial", (), build_general,
         "a polynomial with one unknown coefficient for each condition",
         grows=True),
    Form("roots-times", "polynomial", ("roots-given",), build_roots_times,
         "a factor for each root times a polynomial for the rest",
         grows=True, from_data=True),
    Form("arithmetic", "sequence", ("terms-given",), build_arithmetic,
         "a first term and a common difference"),
    Form("geometric", "sequence", ("terms-given",), build_geometric,
         "a first term and a common ratio"),
    Form("polynomial", "sequence", (), build_general,
         "a polynomial in n with one unknown for each condition",
         grows=True),
    Form("power", "function", (), build_general,
         "a polynomial in x with one unknown for each condition",
         grows=True),
    Form("exponential", "function", (), build_exponential,
         "a number times e to a multiple of x"),
    Form("oscillation", "function", (), build_oscillation,
         "a sine and a cosine of one frequency"),
    Form("antiderivative", "function", ("derivative-given",),
         build_antiderivative,
         "the integral of the derivative asked for, plus a constant",
         from_data=True),
)


def _sum(first: Form, second: Form) -> Form:
    """A form that is the sum of two: its unknowns are both forms', the
    second's renamed so the two do not share any."""
    def build(spec: Spec, extra: int = 0):
        one, holes, defaults = first.build(spec, 0)
        two, others, more = second.build(spec, 0)
        renamed = {hole: S.Symbol(f"{hole.name}2") for hole in others}
        return (one + two.subs(renamed), holes + list(renamed.values()),
                {**defaults, **{renamed[hole]: value
                                for hole, value in more.items()}})
    return Form(f"{first.name}+{second.name}", first.kind,
                tuple(sorted(set(first.needs) | set(second.needs))), build,
                f"an {first.name} {first.kind} plus an {second.name} "
                f"{second.kind}" if first.name[0] in "aeiou" else
                f"a {first.name} {first.kind} plus a {second.name} "
                f"{second.kind}",
                parts=(first.name, second.name))


def _linear(form: Form) -> bool:
    """Whether the form is linear in its unknowns: then a sum of two of it
    is only itself again (a line plus a line is a line), and says nothing
    new. Worked out, not listed: every second derivative in the unknowns
    is zero."""
    from research.v693.spec import C, Spec
    probe = Spec(form.kind, {
        "polynomial": (C("root", 1), C("root", 2)),
        "sequence": (C("term", 1, 1), C("term", 2, 2)),
        "function": (C("value", 0, 1), C("value", 1, 2))}[form.kind])
    obj, holes, _ = form.build(probe, 0)
    return all(S.diff(obj, one, other) == 0 for one in holes
               for other in holes)


#: The most unknowns a composed form may have. A sum of two oscillations
#: has six, in a system of powers of two frequencies, and sympy takes
#: longer to solve it than the rest of a design put together.
MOST_UNKNOWNS = 5


def _unknowns(form: Form) -> int:
    from research.v693.spec import C, Spec
    probe = Spec(form.kind, {
        "polynomial": (C("root", 1), C("root", 2)),
        "sequence": (C("term", 1, 1), C("term", 2, 2)),
        "function": (C("value", 0, 1), C("value", 1, 2))}[form.kind])
    return len(form.build(probe, 0)[1])


def composed(forms) -> tuple:
    """Forms made from forms: every sum of two of a kind's forms, except a
    growing form (a polynomial plus anything is found by the polynomial
    growing) and a form linear in its unknowns summed with itself. Nothing
    lists these by hand; the table grows when the base does."""
    out = []
    for index, first in enumerate(forms):
        for second in forms[index:]:
            if first.kind != second.kind or first.grows or second.grows:
                continue
            if first.from_data or second.from_data:
                continue
            if _unknowns(first) + _unknowns(second) > MOST_UNKNOWNS:
                continue
            if first.name == second.name and _linear(first):
                continue
            out.append(_sum(first, second))
    return tuple(out)


FORMS = BASE + composed(BASE)
BY_NAME = {one.name: one for one in FORMS}
