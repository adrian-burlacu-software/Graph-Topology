"""Design goals made from objects: what the proposer is taught from.

The same trick as v692's corpus, turned round. There, an object was made
and said aloud, so every word's meaning was known by construction. Here an
object is made and **true things are said of it** -- some of its roots,
its value at a few points, a turning point it has, how many terms of it
add up to what -- and those clauses are a specification it certainly
meets. So every goal made here can be designed, and nobody had to write
it.

What is taught is not the object it came from. A goal made from 3·2ⁿ⁻¹
with two of its terms given is met just as well by a line, and the
proposer should learn that. So each goal is labelled by **what fitting
every form to it came to** (`designing.fit`): exact and checked.

And by which of those is **simplest**: the form with the fewest unknowns
(`designing.simplest`). A polynomial through four points fits any four
terms, so *fits* alone would teach a proposer to reach for it every time;
for 2, 6, 18, 54 the design anyone wants is 2·3ⁿ⁻¹, two unknowns, not a
cubic with four. The fewest unknowns is Occam's razor, said exactly.
"""
from __future__ import annotations

import random

import sympy as S

from research.v693.spec import C, Spec, n, x

SMALL_ROOTS = (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, S.Rational(1, 2),
               S.Rational(-1, 2), S.Rational(3, 2))
SCALES = (1, 1, 1, 2, -1, 3, -2, S.Rational(1, 2))


def polynomial(rng: random.Random):
    """A polynomial someone might design: roots and a scale, or small
    coefficients."""
    if rng.random() < 0.6:
        count = rng.choice((1, 2, 2, 3, 3))
        out = S.Integer(rng.choice(SCALES))
        for _ in range(count):
            out *= (x - rng.choice(SMALL_ROOTS)) ** rng.choice((1, 1, 1, 2))
        return S.expand(out)
    degree = rng.choice((1, 2, 2, 3))
    out = sum(rng.randint(-5, 5) * x ** k for k in range(degree))
    return S.expand(out + rng.choice((1, 1, 2, -1, 3)) * x ** degree)


def summed(rng: random.Random):
    """A sum of two: what composed forms are for -- 2^n + 1, 3^n - n, a
    Fibonacci-like 2^n + 3^n."""
    ratio = rng.choice((2, 3, -2, 5))
    other = (rng.randint(-5, 5) + (n - 1) * rng.randint(-3, 3)
             if rng.random() < 0.6 else rng.choice((1, 2, -1)) *
             S.Integer(rng.choice((2, 3, 4))) ** (n - 1))
    return rng.choice((1, 2, 3, -1)) * S.Integer(ratio) ** (n - 1) + other


def sequence(rng: random.Random):
    choice = rng.random()
    if choice < 0.4:
        return rng.randint(-10, 20) + (n - 1) * rng.choice(
            (1, 2, 3, 4, 5, -2, -3, 7, 10))
    if choice < 0.75:
        return rng.choice((1, 2, 3, 5, -1, -2, 4)) * S.Integer(
            rng.choice((2, 2, 3, 3, -2, 5, 10))) ** (n - 1)
    return S.expand(rng.choice((1, 1, 2, -1)) * n ** rng.choice((2, 2, 3))
                    + rng.randint(-4, 4) * n + rng.randint(-5, 5))


def _roots(obj) -> list:
    """Its rational roots with their multiplicities."""
    return [(root, times) for root, times in S.roots(obj, x).items()
            if root.is_rational]


def polynomial_goal(rng: random.Random) -> Spec | None:
    obj = polynomial(rng)
    if obj.is_number:
        # A constant, or nothing: not a polynomial anyone designs.
        return None
    clauses = []
    roots = _roots(obj)
    rng.shuffle(roots)
    for root, times in roots[:rng.randint(0, len(roots))]:
        clauses.append(C("root", root, times) if times > 1 and
                       rng.random() < 0.7 else C("root", root))
    points = rng.sample(range(-3, 5), rng.randint(0, 4))
    for at in points:
        kind = rng.random()
        if kind < 0.7:
            clauses.append(C("value", at, obj.subs(x, at)))
        elif kind < 0.85:
            clauses.append(C("slope", at, S.diff(obj, x).subs(x, at)))
    turning = [one for one in S.roots(S.diff(obj, x), x)
               if one.is_rational]
    if turning and rng.random() < 0.3:
        clauses.append(C("stationary", rng.choice(turning)))
    poly = S.Poly(obj, x)
    if rng.random() < 0.3:
        clauses.append(C("leading", poly.LC()))
    if rng.random() < 0.35:
        clauses.append(C("degree", poly.degree()))
    if rng.random() < 0.15 and all(one.is_integer
                                   for one in poly.all_coeffs()):
        clauses.append(C("integer"))
    if sum(one.conditions() for one in clauses) < 2:
        return None
    rng.shuffle(clauses)
    return Spec("polynomial", clauses)


def sequence_goal(rng: random.Random) -> Spec | None:
    is_sum = rng.random() < 0.2
    obj = summed(rng) if is_sum else sequence(rng)
    clauses = []
    # A sum of two only says less than a polynomial through its terms when
    # there are five or more of them: given four, the polynomial ties, and
    # the proposer would never see a sum be the answer. So long runs are
    # made for every kind of sequence -- the proposer has to learn what
    # tells them apart, not that long runs are sums.
    # A sum is only told from a polynomial by a long run of its terms, so
    # a sum always gets one.
    long_run = is_sum or rng.random() < 0.3
    if long_run or rng.random() < 0.6:
        start = rng.randint(1, 3)
        places = list(range(start, start + (rng.randint(5, 7) if long_run
                                            else rng.randint(2, 5))))
    else:
        places = sorted(rng.sample(range(1, 9), rng.randint(2, 4)))
    for place in places:
        clauses.append(C("term", place, obj.subs(n, place)))
    if rng.random() < 0.2:
        count = rng.randint(3, 10)
        clauses.append(C("total", count, sum(obj.subs(n, k)
                                             for k in range(1, count + 1))))
    if any(abs(one.args[-1]) > 10 ** 6 for one in clauses):
        return None
    return Spec("sequence", clauses)


#: Derivatives a function might be asked to have, by what integrates them.
DERIVATIVES = (2 * x * S.cos(x ** 2), 3 * x ** 2 + 1, S.exp(x) + 1,
               S.cos(x), 2 * x * S.exp(x ** 2), 1 / x, S.sin(x),
               S.exp(2 * x), 4 * x ** 3, 2 * x + 3, 1 / (x + 1),
               S.cos(2 * x), x * S.exp(x))


def function_goal(rng: random.Random) -> Spec | None:
    """A function and what is true of it: the equation it satisfies, a
    derivative it has, and its value or slope somewhere."""
    choice = rng.random()
    clauses = []
    if choice < 0.3:
        rate = rng.choice((1, 2, 3, -1, -2, 5))
        scale = rng.choice((1, 2, 3, -1, 4))
        obj = scale * S.exp(rate * x)
        clauses.append(C("ode", 0, 1, -rate))
    elif choice < 0.55:
        frequency = rng.choice((1, 2, 3))
        sine, cosine = rng.choice(((1, 0), (0, 1), (2, 0), (0, 3), (1, 1),
                                   (3, -2)))
        obj = sine * S.sin(frequency * x) + cosine * S.cos(frequency * x)
        clauses.append(C("ode", 1, 0, frequency ** 2))
    elif choice < 0.8:
        wanted = rng.choice(DERIVATIVES)
        at = 1 if wanted.subs(x, 0).is_finite is False or wanted.has(
            1 / x) else 0
        obj = S.integrate(wanted, x) + rng.randint(-3, 3)
        clauses.append(C("derivative", wanted))
        clauses.append(C("value", at, obj.subs(x, at)))
    elif choice < 0.9:
        first, second = rng.sample((1, 2, 3, -1), 2)
        obj = S.exp(first * x) + S.exp(second * x)
        clauses.append(C("ode", 1, -(first + second), first * second))
    else:
        obj = polynomial(rng)
        for at in rng.sample(range(-2, 3), rng.randint(2, 3)):
            clauses.append(C("value", at, obj.subs(x, at)))
    if any(one.kind == "ode" for one in clauses):
        clauses.append(C("value", 0, obj.subs(x, 0)))
        if clauses[0].args[0] == 1 or rng.random() < 0.3:
            clauses.append(C("slope", 0, S.diff(obj, x).subs(x, 0)))
    if not all(S.sympify(one.args[-1]).is_finite for one in clauses
               if one.args and S.sympify(one.args[-1]).is_number):
        return None
    rng.shuffle(clauses)
    return Spec("function", clauses)


def goal(rng: random.Random) -> Spec | None:
    choice = rng.random()
    if choice < 0.45:
        return polynomial_goal(rng)
    if choice < 0.8:
        return sequence_goal(rng)
    return function_goal(rng)


def labelled(count: int, seed: int = 693) -> list:
    """(spec, the simplest form that fits it) for `count` goals made from
    objects (`designing.best_fit`)."""
    from research.v693 import designing
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        spec = goal(rng)
        if spec is None:
            continue
        out.append((spec, designing.best_fit(spec)))
    return out
