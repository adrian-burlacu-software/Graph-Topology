"""What mathematics the subject covers: branches, kinds, and what is asked.

One table, read by everything else. The corpus that teaches the encoder is
drawn from it (`corpus.py`), the semantics answer from its kinds
(`semantics.py`), and the evaluation is organised by its branches. Adding a
branch is adding rows here -- and it is what "systematic" means: every
branch sympy can do exactly is on the list, with the kinds WordNet names in
it and the questions people put about them.

**Kinds** are where the store and computation meet. `prime number.n.01` is
in the store, with its gloss and its place under `integer`; whether 91 is
one is not a fact anybody could store, it is a computation. So a kind here
is a store concept (or, where WordNet has none -- *even number* -- the one
it is a kind of) with the test that decides membership. The tests are the
exact part and are never learned, like `v691/numbers.py`.

**Acts** are what an utterance asks of mathematics, and **templates** the
ways it is asked. Slots name what fills them: `{E}` an expression, `{EQ}` an
equation, `{V}` a variable, `{A}`/`{B}` numbers, `{L}` a list, `{M}` a
matrix, `{S}`/`{T}` sets, `{K}` a kind. What fills a slot is said by
`saying.py`, and the encoder learns to read it back.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import sympy as S
from sympy.logic.inference import satisfiable

x, y, z, t, n, a, b = S.symbols("x y z t n a b")
VARIABLES = (x, y, z, t, n, a, b)


def _integer(obj) -> bool:
    return bool(getattr(obj, "is_integer", False)) and obj.is_number


def _is(test):
    """A test that is false, not an error, for what it does not apply to."""
    def safe(obj) -> bool:
        try:
            return bool(test(obj))
        except Exception:                            # noqa: BLE001
            return False
    return safe


def _terms(obj) -> int:
    return len(S.Add.make_args(S.expand(obj)))


def _degree(obj) -> int:
    free = obj.free_symbols
    return S.Poly(obj, *free).total_degree() if free else 0


def _boolean(obj) -> bool:
    return isinstance(obj, (S.And, S.Or, S.Not, S.Implies, S.Symbol))


def _lhs(obj):
    return obj.lhs - obj.rhs if isinstance(obj, S.Equality) else None


@dataclass(frozen=True)
class Kind:
    """A kind of mathematical thing and how to tell a member."""

    name: str
    #: how it is said, the name first
    said: tuple
    #: the store's concept, or "" where WordNet has none
    concept: str
    #: the store concept it is a kind of, where it has none of its own
    under: str
    #: what it is a kind of: number, integer, expression, equation, matrix,
    #: set
    of: str
    holds: Callable


KINDS = (
    Kind("integer", ("an integer", "a whole number", "integer"),
         "integer.n.01", "", "number", _is(_integer)),
    Kind("natural number", ("a natural number", "natural"),
         "natural number.n.01", "", "number",
         _is(lambda o: _integer(o) and o > 0)),
    Kind("prime number", ("prime", "a prime number", "a prime"),
         "prime number.n.01", "", "number",
         _is(lambda o: _integer(o) and S.isprime(int(o)))),
    Kind("composite number", ("composite", "a composite number"),
         "composite number.n.01", "", "number",
         _is(lambda o: _integer(o) and o > 1 and not S.isprime(int(o)))),
    Kind("even number", ("even", "an even number"), "", "integer.n.01",
         "number", _is(lambda o: _integer(o) and o % 2 == 0)),
    Kind("odd number", ("odd", "an odd number"), "", "integer.n.01",
         "number", _is(lambda o: _integer(o) and o % 2 == 1)),
    Kind("positive number", ("positive", "a positive number"), "",
         "real number.n.01", "number", _is(lambda o: o.is_positive)),
    Kind("negative number", ("negative", "a negative number"), "",
         "real number.n.01", "number", _is(lambda o: o.is_negative)),
    Kind("rational number", ("rational", "a rational number"),
         "rational number.n.01", "", "number", _is(lambda o: o.is_rational)),
    Kind("irrational number", ("irrational", "an irrational number"),
         "irrational number.n.01", "", "number",
         _is(lambda o: o.is_irrational)),
    Kind("real number", ("real", "a real number"), "real number.n.01", "",
         "number", _is(lambda o: o.is_real)),
    Kind("complex number", ("complex", "a complex number"),
         "complex number.n.01", "", "number",
         _is(lambda o: o.is_complex)),
    Kind("perfect square", ("a perfect square", "a square number"), "",
         "integer.n.01", "number",
         _is(lambda o: _integer(o) and o >= 0
             and S.sqrt(o).is_integer)),
    Kind("perfect cube", ("a perfect cube", "a cube number"), "",
         "integer.n.01", "number",
         _is(lambda o: _integer(o) and S.cbrt(abs(o)).is_integer)),
    Kind("perfect number", ("a perfect number", "perfect"), "",
         "natural number.n.01", "number",
         _is(lambda o: _integer(o) and o > 0
             and S.divisor_sigma(int(o)) == 2 * o)),
    Kind("fibonacci number", ("a fibonacci number",),
         "fibonacci number.n.01", "", "number",
         _is(lambda o: _integer(o) and o >= 0 and any(
             S.fibonacci(k) == o for k in range(0, 60)))),
    Kind("polynomial", ("a polynomial",), "polynomial.n.01", "",
         "expression", _is(lambda o: o.free_symbols
                           and o.is_polynomial(*o.free_symbols))),
    Kind("monomial", ("a monomial",), "", "polynomial.n.01", "expression",
         _is(lambda o: o.free_symbols and o.is_polynomial()
             and _terms(o) == 1)),
    Kind("binomial", ("a binomial",), "binomial.n.01", "", "expression",
         _is(lambda o: o.free_symbols and o.is_polynomial()
             and _terms(o) == 2)),
    Kind("trinomial", ("a trinomial",), "", "polynomial.n.01", "expression",
         _is(lambda o: o.free_symbols and o.is_polynomial()
             and _terms(o) == 3)),
    Kind("linear equation", ("a linear equation", "linear"),
         "linear equation.n.01", "", "equation",
         _is(lambda o: _degree(_lhs(o)) == 1)),
    Kind("quadratic equation", ("a quadratic equation", "quadratic"),
         "quadratic equation.n.01", "", "equation",
         _is(lambda o: _degree(_lhs(o)) == 2)),
    Kind("square matrix", ("a square matrix", "square"), "", "matrix.n.01",
         "matrix", _is(lambda o: o.is_square)),
    Kind("symmetric matrix", ("symmetric", "a symmetric matrix"), "",
         "matrix.n.01", "matrix", _is(lambda o: o.is_symmetric())),
    Kind("invertible matrix", ("invertible", "an invertible matrix"), "",
         "matrix.n.01", "matrix", _is(lambda o: o.is_square
                                      and o.det() != 0)),
    Kind("empty set", ("empty", "the empty set"), "", "set.n.02", "set",
         _is(lambda o: o == S.EmptySet)),
    Kind("tautology", ("a tautology", "always true"), "tautology.n.01", "",
         "proposition", _is(lambda o: _boolean(o)
                            and not satisfiable(S.Not(o)))),
    Kind("contradiction", ("a contradiction", "never true"),
         "contradiction.n.02", "", "proposition",
         _is(lambda o: _boolean(o) and not satisfiable(o))),
    Kind("satisfiable", ("satisfiable", "sometimes true"), "",
         "proposition.n.01", "proposition",
         _is(lambda o: _boolean(o) and bool(satisfiable(o)))),
    Kind("arithmetic sequence", ("an arithmetic sequence",
                                 "an arithmetic progression", "arithmetic"),
         "arithmetic progression.n.01", "", "sequence",
         _is(lambda o: len(o) > 2 and len({o[i + 1] - o[i]
                                            for i in range(len(o) - 1)})
             == 1)),
    Kind("geometric sequence", ("a geometric sequence",
                                "a geometric progression", "geometric"),
         "geometric progression.n.01", "", "sequence",
         _is(lambda o: len(o) > 2 and 0 not in o
             and len({o[i + 1] / o[i] for i in range(len(o) - 1)}) == 1)),
)
BY_NAME = {one.name: one for one in KINDS}


def kind_said(words: str) -> Kind | None:
    """The kind a phrase names, as the curriculum says it: `prime`, `a
    prime number`, `an even number`."""
    plain = " ".join(words.lower().split())
    for kind in KINDS:
        bare = {one.split(" ", 1)[1] if one.split()[0] in ("a", "an", "the")
                and " " in one else one for one in kind.said}
        if plain == kind.name or plain in kind.said or plain in bare or                 plain.rstrip("s") == kind.name:
            return kind
    return None


# -- what fills a slot --------------------------------------------------------

def integer(rng, low=-20, high=100):
    return S.Integer(rng.randint(low, high))


def natural(rng, high=200):
    return S.Integer(rng.randint(1, high))


def rational(rng):
    bottom = rng.choice((2, 3, 4, 5, 6, 8, 10, 12))
    top = rng.randint(-11, 11) or 1
    return S.Rational(top, bottom)


def variable(rng):
    return rng.choice(VARIABLES[:4]) if rng.random() < 0.78 else rng.choice(
        VARIABLES[4:])


def polynomial(rng, var=None, degree=None):
    var = var if var is not None else variable(rng)
    degree = degree if degree is not None else rng.choice((1, 2, 2, 3))
    out = S.Integer(0)
    for power in range(degree + 1):
        coefficient = rng.randint(-9, 9)
        if power == degree and coefficient == 0:
            coefficient = rng.choice((1, 1, 2, 3, -1))
        out += coefficient * var ** power
    return out


def factored(rng, var=None):
    """A product of linear factors with small integer roots: what `expand`
    and `factor` are asked about."""
    var = var if var is not None else variable(rng)
    count = rng.choice((2, 2, 3))
    out = S.Integer(1)
    for _ in range(count):
        out *= (rng.choice((1, 1, 1, 2)) * var - rng.randint(-6, 6))
    return out


def arithmetic(rng):
    """A sum as a person writes one, left unevaluated: `17 * 4 + 3`."""
    ops = rng.choice((1, 1, 2))
    out = integer(rng, 1, 99)
    for _ in range(ops):
        other = integer(rng, 1, 30)
        choice = rng.choice(("+", "-", "*", "/"))
        if choice == "+":
            out = S.Add(out, other, evaluate=False)
        elif choice == "-":
            out = S.Add(out, S.Mul(-1, other, evaluate=False),
                        evaluate=False)
        elif choice == "*":
            out = S.Mul(out, other, evaluate=False)
        else:
            out = S.Mul(out, S.Pow(other, -1, evaluate=False),
                        evaluate=False)
    return out


def special(rng):
    """Values with exact answers: roots, trigonometry at the known angles,
    powers, factorials, logarithms of powers of e."""
    choice = rng.randrange(6)
    if choice == 0:
        return S.sqrt(S.Integer(rng.choice((2, 3, 4, 9, 16, 25, 49, 50,
                                            72, 81, 100))),
                      evaluate=False)
    if choice == 1:
        angle = S.pi / rng.choice((2, 3, 4, 6))
        func = rng.choice((S.sin, S.cos, S.tan))
        return func(angle, evaluate=False)
    if choice == 2:
        return S.Pow(S.Integer(rng.randint(2, 12)), rng.randint(2, 4),
                     evaluate=False)
    if choice == 3:
        return S.factorial(S.Integer(rng.randint(3, 8)), evaluate=False)
    if choice == 4:
        return S.log(S.exp(S.Integer(rng.randint(1, 5)), evaluate=False),
                     evaluate=False)
    return S.Add(rational(rng), rational(rng), evaluate=False)


def equation(rng, var=None):
    var = var if var is not None else variable(rng)
    if rng.random() < 0.5:
        root = rng.randint(-10, 10)
        coefficient = rng.choice((1, 2, 3, 4, 5, -2))
        shift = rng.randint(-15, 15)
        return S.Eq(coefficient * var + shift, coefficient * root + shift)
    first, second = rng.randint(-8, 8), rng.randint(-8, 8)
    return S.Eq(S.expand((var - first) * (var - second)), 0)


def numbers(rng, low=1, high=40, count=None):
    count = count or rng.randint(3, 7)
    return [S.Integer(rng.randint(low, high)) for _ in range(count)]


def matrix(rng):
    size = rng.choice((2, 2, 3))
    return S.Matrix(size, size, lambda i, j: rng.randint(-5, 9))


def finite_set(rng):
    return S.FiniteSet(*sorted(rng.sample(range(0, 15), rng.randint(2, 5))))


def number_like(rng):
    choice = rng.random()
    if choice < 0.6:
        return natural(rng, 150)
    if choice < 0.75:
        return integer(rng, -30, 30)
    if choice < 0.85:
        return rational(rng)
    return S.sqrt(S.Integer(rng.choice((2, 3, 4, 5, 9, 16))))


p, q, r = S.symbols("p q r")
PROPOSITIONAL = (p, q, r)


def proposition(rng, depth=2):
    """A proposition in p, q and r -- some always true (`p or not p`),
    some never, most sometimes."""
    if rng.random() < 0.25:
        one, other = rng.sample(PROPOSITIONAL, 2)
        return rng.choice((one | ~one, one & ~one,
                           S.Implies(one & other, one),
                           S.Implies(one, one | other), ~(one & ~one),
                           (one | other) & ~one & ~other))
    if depth == 0 or rng.random() < 0.3:
        atom = rng.choice(PROPOSITIONAL)
        return ~atom if rng.random() < 0.3 else atom
    left, right = (proposition(rng, depth - 1),
                   proposition(rng, depth - 1))
    if left == right:
        return left
    return rng.choice((S.And, S.Or, S.Or, S.Implies))(left, right)


def inequality(rng, var=None):
    var = var if var is not None else variable(rng)
    coefficient = rng.choice((1, 2, 3, 4, 5, -2, -3))
    shift = rng.randint(-12, 12)
    bound = rng.randint(-15, 25)
    relation = rng.choice((S.Lt, S.Gt, S.Le, S.Ge))
    return relation(coefficient * var + shift, bound)


def system(rng):
    """Two equations in x and y with a whole solution."""
    first, second = rng.randint(-6, 9), rng.randint(-6, 9)
    out = []
    while len(out) < 2:
        a_, b_ = rng.randint(-4, 5) or 1, rng.randint(-4, 5) or 1
        equation = S.Eq(a_ * x + b_ * y, a_ * first + b_ * second)
        if out and S.Matrix([[out[0].lhs.coeff(x), out[0].lhs.coeff(y)],
                             [a_, b_]]).det() == 0:
            continue
        out.append(equation)
    return out


def sequence(rng):
    """The first terms of a sequence someone might ask to continue."""
    count = rng.randint(4, 6)
    choice = rng.random()
    if choice < 0.45:
        start, step = rng.randint(-10, 20), rng.choice(
            (1, 2, 3, 4, 5, 6, 7, 10, -2, -3, -5))
        return [S.Integer(start + step * k) for k in range(count)]
    if choice < 0.7:
        start, ratio = rng.randint(1, 6), rng.choice((2, 2, 3, 3, 5, 10, -2))
        return [S.Integer(start * ratio ** k) for k in range(count)]
    offset = rng.randint(0, 3)
    power = rng.choice((2, 2, 3))
    count = max(count, power + 2)
    return [S.Integer((k + offset + 1) ** power) for k in range(count)]


def summand(rng, var):
    return rng.choice((1 / var ** 2, var, var ** 2, S.Rational(1, 2) ** var,
                       1 / (var * (var + 1)), 2 * var - 1, var ** 3,
                       S.Rational(1, 3) ** var))


#: Units a length, a mass, a time or a volume is said in.
UNIT_KINDS = {"length": ("meter", "centimeter", "millimeter", "kilometer",
                         "inch", "foot", "yard", "mile"),
              "mass": ("gram", "kilogram", "pound"),
              "time": ("second", "minute", "hour", "day"),
              "volume": ("liter", "milliliter")}


def unit(name: str):
    from sympy.physics import units
    return getattr(units, name)


def amount(rng, kind=None):
    """`5 kilometers`: an amount of something in a unit."""
    kind = kind or rng.choice(tuple(UNIT_KINDS))
    name = rng.choice(UNIT_KINDS[kind])
    number = (S.Integer(rng.randint(1, 500)) if rng.random() < 0.75
              else decimal(rng, 1))
    return number * unit(name)


def unit_like(rng, quantity):
    """Another unit of what an amount measures."""
    from sympy.physics.units import Quantity
    had = {str(one.name) for one in quantity.atoms(Quantity)}
    names = [one for group in UNIT_KINDS.values() if had & set(group)
             for one in group if one not in had]
    return unit(rng.choice(names))


def decimal(rng, places=None):
    places = places if places is not None else rng.choice((1, 2, 3, 4))
    whole = rng.randint(0, 999)
    part = rng.randint(1, 10 ** places - 1)
    return S.Float(f"{whole}.{part:0{places}d}")


def vector(rng, size=None):
    size = size or rng.choice((2, 3, 3))
    return S.Tuple(*[S.Integer(rng.randint(-6, 9)) for _ in range(size)])


def complex_number(rng):
    return S.Integer(rng.randint(-9, 9)) + (rng.randint(1, 9) * rng.choice(
        (1, -1))) * S.I


# -- what is asked, and how ---------------------------------------------------

@dataclass(frozen=True)
class Act:
    name: str
    branch: str
    templates: tuple
    #: slot -> what fills it
    fills: dict


def _fills(**slots):
    return slots


ACTS = (
    Act("value", "arithmetic", (
        "what is {E}", "{E}", "calculate {E}", "work out {E}",
        "how much is {E}", "what's {E}", "evaluate {E}", "compute {E}",
        "what do you get for {E}", "{E} = ?", "what does {E} equal"),
        _fills(E=lambda rng: arithmetic(rng) if rng.random() < 0.6
               else special(rng))),
    Act("simplify", "algebra", (
        "simplify {E}", "what is {E} simplified", "can you simplify {E}",
        "write {E} more simply"),
        _fills(E=lambda rng: S.Add(polynomial(rng), polynomial(rng),
                                   evaluate=False))),
    Act("expand", "algebra", (
        "expand {E}", "multiply out {E}", "expand the brackets in {E}",
        "what is {E} expanded"),
        _fills(E=lambda rng: factored(rng))),
    Act("factor", "algebra", (
        "factor {E}", "factorise {E}", "factorize {E}",
        "what are the factors of {E}", "write {E} as a product"),
        _fills(E=lambda rng: S.expand(factored(rng)))),
    Act("solve", "algebra", (
        "solve {EQ}", "solve {EQ} for {V}", "what is {V} if {EQ}",
        "find {V} when {EQ}", "{EQ} , what is {V}", "solve for {V} : {EQ}",
        "what value of {V} makes {EQ} true", "if {EQ} what is {V}"),
        _fills(EQ="equation", V="variable")),
    Act("derivative", "calculus", (
        "differentiate {E}", "what is the derivative of {E}",
        "the derivative of {E} with respect to {V}",
        "differentiate {E} with respect to {V}",
        "find the derivative of {E}", "what is the rate of change of {E}"),
        _fills(E="polynomial or function", V="variable")),
    Act("integral", "calculus", (
        "integrate {E}", "what is the integral of {E}",
        "integrate {E} with respect to {V}",
        "find the antiderivative of {E}",
        "integrate {E} from {A} to {B}",
        "what is the integral of {E} from {A} to {B}"),
        _fills(E="polynomial or function", V="variable", A="bound",
               B="bound")),
    Act("limit", "calculus", (
        "what is the limit of {E} as {V} approaches {A}",
        "the limit as {V} goes to {A} of {E}",
        "find the limit of {E} as {V} tends to {A}"),
        _fills(E="rational function", V="variable", A="point")),
    Act("at", "algebra", (
        "what is {E} when {V} is {A}", "evaluate {E} at {V} = {A}",
        "what is {E} if {V} = {A}", "find {E} for {V} = {A}",
        "substitute {V} = {A} into {E}"),
        _fills(E="polynomial", V="variable", A="small integer")),
    Act("gcd", "number theory", (
        "what is the greatest common divisor of {L}", "gcd of {L}",
        "what is the highest common factor of {L}", "the hcf of {L}",
        "greatest common factor of {L}"),
        _fills(L="two or three naturals")),
    Act("lcm", "number theory", (
        "what is the least common multiple of {L}", "lcm of {L}",
        "what is the lowest common multiple of {L}",
        "smallest common multiple of {L}"),
        _fills(L="two or three naturals")),
    Act("divisors", "number theory", (
        "what are the factors of {A}", "list the divisors of {A}",
        "what divides {A}", "what are the divisors of {A}"),
        _fills(A="natural")),
    Act("prime factors", "number theory", (
        "what are the prime factors of {A}",
        "what is the prime factorisation of {A}",
        "break {A} into prime factors", "prime factorization of {A}"),
        _fills(A="natural")),
    Act("is kind", "number theory", (
        "is {A} {K}", "is {A} a {K}", "{A} is {K} , is it not",
        "can {A} be {K}", "is {E} {K}", "is the number {A} {K}",
        "is {E} a {K}", "is {L} {K}", "is {L} a {K}",
        "is the sequence {L} {K}"),
        _fills(A="number", K="kind", E="expression")),
    Act("divides", "number theory", (
        "is {A} divisible by {B}", "does {B} divide {A}",
        "is {B} a factor of {A}", "is {A} a multiple of {B}",
        "can {A} be divided by {B}"),
        _fills(A="natural", B="small natural")),
    Act("check", "arithmetic", (
        "is {R}", "is it true that {R}", "true or false : {R}",
        "check {R}"),
        _fills(R="relation")),
    Act("mean", "statistics", (
        "what is the mean of {L}", "the average of {L}",
        "find the mean of {L}", "what is the average of {L}"),
        _fills(L="list")),
    Act("median", "statistics", (
        "what is the median of {L}", "find the median of {L}",
        "the median of {L}"),
        _fills(L="list")),
    Act("mode", "statistics", (
        "what is the mode of {L}", "find the mode of {L}",
        "which value is most common in {L}"),
        _fills(L="list")),
    Act("range", "statistics", (
        "what is the range of {L}", "find the range of {L}"),
        _fills(L="list")),
    Act("sum", "statistics", (
        "what is the sum of {L}", "add up {L}", "the total of {L}",
        "what is the total of {L}"),
        _fills(L="list")),
    Act("choose", "combinatorics", (
        "{A} choose {B}", "how many ways can you choose {B} from {A}",
        "in how many ways can {B} things be chosen from {A}",
        "how many combinations of {B} from {A}"),
        _fills(A="small natural", B="smaller natural")),
    Act("arrange", "combinatorics", (
        "how many ways can {A} things be arranged",
        "how many permutations of {A} things are there",
        "in how many orders can {A} people stand in a line",
        "how many ways are there to arrange {A} books on a shelf"),
        _fills(A="small natural")),
    Act("determinant", "linear algebra", (
        "what is the determinant of {M}", "det of {M}",
        "find the determinant of {M}"),
        _fills(M="matrix")),
    Act("inverse", "linear algebra", (
        "what is the inverse of {M}", "invert {M}",
        "find the inverse of {M}"),
        _fills(M="matrix")),
    Act("transpose", "linear algebra", (
        "what is the transpose of {M}", "transpose {M}"),
        _fills(M="matrix")),
    Act("union", "sets", (
        "what is the union of {S} and {T}", "{S} union {T}",
        "combine the sets {S} and {T}"),
        _fills(S="set", T="set")),
    Act("intersection", "sets", (
        "what is the intersection of {S} and {T}", "{S} intersect {T}",
        "what do {S} and {T} have in common"),
        _fills(S="set", T="set")),
    Act("subset", "sets", (
        "is {S} a subset of {T}", "is {S} contained in {T}",
        "is every element of {S} in {T}"),
        _fills(S="set", T="set")),
    Act("let", "algebra", (
        "let {V} be {A}", "let {V} = {E}", "suppose {V} is {A}",
        "{V} = {E}", "set {V} to {A}", "say {V} equals {A}",
        "assume {V} = {A}", "let {F} = {E}", "define {F} = {E}",
        "{F} = {E}", "let {F} be {E}", "suppose {F} = {E}"),
        _fills(V="variable", A="number", E="expression")),
)
ACTS += (
    Act("solve system", "algebra", (
        "solve {EQ} and {T}", "solve the simultaneous equations {EQ} and {T}",
        "if {EQ} and {T} , what are x and y", "find x and y if {EQ} and {T}",
        "{EQ} , {T} . solve for x and y"),
        _fills(EQ="equation in x and y", T="another")),
    Act("measure", "geometry", (
        "what is the {W} of a {H} with {G}", "find the {W} of a {H} with {G}",
        "a {H} has {G} . what is its {W}", "what is the {W} of a {H} whose {G}",
        "a {H} with {G} , what is the {W}", "if a {H} has {G} , what is its {W}",
        "calculate the {W} of a {H} with {G}", "what is the {W} of a {H}",
        "what is the {W} of a regular {H} with {G}",
        "work out the {W} of a {H} of {G}"),
        _fills(H="shape", W="what is wanted", G="what is given")),
    Act("given", "geometry", (
        "{G}", "the {G}", "its {G}", "it has {G}", "the {H} has {G}"),
        _fills(G="what is given", H="shape")),
    Act("convert", "measures", (
        "convert {A} to {U}", "how many {U} are in {A}", "what is {A} in {U}",
        "{A} in {U}", "change {A} into {U}", "express {A} in {U}",
        "how many {U} is {A}"),
        _fills(A="an amount", U="a unit of the same")),
    Act("next term", "sequences", (
        "what is the next term in {L}", "what comes next : {L}",
        "continue the sequence {L}", "what comes next in {L}",
        "what is the next number in the sequence {L}",
        "{L} , what comes next"),
        _fills(L="sequence")),
    Act("nth term", "sequences", (
        "what is the nth term of {L}", "find a formula for the sequence {L}",
        "what is the general term of {L}", "give the nth term of {L}",
        "what is the rule for {L}"),
        _fills(L="sequence")),
    Act("term", "sequences", (
        "what is the {N} term of {L}", "find the {N} term of the sequence {L}",
        "in the sequence {L} what is the {N} term",
        "what is term {A} of {L}"),
        _fills(N="ordinal", A="place", L="sequence")),
    Act("series", "sequences", (
        "what is the sum of {E} for {V} from {A} to {B}",
        "sum {E} from {V} = {A} to {B}",
        "the sum of {E} as {V} goes from {A} to {B}",
        "add up {E} for {V} from {A} to {B}",
        "what does the series {E} from {V} = {A} to {B} add up to"),
        _fills(E="term", V="variable", A="start", B="end or infinity")),
    Act("equivalent", "logic", (
        "is {E} equivalent to {T}", "are {E} and {T} equivalent",
        "is {E} the same as {T}", "does {E} mean the same as {T}"),
        _fills(E="proposition", T="proposition")),
    Act("dice", "probability", (
        "what is the probability of rolling a {B} with a die",
        "what is the chance of rolling a {B}",
        "what is the probability of rolling a total of {B} with {A} dice",
        "if you roll {A} dice what is the chance of a total of {B}",
        "probability of rolling {B} with {A} dice",
        "what are the odds of rolling a {B} on a die"),
        _fills(A="dice", B="total")),
    Act("coins", "probability", (
        "what is the probability of getting {B} heads in {A} coin tosses",
        "if you toss {A} coins what is the chance of {B} heads",
        "probability of {B} heads in {A} flips",
        "what is the probability of exactly {B} heads when {A} coins are "
        "flipped", "flip a coin {A} times , what is the chance of {B} heads"),
        _fills(A="coins", B="heads")),
    Act("round places", "arithmetic", (
        "round {A} to {B} decimal places", "what is {A} to {B} decimal places",
        "round {A} to {B} places", "{A} rounded to {B} decimal places"),
        _fills(A="decimal", B="places")),
    Act("round nearest", "arithmetic", (
        "round {A} to the nearest {B}", "what is {A} to the nearest {B}",
        "{A} rounded to the nearest {B}", "round {A} to the nearest whole number",
        "round {A}"),
        _fills(A="decimal or number", B="10 100 or 1000")),
    Act("part of", "arithmetic", (
        "what is {E} of {A}", "how much is {E} of {A}",
        "what is {E} of {A} ?", "find {E} of {A}",
        "what do you get for {E} of {A}"),
        _fills(E="a fraction", A="an amount")),
    Act("percent of", "arithmetic", (
        "what is {A} percent of {B}", "{A} % of {B}", "find {A} percent of {B}",
        "how much is {A} percent of {B}", "calculate {A} % of {B}",
        "what is {A} % of {B}"),
        _fills(A="percent", B="amount")),
    Act("as percent", "arithmetic", (
        "what is {A} as a percentage of {B}", "what percent of {B} is {A}",
        "{A} out of {B} as a percentage", "what percentage is {A} of {B}",
        "express {A} out of {B} as a percent"),
        _fills(A="part", B="whole")),
    Act("percent change", "arithmetic", (
        "what is the percentage change from {A} to {B}",
        "by what percent does {A} change to {B}",
        "percent change from {A} to {B}",
        "what is the percent increase from {A} to {B}",
        "it went from {A} to {B} , what is the percentage change"),
        _fills(A="before", B="after")),
    Act("dot", "linear algebra", (
        "what is the dot product of {S} and {T}", "{S} dot {T}",
        "find the dot product of {S} and {T}",
        "what is the scalar product of {S} and {T}"),
        _fills(S="vector", T="vector")),
    Act("cross", "linear algebra", (
        "what is the cross product of {S} and {T}", "{S} cross {T}",
        "find the cross product of {S} and {T}"),
        _fills(S="vector", T="vector")),
    Act("magnitude", "linear algebra", (
        "what is the magnitude of {S}", "how long is the vector {S}",
        "what is the length of the vector {S}", "find the magnitude of {S}",
        "what is the norm of {S}"),
        _fills(S="vector")),
    Act("modulus", "complex numbers", (
        "what is the modulus of {E}", "what is the absolute value of {E}",
        "find the modulus of {E}", "how far is {E} from zero"),
        _fills(E="complex number")),
    Act("conjugate", "complex numbers", (
        "what is the conjugate of {E}", "find the complex conjugate of {E}"),
        _fills(E="complex number")),
)
BY_ACT = {one.name: one for one in ACTS}

#: The branches, in the order they are taught, and what each covers.
BRANCHES = {
    "arithmetic": "numbers, the four operations, fractions, decimals, "
                  "powers, roots, percentages, rounding, order of "
                  "operations, checking equalities",
    "number theory": "primes, factors and multiples, divisibility, gcd and "
                     "lcm, parity, perfect squares and cubes",
    "algebra": "expressions, simplifying, expanding, factorising, "
               "equations, inequalities, simultaneous equations, "
               "substitution, variables, functions",
    "calculus": "derivatives, integrals, limits",
    "statistics": "mean, median, mode, range, totals",
    "combinatorics": "choosing, arranging",
    "linear algebra": "matrices: determinant, inverse, transpose; vectors: dot and cross products, magnitude",
    "sets": "union, intersection, subsets",
    "geometry": "circles, polygons, triangles and solids: lengths, areas, "
                "volumes and angles, worked out by planning from what is "
                "given, asking for what is missing",
    "measures": "units of length, mass, time and volume, converted",
    "sequences": "next terms, nth terms, terms by place, series",
    "logic": "propositions: tautologies, contradictions, equivalence",
    "probability": "dice and coins",
    "complex numbers": "modulus, conjugate, arithmetic with i",
}
