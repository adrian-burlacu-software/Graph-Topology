"""Questions of mathematics as people put them, and what the answers are.

    python -m research.v692.bank              # every branch
    python -m research.v692.bank --branch algebra -v

Hand-written, and written **before** the encoder taught from `corpus.py`
was run on any of them: other phrasings than the templates', other numbers,
some typed as symbols, some as words. Each is read by the encoder
(`reading.read`), done (`doing.do`), and scored three ways:

    read      the act is the one asked
    right     the answer is the expected one (a value, or yes/no)
    honest    right, or said not to be known -- never a wrong answer

Sequences share a workspace, so `let x be 4` and then `is x even` is one
item. The bank is not the curriculum's templates and must not become them:
what fails here is what the encoder did not generalise to.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import sympy as S

from research.v692 import doing, reading
from research.v692.symbols import Unreadable, parsed, same

x = S.Symbol("x")


@dataclass
class Item:
    branch: str
    said: tuple
    act: str
    #: the answer: a value to compare (as symbols), or yes/no
    expected: str
    got_act: str = ""
    got: str = ""
    right: bool = False
    honest: bool = False
    read: bool = False


def I(branch, *said, act, expected) -> Item:
    return Item(branch, said, act, expected)


BANK = [
    # arithmetic
    I("arithmetic", "what's 12 times 11", act="value", expected="132"),
    I("arithmetic", "how much is three hundred minus 45", act="value",
      expected="255"),
    I("arithmetic", "calculate 7 + 8 * 2", act="value", expected="23"),
    # `a fraction of an amount` is its own act since the branch grew
    # (`part of`); the answer it was written for is unchanged.
    I("arithmetic", "what is two thirds of 9", act="part of", expected="6"),
    I("arithmetic", "(4+6)/5", act="value", expected="2"),
    I("arithmetic", "work out 2 to the power of 8", act="value",
      expected="256"),
    I("arithmetic", "what is the square root of 144", act="value",
      expected="12"),
    I("arithmetic", "what is one half plus one quarter", act="value",
      expected="3/4"),
    I("arithmetic", "what does 6 factorial come to", act="value",
      expected="720"),
    I("arithmetic", "is 7 times 8 equal to 54", act="check", expected="no"),
    I("arithmetic", "is 15 + 27 = 42", act="check", expected="yes"),
    I("arithmetic", "what is sin of pi over 2", act="value", expected="1"),
    # number theory
    I("number theory", "is 97 prime", act="is kind", expected="yes"),
    I("number theory", "is 91 a prime number", act="is kind",
      expected="no"),
    I("number theory", "is 64 a perfect square", act="is kind",
      expected="yes"),
    I("number theory", "is 35 odd", act="is kind", expected="yes"),
    I("number theory", "is 1/2 an integer", act="is kind", expected="no"),
    I("number theory", "is 28 a perfect number", act="is kind",
      expected="yes"),
    I("number theory", "is 84 divisible by 7", act="divides",
      expected="yes"),
    I("number theory", "is 5 a factor of 52", act="divides", expected="no"),
    I("number theory", "what is the gcd of 48 and 36", act="gcd",
      expected="12"),
    I("number theory", "what's the lowest common multiple of 4 and 6",
      act="lcm", expected="12"),
    I("number theory", "what are the prime factors of 60",
      act="prime factors", expected="60"),
    I("number theory", "list the factors of 18", act="divisors",
      expected="1, 2, 3, 6, 9, 18"),
    # algebra
    I("algebra", "solve 3x + 4 = 19", act="solve", expected="5"),
    I("algebra", "solve x squared minus 9 equals 0", act="solve",
      expected="-3, 3"),
    I("algebra", "what is y if 2y - 3 = 11", act="solve", expected="7"),
    I("algebra", "factorise x^2 + 7x + 12", act="factor",
      expected="(x+3)*(x+4)"),
    I("algebra", "expand (x + 2)(x - 5)", act="expand",
      expected="x^2 - 3x - 10"),
    I("algebra", "simplify 3x + 2x - 4 + 9", act="simplify",
      expected="5x + 5"),
    I("algebra", "what is 2x + 1 when x is 6", act="at", expected="13"),
    I("algebra", "let a be 3", "what is a squared plus 1", act="value",
      expected="10"),
    I("algebra", "let n = 10", "is n even", act="is kind", expected="yes"),
    I("algebra", "is x^2 + 3x + 1 a polynomial", act="is kind",
      expected="yes"),
    # calculus
    I("calculus", "differentiate x cubed", act="derivative",
      expected="3x^2"),
    I("calculus", "what is the derivative of 5x^2 + 3x", act="derivative",
      expected="10x + 3"),
    I("calculus", "integrate 2x with respect to x", act="integral",
      expected="x^2"),
    I("calculus", "what is the integral of x from 0 to 2",
      act="integral", expected="2"),
    I("calculus", "find the limit of (x^2 - 4)/(x - 2) as x approaches 2",
      act="limit", expected="4"),
    I("calculus", "what is the derivative of sin x", act="derivative",
      expected="cos(x)"),
    # statistics
    I("statistics", "what is the mean of 4, 8 and 12", act="mean",
      expected="8"),
    I("statistics", "find the median of 3, 9, 1, 7 and 5", act="median",
      expected="5"),
    I("statistics", "what is the mode of 2, 3, 3, 5 and 3", act="mode",
      expected="3"),
    I("statistics", "what is the range of 10, 4, 17 and 8", act="range",
      expected="13"),
    I("statistics", "add up 5, 10, 15 and 20", act="sum", expected="50"),
    # combinatorics
    I("combinatorics", "how many ways can you choose 2 from 6",
      act="choose", expected="15"),
    I("combinatorics", "10 choose 3", act="choose", expected="120"),
    I("combinatorics", "in how many ways can 4 people stand in a queue",
      act="arrange", expected="24"),
    # linear algebra
    I("linear algebra", "what is the determinant of [[1,2],[3,4]]",
      act="determinant", expected="-2"),
    I("linear algebra", "transpose [[1,2,3],[4,5,6]]", act="transpose",
      expected="[[1,4],[2,5],[3,6]]"),
    # sets
    I("sets", "what is the union of {1,2} and {2,3}", act="union",
      expected="{1,2,3}"),
    I("sets", "what is the intersection of {1,2,3,4} and {3,4,5}",
      act="intersection", expected="{3,4}"),
    I("sets", "is {1,2} a subset of {1,2,3}", act="subset", expected="yes"),
    # geometry: measures worked out by planning, and asked for
    I("geometry", "what is the area of a circle of radius 3", act="measure",
      expected="9 pi"),
    I("geometry", "a square has an area of 64. what is its perimeter",
      act="measure", expected="32"),
    I("geometry", "find the hypotenuse of a right triangle whose legs are "
      "6 and 8", act="measure", expected="10"),
    I("geometry", "what is the volume of a cube with side length 4 cm",
      act="measure", expected="64 centimeter^3"),
    I("geometry", "what is the sum of the interior angles of an octagon",
      act="measure", expected="1080"),
    I("geometry", "how big is each interior angle of a regular hexagon",
      act="measure", expected="120"),
    I("geometry", "what is the area of a rectangle with length 7 and width "
      "3", act="measure", expected="21"),
    I("geometry", "what is the circumference of a circle with diameter 10",
      act="measure", expected="10 pi"),
    I("geometry", "what is the volume of a cylinder with radius 2 and "
      "height 5", act="measure", expected="20 pi"),
    I("geometry", "what is the area of a rectangle with length 4",
      "the width is 6", act="measure", expected="24"),
    # algebra, further
    I("algebra", "solve 2x + 3 > 7", act="solve", expected="x > 2"),
    I("algebra", "solve x + y = 10 and x - y = 2", act="solve system",
      expected="x = 6 and y = 4"),
    I("algebra", "let f(x) = x^2 + 1", "what is f(3)", act="value",
      expected="10"),
    # measures
    I("measures", "convert 3 feet to inches", act="convert",
      expected="36 inch"),
    I("measures", "how many meters are in 2.5 km", act="convert",
      expected="2500 meter"),
    I("measures", "what is 90 minutes in hours", act="convert",
      expected="1.5 hour"),
    # sequences
    I("sequences", "what comes next: 5, 10, 15, 20", act="next term",
      expected="25"),
    I("sequences", "what is the next number in 3, 9, 27, 81",
      act="next term", expected="243"),
    I("sequences", "what is the nth term of 2, 5, 8, 11", act="nth term",
      expected="3n - 1"),
    I("sequences", "what is the 10th term of 4, 7, 10, 13", act="term",
      expected="31"),
    I("sequences", "what is the sum of 1/2^n for n from 1 to infinity",
      act="series", expected="1"),
    # logic
    I("logic", "is p or not p a tautology", act="is kind", expected="yes"),
    I("logic", "is p and not p a contradiction", act="is kind",
      expected="yes"),
    I("logic", "is not (p and q) equivalent to not p or not q",
      act="equivalent", expected="yes"),
    # probability
    I("probability", "what is the probability of rolling a 4 on a die",
      act="dice", expected="1/6"),
    I("probability", "what is the chance of getting a total of 7 with two "
      "dice", act="dice", expected="1/6"),
    I("probability", "what is the probability of 2 heads in 2 coin tosses",
      act="coins", expected="1/4"),
    # arithmetic, further
    I("arithmetic", "round 3.14159 to 2 decimal places",
      act="round places", expected="3.14"),
    I("arithmetic", "round 1234 to the nearest hundred",
      act="round nearest", expected="1200"),
    I("arithmetic", "what is 15% of 80", act="percent of", expected="12"),
    I("arithmetic", "what percentage of 50 is 20", act="as percent",
      expected="40"),
    I("arithmetic", "what is the percentage change from 80 to 100",
      act="percent change", expected="25"),
    # vectors and complex numbers
    I("linear algebra", "what is the dot product of (1,2,3) and (4,5,6)",
      act="dot", expected="32"),
    I("linear algebra", "what is the magnitude of the vector (3, 4)",
      act="magnitude", expected="5"),
    I("complex numbers", "what is the modulus of 3 + 4i", act="modulus",
      expected="5"),
    I("complex numbers", "what is (2 + 3i) times (1 - i)", act="value",
      expected="5 + i"),
    # not mathematics: must be left alone
    I("none", "what is a prime number", act="none", expected=""),
    I("none", "is a square a rectangle", act="none", expected=""),
    I("none", "mary has 3 apples", act="none", expected=""),
    I("none", "how many legs does a spider have", act="none", expected=""),
    I("none", "what is a dog", act="none", expected=""),
    I("none", "who discovered the number zero", act="none", expected=""),
    I("none", "how many sides does a hexagon have", act="none",
      expected=""),
    I("none", "i rolled a die yesterday", act="none", expected=""),
    I("none", "the recipe needs 2 cups of flour", act="none", expected=""),
]


def _matches(result: doing.Result, expected: str) -> bool:
    if expected in ("yes", "no"):
        return result.stance == expected
    if result.stance != "value":
        return False
    value = result.value
    if isinstance(value, dict):
        # prime factors: the product is the number
        value = S.Mul(*[S.Pow(prime, power) for prime, power in
                        value.items()])
    try:
        wanted = parsed(expected if "," not in expected or "[" in expected
                        or "{" in expected else f"({expected})")
    except Unreadable:
        return result.text.replace(" ", "") == expected.replace(" ", "")
    if isinstance(value, list):
        if isinstance(wanted, S.Tuple):
            return sorted(map(S.sympify, value), key=S.default_sort_key) == \
                sorted(wanted, key=S.default_sort_key)
        value = value[0] if len(value) == 1 else value
    if isinstance(value, S.Basic) and result.act == "integral" and \
            value.free_symbols:
        return S.simplify(S.diff(value - wanted, *value.free_symbols)) == 0
    return same(value, wanted)


def run(item: Item) -> Item:
    bound: dict = {}
    pending = None
    for line in item.said[:-1]:
        read = reading.read(line)
        if read is not None and read.act == "let":
            result = doing.do("let", read.parts)
            if isinstance(result.value, dict):
                bound.update(result.value)
        elif read is not None and read.act == "measure":
            if doing.do("measure", dict(read.parts)).needs:
                pending = dict(read.parts)
    read = reading.read(item.said[-1])
    if read is not None and read.act == "given" and pending is not None:
        # What was asked for, given: the question it finishes, again.
        read.act = "measure"
        read.parts = doing.continued(pending, read.parts)
    item.got_act = read.act if read is not None else "none"
    item.read = item.got_act == item.act
    if item.act == "none":
        item.right = item.honest = read is None
        item.got = "(left alone)" if read is None else f"read as {read.act}"
        return item
    if read is None:
        item.got = "(not read as mathematics)"
        item.honest = True
        return item
    parts = dict(read.parts)
    if read.kind is not None:
        parts["KIND"] = read.kind
    result = doing.do(read.act, parts, bound)
    item.got = result.text or result.stance
    item.right = item.read and _matches(result, item.expected)
    item.honest = item.right or result.stance == "unknown" or bool(
        read.trouble)
    return item


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--branch", default="")
    parser.add_argument("-v", "--verbose", action="store_true")
    options = parser.parse_args(argv)
    if not reading.enabled():
        print("the encoder in use has no mathematics heads "
              "(V689_READER_MODEL=llm/reader-math)")
        return 1
    table: dict = {}
    for item in BANK:
        if options.branch and item.branch != options.branch:
            continue
        run(item)
        mark = "ok " if item.right else ("?  " if item.honest else "NO ")
        if options.verbose or not item.right:
            print(f"{mark}[{item.branch}] {' / '.join(item.said)}  -> "
                  f"{item.got_act}: {item.got}"
                  + ("" if item.right else f"   (expected {item.act}: "
                                           f"{item.expected})"))
        row = table.setdefault(item.branch, [0, 0, 0, 0])
        row[0] += 1
        row[1] += item.read
        row[2] += item.right
        row[3] += item.honest
    print(f"\n{'branch':<16}{'read':>8}{'right':>8}{'honest':>8}")
    totals = [0, 0, 0, 0]
    for branch, row in table.items():
        print(f"{branch:<16}" + "".join(f"{one:>5}/{row[0]:<2}"
                                        for one in row[1:]))
        totals = [a + b for a, b in zip(totals, row)]
    print(f"{'all':<16}" + "".join(f"{one:>5}/{totals[0]:<2}"
                                   for one in totals[1:]))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
