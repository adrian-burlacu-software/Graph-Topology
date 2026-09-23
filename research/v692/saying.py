"""A mathematical object said aloud, each word with the symbols it stands for.

This is the teacher, and it only ever goes one way: **from an object to
words, never from words to an object.** The encoder learns the other
direction from what this says (`corpus.py`), and at run time it reads maths
said by anyone (`reading.py`); nothing here is asked then. Because the
object is where each sentence starts, every word's label is known by
construction -- `squared` stands for `^2` because it was said *for* `^2` --
and there is no alignment to guess and no parser to write.

    x^2 - 5x + 6      x      squared  minus  five  x  plus  six
                      x      ^2       -      5     x  +     6

A word's label is the symbols it stands for, `KEEP` where the word is its
own symbols (`x`, `17`, `2x^2`) and `DROP` where it stands for nothing
(`the`, `of`). A group closes on the word that ends it: in *the square root
of the quantity x plus one*, `one` is `1 ))`.

Several ways of saying each thing are kept (`STYLES`), because people say
`x^2` and `x squared` and `x to the power of 2`, and an encoder taught one
reads only that one. Which is said is drawn, so a corpus has all of them.

Every saying is checked by the one place that turns symbols back into an
object (`symbols.parsed`, sympy's own parser behind a whitelist): a saying
that does not come back as the object it was said for is not kept.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import sympy as S
from sympy.core.function import AppliedUndef
from sympy.physics.units import Quantity

#: Numbers said as one word.
SMALL = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
         6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
         11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
         15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
         19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty",
         50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty",
         90: "ninety", 100: "a hundred", 1000: "a thousand"}
#: Denominators said as words: `three quarters`, `two thirds`.
PARTS = {2: ("half", "halves"), 3: ("third", "thirds"),
         4: ("quarter", "quarters"), 5: ("fifth", "fifths"),
         6: ("sixth", "sixths"), 8: ("eighth", "eighths"),
         10: ("tenth", "tenths")}
#: Functions said as words, and the symbols each opens with.
FUNCTIONS = {S.sin: ("sine", "sin("), S.cos: ("cosine", "cos("),
             S.tan: ("tangent", "tan("), S.log: ("log", "log("),
             S.exp: ("e to the", "exp("), S.Abs: ("absolute value", "Abs(")}

KEEP, DROP = "KEEP", "DROP"

#: Units as they are said: the word, then how each is written short.
#: A unit is never said as a letter English uses for something else --
#: `in`, `m`, `s`, `g`, `h`, `l` -- because a reader taught those drops
#: the preposition from *expand in warm temperatures*. It was measured:
#: with them, eight tests of ordinary reading moved.
UNIT_WORDS = {"meter": ("meter", "metre"),
              "centimeter": ("centimeter", "cm", "centimetre"),
              "millimeter": ("millimeter", "mm"),
              "kilometer": ("kilometer", "km", "kilometre"),
              "inch": ("inch",), "foot": ("foot", "ft"),
              "yard": ("yard", "yd"), "mile": ("mile",),
              "gram": ("gram",), "kilogram": ("kilogram", "kg"),
              "pound": ("pound", "lb"), "second": ("second", "sec"),
              "minute": ("minute",), "hour": ("hour",),
              "day": ("day",), "liter": ("liter", "litre"),
              "milliliter": ("milliliter", "ml")}
#: Propositions: what `logic` says.
BOOLEAN = (S.And, S.Or, S.Not, S.Implies)


def _is_unit(obj) -> bool:
    return isinstance(obj, Quantity) or (
        isinstance(obj, S.Pow) and isinstance(obj.base, Quantity))


def decimal(obj) -> str:
    """A decimal as written: `3.14`, not `3.14000000000000`."""
    text = str(obj)
    if "." in text and "e" not in text:
        text = text.rstrip("0").rstrip(".")
    return text


#: Ordinals said as words: `the tenth term`.
ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
            6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
            11: "eleventh", 12: "twelfth", 15: "fifteenth",
            20: "twentieth", 50: "fiftieth", 100: "hundredth"}


def ordinal(value: int, rng) -> Said:
    """`the 10th`, `the tenth`: a place in a row, standing for its number."""
    if value in ORDINALS and rng.random() < 0.6:
        return Said().add(ORDINALS[value], str(value))
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(
        value % 10 if value % 100 not in (11, 12, 13) else 0, "th")
    return Said().add(f"{value}{suffix}", str(value))


@dataclass
class Said:
    """Words, and for each the symbols it stands for."""

    words: list = field(default_factory=list)
    labels: list = field(default_factory=list)

    def add(self, word: str, label: str = KEEP) -> "Said":
        for one in word.split():
            self.words.append(one)
            self.labels.append(label if one == word.split()[-1] or
                               label in (KEEP, DROP) else DROP)
        return self

    def then(self, other: "Said") -> "Said":
        self.words += other.words
        self.labels += other.labels
        return self

    def close(self, count: int = 1) -> "Said":
        """The group ends at the last word said."""
        label = self.labels[-1]
        if label == DROP:
            label = ""
        self.labels[-1] = (label + " " + " ".join(")" * count)).strip()
        return self

    def close_brace(self) -> "Said":
        """The set ends at the last word said."""
        label = self.labels[-1]
        self.labels[-1] = "}" if label == DROP else f"{label} }}"
        return self

    def symbols(self) -> str:
        """What the words stand for, as one string of symbols."""
        out = []
        for word, label in zip(self.words, self.labels):
            if label == DROP:
                continue
            if label == KEEP:
                out.append(word)
            elif label.startswith(KEEP + " "):
                out.append(word + label[len(KEEP):])
            else:
                out.append(label)
        return " ".join(out)

    def text(self) -> str:
        return " ".join(self.words)


class Speaker:
    """Says objects, in one style drawn for the whole sentence or per part.

    `style` is `spoken` (every operation in words), `written` (symbols,
    spaced), `tight` (symbols run together, one word) or `mixed` (each part
    drawn)."""

    def __init__(self, rng: random.Random, style: str = "mixed") -> None:
        self.rng = rng
        self.style = style
        #: how a relation is said where it stands: `free` (`x is at least
        #: 4`), `after is` (the sentence's own `is` is its verb: `is 2 + 2
        #: equal to 4`) or `embedded` (inside a clause: `what value of x
        #: makes 2x equal 6 true`)
        self.relating = "free"

    def chosen(self) -> str:
        if self.style != "mixed":
            return self.style
        return self.rng.choice(("spoken", "spoken", "written"))

    # -- the whole object -------------------------------------------------
    def say(self, obj) -> Said:
        if isinstance(obj, (int, float)) or (
                not isinstance(obj, (S.Basic, S.MatrixBase))
                and hasattr(obj, "numerator")):
            obj = S.sympify(obj)
        style = self.chosen()
        if style == "tight" and not isinstance(obj, (S.MatrixBase,)):
            return Said().add(tight(obj), KEEP)
        if isinstance(obj, S.MatrixBase):
            return Said().add(tight(obj), KEEP)
        if isinstance(obj, S.Tuple):
            return self.vector(obj, style)
        if isinstance(obj, S.FiniteSet):
            return self.set_(obj, style)
        if isinstance(obj, S.Rel):
            return self.relation(obj, style)
        if isinstance(obj, BOOLEAN):
            return self.logic(obj, style, 0)
        return self.expression(obj, style, 0)

    def vector(self, obj, style: str) -> Said:
        """`(1, 2, 3)`: each part a word of its own, or all one."""
        if style == "spoken" and self.rng.random() < 0.5:
            return Said().add(tight(obj), KEEP)
        out = Said().add("(", KEEP)
        for index, one in enumerate(obj):
            if index:
                out.add(",", KEEP)
            out.then(self.expression(one, "written", 0))
        return out.add(")", KEEP)

    def logic(self, obj, style: str, holding: int) -> Said:
        """Propositions: `p and not q`, `if p then q`, `p & ~q`. A part
        that holds more loosely than where it stands is bracketed."""
        if style != "spoken":
            return Said().add(tight(obj), KEEP)
        need = {S.Or: 0, S.And: 1, S.Implies: 0}.get(type(obj), 2)
        if need < holding:
            out = Said().add("open", DROP).add("bracket", "(")
            out.then(self.logic(obj, style, 0))
            return out.add("close", DROP).add("bracket", ")")
        if isinstance(obj, S.Symbol):
            return Said().add(obj.name, KEEP)
        if obj in (S.true, S.false):
            return Said().add("true" if obj == S.true else "false",
                              "True" if obj == S.true else "False")
        if isinstance(obj, S.Not):
            return Said().add("not", "~").then(self.logic(obj.args[0],
                                                          style, 2))
        if isinstance(obj, S.Implies):
            out = Said().add("if", "implies(")
            out.then(self.logic(obj.args[0], style, 1))
            out.add("then", ",")
            return out.then(self.logic(obj.args[1], style, 1)).close()
        word, symbol = ("and", "&") if isinstance(obj, S.And) else ("or",
                                                                    "|")
        out = Said()
        args = sorted(obj.args, key=S.default_sort_key)
        for index, one in enumerate(args):
            if index:
                out.add(word, symbol)
            out.then(self.logic(one, style, need + 1))
        return out

    def relation(self, obj, style: str) -> Said:
        ops = {S.Eq: ("=", ("equals", "is equal to", "is")),
               S.Lt: ("<", ("is less than",)),
               S.Gt: (">", ("is greater than", "is more than")),
               S.Le: ("<=", ("is less than or equal to", "is at most")),
               S.Ge: (">=", ("is greater than or equal to",
                             "is at least")),
               S.Ne: ("!=", ("is not equal to",))}
        symbol, words = ops[type(obj)]
        if self.relating != "free":
            # The sentence has its own `is`, or the relation is inside a
            # clause: said without one.
            words = tuple(one[3:] for one in words if one.startswith("is ")
                          ) or words
            if self.relating == "embedded" and type(obj) is S.Eq:
                words = ("equals",)
        out = self.expression(obj.lhs, style, 0)
        if style == "spoken":
            said = self.rng.choice(words)
            first, *rest = said.split()
            out.add(first, symbol)
            for word in rest:
                out.add(word, DROP)
        else:
            out.add(symbol, KEEP)
        return out.then(self.expression(obj.rhs, style, 0))

    def set_(self, obj, style: str) -> Said:
        items = sorted(obj, key=S.default_sort_key)
        if style == "spoken" and not items:
            return Said().add("the", DROP).add("empty", DROP).add(
                "set", "EmptySet")
        if style != "spoken" or not items:
            return Said().add(tight(obj), KEEP)
        out = Said().add("the", DROP).add("set", "{")
        for index, one in enumerate(items):
            if index:
                out.add("," if index < len(items) - 1 else "and", ",")
            out.then(self.expression(one, "spoken", 0))
        return out.close_brace()

    # -- expressions, by how tightly each holds --------------------------
    #: 0 a sum, 1 a product, 2 a power's base, 3 an atom
    def expression(self, obj, style: str, holding: int) -> Said:
        need = level(obj)
        if need < holding:
            return self.grouped(obj, style)
        if isinstance(obj, S.Add):
            return self.sum_(obj, style)
        if isinstance(obj, S.Mul):
            return self.product(obj, style)
        if isinstance(obj, S.Pow):
            return self.power(obj, style)
        if isinstance(obj, S.factorial):
            out = self.expression(obj.args[0], style, 3)
            return out.add("factorial" if style == "spoken" else "!", "!")
        if obj.func in FUNCTIONS:
            return self.function(obj, style)
        if isinstance(obj, AppliedUndef):
            return self.applied(obj, style)
        return self.atom(obj, style)

    def applied(self, obj, style: str) -> Said:
        """`f(3)`, `f of x`."""
        name = str(obj.func)
        if style != "spoken" or self.rng.random() < 0.4:
            out = Said().add(name + "(", KEEP)
            for index, one in enumerate(obj.args):
                if index:
                    out.add(",", KEEP)
                out.then(self.expression(one, "written", 0))
            return out.add(")", KEEP)
        out = Said().add(name, name + "(").add("of", DROP)
        return out.then(self.expression(obj.args[0], style, 3)).close()

    def grouped(self, obj, style: str) -> Said:
        """A part said as one: `the quantity x plus one`, `open bracket x
        plus one close bracket`, or `(x + 1)`."""
        if style == "spoken":
            if self.rng.random() < 0.5:
                out = Said().add("the", DROP).add("quantity", "(")
                return out.then(self.expression(obj, style, 0)).close()
            out = Said().add("open", DROP).add("bracket", "(")
            out.then(self.expression(obj, style, 0))
            return out.add("close", DROP).add("bracket", ")")
        out = Said().add("(", KEEP).then(self.expression(obj, style, 0))
        return out.add(")", KEEP)

    def sum_(self, obj, style: str) -> Said:
        terms = S.Add.make_args(obj)
        terms = sorted(terms, key=_order)
        out = Said()
        for index, term in enumerate(terms):
            negative = _negative(term)
            shown = -term if negative else term
            if index == 0:
                if negative:
                    out.add("minus" if style == "spoken" else "-", "-")
            else:
                sign = "-" if negative else "+"
                out.add(("minus" if negative else "plus")
                        if style == "spoken" else sign, sign)
            out.then(self.expression(shown, style, 1))
        return out

    def product(self, obj, style: str) -> Said:
        numerator, denominator = S.fraction(obj)
        if denominator != 1 and not obj.is_Rational:
            out = self.expression(numerator, style, 1)
            out.add(self.rng.choice(("over", "divided by"))
                    if style == "spoken" else "/", "/")
            if style == "spoken" and out.words[-1] == "by":
                out.labels[-2:] = ["/", DROP]
            return out.then(self.expression(denominator, style, 2))
        coefficient, rest = obj.as_coeff_Mul()
        out = Said()
        factors = [one for one in S.Mul.make_args(rest)]
        units = [one for one in factors if _is_unit(one)]
        if units:
            # `25 pi square centimeters`: the amount, then its units, with
            # nothing between.
            amount = obj / S.Mul(*units)
            out = (self.expression(amount, style, 1) if amount != 1
                   else Said())
            for one in units:
                out.then(self.expression(one, style, 2))
            return out
        if coefficient.is_negative:
            out.add("minus" if style == "spoken" else "-", "-")
            coefficient = -coefficient
        if coefficient != 1:
            out.then(self.expression(coefficient, style, 2))
            if style == "spoken" and self.rng.random() < 0.3:
                out.add("times", "*")
        for index, factor in enumerate(factors):
            # Two numbers side by side are one number: `17 4` is not 68.
            numbers = (factor.is_Number and (
                index or (coefficient != 1 and coefficient != -1))) or (
                index and level(factor) < 2)
            if (numbers or index) and style == "spoken" and (
                    numbers or self.rng.random() < 0.4):
                out.add(self.rng.choice(("times", "multiplied by"))
                        if numbers else "times", "*")
                if out.words[-1] == "by":
                    out.labels[-2:] = ["*", DROP]
            elif (numbers or index) and style != "spoken":
                out.add("*", KEEP)
            out.then(self.expression(factor, style, 2))
        return out

    def power(self, obj, style: str) -> Said:
        base, exponent = obj.args
        if isinstance(base, Quantity) and exponent in (2, 3):
            return self.unit(base, style, int(exponent))
        if exponent == S.Rational(1, 2):
            return self.root(base, style, "square root", "sqrt(")
        if exponent == -1:
            out = Said().add("one" if style == "spoken" else "1", "1")
            out.add("over" if style == "spoken" else "/", "/")
            return out.then(self.expression(base, style, 2))
        if exponent.is_Rational and exponent == S.Rational(1, 3):
            return self.root(base, style, "cube root", "cbrt(")
        out = self.expression(base, style, 3)
        if style != "spoken":
            return out.add("^", KEEP).then(self.expression(exponent, style,
                                                           3))
        if exponent == 2 and self.rng.random() < 0.8:
            return out.add("squared", "^2")
        if exponent == 3 and self.rng.random() < 0.8:
            return out.add("cubed", "^3")
        out.add("to", "^").add("the", DROP).add("power", DROP).add(
            "of", DROP)
        return out.then(self.expression(exponent, style, 3))

    def root(self, base, style: str, name: str, symbol: str) -> Said:
        if style != "spoken":
            out = Said().add(symbol.rstrip("(") + "(", KEEP)
            return out.then(self.expression(base, style, 0)).add(")", KEEP)
        out = Said().add("the", DROP)
        words = name.split()
        out.add(words[0], DROP).add(words[1], symbol).add("of", DROP)
        return out.then(self.expression(base, style, 3)).close()

    def function(self, obj, style: str) -> Said:
        name, symbol = FUNCTIONS[obj.func]
        argument = obj.args[0]
        if obj.func is S.log and len(obj.args) == 1 and style == "spoken":
            name = self.rng.choice(("log", "natural log", "the log of"))
        if style != "spoken":
            out = Said().add(symbol, KEEP)
            return out.then(self.expression(argument, style, 0)).add(")",
                                                                    KEEP)
        out = Said()
        words = name.split()
        for word in words[:-1]:
            out.add(word, DROP)
        out.add(words[-1], symbol)
        if words[-1] != "of" and self.rng.random() < 0.5 and \
                obj.func is not S.exp:
            out.add("of", DROP)
        return out.then(self.expression(argument, style, 3)).close()

    def atom(self, obj, style: str) -> Said:
        if obj is S.pi:
            return Said().add("pi", "pi")
        if obj is S.E:
            return Said().add("e", "E")
        if obj is S.I:
            return Said().add("i", "I")
        if obj is S.oo:
            return Said().add("infinity", "oo")
        if isinstance(obj, S.Symbol):
            return Said().add(obj.name, KEEP)
        if isinstance(obj, Quantity):
            return self.unit(obj, style, 1)
        if obj.is_Integer:
            return self.integer(int(obj), style)
        if obj.is_Rational:
            return self.fraction(obj, style)
        if obj.is_Float:
            return Said().add(decimal(obj), KEEP)
        return Said().add(tight(obj), KEEP)

    def unit(self, obj, style: str, power: int) -> Said:
        """`centimeters`, `cm`, `square meters`, `m^2`: the word stands for
        the unit's sympy name."""
        name = obj.name if hasattr(obj, "name") else str(obj)
        name = str(name)
        full, *short = UNIT_WORDS[name]
        if style == "spoken" or not short:
            word = self.rng.choice((full + "s", full + "s", full)
                                   if full[-1] != "h" else (full + "es",
                                                            full))
            if name == "foot":
                word = self.rng.choice(("feet", "foot"))
            if power == 1:
                return Said().add(word, name)
            said = ("square", "cubic")[power - 2]
            if self.rng.random() < 0.6:
                return Said().add(said, DROP).add(word, f"{name} ^{power}")
            return Said().add(word, name).add(("squared", "cubed")[power - 2],
                                              f"^{power}")
        word = self.rng.choice(short)
        if power == 1:
            return Said().add(word, name)
        return Said().add(f"{word}^{power}", f"{name} ^{power}")

    def integer(self, value: int, style: str) -> Said:
        if value < 0:
            out = Said().add("minus" if style == "spoken" else "-", "-")
            return out.then(self.integer(-value, style))
        if style == "spoken" and value in SMALL and self.rng.random() < 0.7:
            words = SMALL[value].split()
            out = Said()
            for word in words[:-1]:
                out.add(word, DROP)
            return out.add(words[-1], str(value))
        return Said().add(str(value), KEEP)

    def fraction(self, value, style: str) -> Said:
        top, bottom = int(value.p), int(value.q)
        if style == "spoken" and bottom in PARTS and 0 < abs(top) < 10 \
                and self.rng.random() < 0.6:
            out = Said()
            if top < 0:
                out.add("minus", "-")
            count = abs(top)
            if count == 1:
                out.add(self.rng.choice(("one", "a")), "1")
            else:
                out.add(SMALL[count], str(count))
            one, many = PARTS[bottom]
            return out.add(one if count == 1 else many, f"/{bottom}")
        if style == "spoken" and self.rng.random() < 0.3:
            out = self.integer(top, style)
            return out.add("over", "/").then(self.integer(bottom, style))
        return Said().add(f"{top}/{bottom}", KEEP)


def level(obj) -> int:
    """How tightly an object holds together when said in a row: a sum
    loosest, then a product, then a power, then a single thing."""
    if isinstance(obj, S.Add):
        return 0
    if isinstance(obj, S.Mul):
        return 1
    if obj.is_Rational and not obj.is_Integer:
        return 1
    if obj.is_Number and obj < 0:
        return 0
    if isinstance(obj, S.Pow):
        return 2
    return 3


def _negative(term) -> bool:
    coefficient, _ = term.as_coeff_Mul()
    return bool(coefficient.is_negative)


def _order(term):
    """Terms in the order a person writes a polynomial: highest power
    first, constants last."""
    degree = S.Poly(term).total_degree() if term.free_symbols and \
        term.is_polynomial() else 0
    return (-degree, S.default_sort_key(term))


def tight(obj) -> str:
    """An object as written with no spaces: `2x^2+3x-5`, `{1,2,3}`,
    `[[1,2],[3,4]]` -- one word, which is its own symbols."""
    if isinstance(obj, S.MatrixBase):
        return "[" + ",".join("[" + ",".join(tight(one) for one in
                                             obj.row(index)) + "]"
                              for index in range(obj.rows)) + "]"
    if obj == S.EmptySet:
        return "EmptySet"
    if isinstance(obj, S.FiniteSet):
        return "{" + ",".join(tight(one) for one in
                              sorted(obj, key=S.default_sort_key)) + "}"
    if isinstance(obj, S.Rel):
        symbol = {S.Eq: "=", S.Lt: "<", S.Gt: ">", S.Le: "<=", S.Ge: ">=",
                  S.Ne: "!="}[type(obj)]
        return f"{tight(obj.lhs)}{symbol}{tight(obj.rhs)}"
    if getattr(obj, "is_Float", False):
        return decimal(obj)
    text = S.sstr(obj, order="grlex").replace("**", "^").replace(" ", "")
    return text


def say(obj, rng: random.Random | None = None, style: str = "mixed") -> Said:
    return Speaker(rng or random.Random(0), style).say(obj)
