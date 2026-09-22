"""Numbers, exactly: what a number word is worth, and arithmetic said aloud.

Everything else in v691 is learned, read off VerbNet or asked of the store,
and a lesson can be scoped, widened and refuted. **Arithmetic cannot be any
of those things.** Nobody should learn from a surprise that seven eights are
fifty-six, and a planner that believed it on the evidence of two tries would
be wrong on the third. So this module is the one place numbers are worked
out, it is exact -- `Fraction`, never a float, so a third of three is one --
and nothing in it is learned.

    value("twenty one")            21
    value("a dozen")               12
    value("two and a half")        5/2
    evaluate("17 times 4")         68
    evaluate("subtract 3 from 10") 7
    said(Fraction(5, 2))           "2.5"

What is here is only what the words are worth. What a number is a number
*of* -- apples, and whose -- is `quantities.py`, which is where arithmetic
meets a world.
"""
from __future__ import annotations

import re
from fractions import Fraction

UNITS = {"zero": 0, "nought": 0, "nil": 0, "none": 0, "one": 1, "two": 2,
         "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6,
          "billion": 10 ** 9}
#: Words that are a number of their own: `a dozen eggs`, `a pair of shoes`.
COLLECTIVES = {"dozen": 12, "score": 20, "pair": 2, "couple": 2,
               "gross": 144}
#: Fractions said as words, as the denominator's name.
PARTS = {"half": 2, "halves": 2, "third": 3, "thirds": 3, "quarter": 4,
         "quarters": 4, "fourth": 4, "fourths": 4, "fifth": 5, "fifths": 5,
         "sixth": 6, "sixths": 6, "eighth": 8, "eighths": 8, "tenth": 10,
         "tenths": 10}

#: A number written in figures: `12`, `3.5`, `1,000`, `3/4`, `-2`.
FIGURES = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
FRACTION = re.compile(r"^(-?\d+)/(\d+)$")


def _figure(word: str) -> Fraction | None:
    found = FRACTION.match(word)
    if found:
        return (Fraction(int(found.group(1)), int(found.group(2)))
                if int(found.group(2)) else None)
    if FIGURES.match(word):
        return Fraction(word.replace(",", ""))
    return None


def value(text: str) -> Fraction | None:
    """What a number said in words or figures is worth; None when the words
    are not a number, all of them.

    English builds a number by adding within a scale and multiplying by it:
    *two hundred and five thousand* is (2 * 100 + 5) * 1000. That is the
    whole grammar, and `and` is only a joiner inside it -- except before a
    fraction, where *two and a half* adds the half.
    """
    found = _figure((text or "").strip())
    if found is not None:
        return found
    words = [one for one in re.split(r"[\s-]+", (text or "").lower().strip())
             if one]
    if not words:
        return None
    total, current, seen = Fraction(0), Fraction(0), False
    index = 0
    while index < len(words):
        word = words[index]
        figure = _figure(word)
        if figure is not None:
            current += figure
            seen = True
        elif word in UNITS:
            current += UNITS[word]
            seen = True
        elif word in TENS:
            current += TENS[word]
            seen = True
        elif word in SCALES:
            current = (current or 1) * SCALES[word]
            if SCALES[word] >= 1000:
                total, current = total + current, Fraction(0)
            seen = True
        elif word in COLLECTIVES:
            current = (current or 1) * COLLECTIVES[word]
            seen = True
        elif word in PARTS:
            # `a half`, `three quarters`, and `two and a half`, where the
            # `and` said the half is added to what came before it.
            count = current if current and not _joined(words, index) else 1
            part = Fraction(count, PARTS[word])
            if _joined(words, index):
                current += part
            else:
                current = part
            seen = True
        elif word in ("a", "an", "one") and index + 1 < len(words):
            # `a dozen`, `a hundred`, `a half`: the article is a one.
            pass
        elif word == "and" and seen:
            pass
        elif word == "of" and index > 0 and words[index - 1] in PARTS:
            # `half of`: the fraction is of whatever follows, which is
            # arithmetic (`evaluate`), not a number.
            return None
        else:
            return None
        index += 1
    return total + current if seen else None


def _joined(words: list, index: int) -> bool:
    """`two and a half`: an `and` (and perhaps `a`) before a fraction."""
    before = words[max(0, index - 2):index]
    return "and" in before


def said(number: Fraction | int | float) -> str:
    """A number as it is said back: `3`, `2.5`, and a fraction only where a
    decimal would not end -- `1/3`, not 0.3333."""
    number = Fraction(number)
    if number.denominator == 1:
        return str(number.numerator)
    denominator = number.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    if denominator == 1:
        return format(float(number), "f").rstrip("0").rstrip(".")
    return f"{number.numerator}/{number.denominator}"


# -- inflection ------------------------------------------------------------

_PLURALS: dict | None = None


def plural(noun: str, count=2) -> str:
    """A noun for a count of things: `1 apple`, `2 apples`, `3 mice`.

    The irregular forms are WordNet's own, read backwards -- it keeps `mice`
    so that it can find `mouse` -- the way `openworld.past` reads the verbs.
    """
    if Fraction(count) == 1:
        return noun
    global _PLURALS
    if _PLURALS is None:
        _PLURALS = {}
        try:
            from nltk.corpus import wordnet
            wordnet.ensure_loaded()
            for form, lemmas in wordnet._exception_map["n"].items():
                for lemma in lemmas:
                    if lemma != form:
                        _PLURALS.setdefault(lemma, form)
        except Exception:                          # noqa: BLE001
            pass
    if noun in _PLURALS:
        return _PLURALS[noun]
    if noun in _PLURALS.values() or noun in _FORMS():
        # Already a plural: `people`, `children`.
        return noun
    if re.search(r"(s|x|z|ch|sh)$", noun):
        return noun + "es"
    if noun.endswith("y") and noun[-2:-1] not in tuple("aeiou"):
        return noun[:-1] + "ies"
    return noun + "s"


_FORMS_SEEN: frozenset | None = None


def _FORMS() -> frozenset:
    """Every irregular noun form WordNet lists -- `people`, `mice` -- which
    are plurals already."""
    global _FORMS_SEEN
    if _FORMS_SEEN is None:
        try:
            from nltk.corpus import wordnet
            wordnet.ensure_loaded()
            _FORMS_SEEN = frozenset(wordnet._exception_map["n"])
        except Exception:                          # noqa: BLE001
            _FORMS_SEEN = frozenset()
    return _FORMS_SEEN


def counted(count, noun: str) -> str:
    """`3 apples`, `1 apple`, `2.5 cakes`."""
    return f"{said(count)} {plural(noun, count)}"


# -- arithmetic said aloud -------------------------------------------------

#: How an operation is said, longest first, as the symbol it is.
SAID_AS = (
    ("to the power of", "^"), ("raised to the power of", "^"),
    ("multiplied by", "*"), ("divided by", "/"), ("added to", "+"),
    ("take away", "-"), ("minus", "-"), ("plus", "+"), ("times", "*"),
    ("over", "/"), ("into", None), ("and", "and"), ("squared", "^2"),
    ("cubed", "^3"), ("percent of", "%of"), ("per cent of", "%of"),
    ("% of", "%of"), ("x", "*"), ("×", "*"), ("÷", "/"),
)
#: What a question about a sum opens with, and is not part of it.
ASKED = re.compile(r"^(?:what(?:'s| is| are| does| do)?|how much(?: is| are)?|"
                   r"calculate|compute|work out|evaluate|find|tell me|"
                   r"what do you get (?:if you|when you)|can you (?:tell me|"
                   r"work out|calculate))\s+")
#: Operations said before their operands: `the sum of 3 and 4`,
#: `subtract 3 from 10`, `the square root of 16`.
PREFIXED = re.compile(
    r"^(?:the )?(sum|product|difference|quotient|square root|cube root|"
    r"square|cube|half|double|twice|triple|average|mean)(?: of| between)?\s+")
VERBED = re.compile(r"^(add|subtract|multiply|divide|take)\s+(.+?)\s+"
                    r"(to|from|by|into|away from|and)\s+(.+)$")


class NotArithmetic(ValueError):
    """The words are not a sum this module can do."""


def evaluate(text: str) -> Fraction:
    """A sum said in words or symbols, worked out exactly.

    `what is 3 plus 4 times 2` is 11: times binds tighter, as it does on
    paper, because that is what the person who wrote it meant. Raises
    `NotArithmetic` for anything that is not all numbers and operations --
    `what is a dog` has to stay somebody else's question.
    """
    plain = " ".join((text or "").lower().replace("?", " ").split())
    plain = re.sub(r"[.!]+$", "", plain).strip()
    plain = ASKED.sub("", plain)
    plain = re.sub(r"\s*=\s*$", "", plain)
    if not plain:
        raise NotArithmetic(text)
    return _expression(plain)


def _operand(plain: str) -> Fraction:
    """One side of an operation: a number on its own, or a sum."""
    found = value(plain)
    return found if found is not None else _expression(plain)


def _expression(plain: str) -> Fraction:
    plain = plain.strip()
    verbed = VERBED.match(plain)
    if verbed:
        verb, first, joiner, second = verbed.groups()
        one, two = _operand(first), _operand(second)
        if verb == "add":
            return one + two
        if verb in ("subtract", "take"):
            return two - one
        if verb == "multiply":
            return one * two
        if verb == "divide":
            if joiner == "into":
                one, two = two, one
            if two == 0:
                raise NotArithmetic("division by zero")
            return one / two
    prefixed = PREFIXED.match(plain)
    if prefixed:
        rest = plain[prefixed.end():]
        what = prefixed.group(1)
        if what in ("sum", "product", "difference", "quotient", "average",
                    "mean"):
            parts = _listed(rest)
            if len(parts) < 2:
                raise NotArithmetic(plain)
            values = [_operand(one) for one in parts]
            if what == "sum":
                return sum(values, Fraction(0))
            if what in ("average", "mean"):
                return sum(values, Fraction(0)) / len(values)
            if what == "product":
                out = Fraction(1)
                for one in values:
                    out *= one
                return out
            if what == "difference":
                return abs(values[0] - values[1])
            if values[1] == 0:
                raise NotArithmetic("division by zero")
            return values[0] / values[1]
        inner = _operand(rest)
        if what in ("square root", "cube root"):
            root = _root(inner, 2 if what == "square root" else 3)
            if root is None:
                raise NotArithmetic(f"no exact {what} of {inner}")
            return root
        return {"square": inner ** 2, "cube": inner ** 3, "half": inner / 2,
                "double": inner * 2, "twice": inner * 2,
                "triple": inner * 3}[what]
    tokens = _tokens(plain)
    if not tokens or not any(kind == "op" for kind, _ in tokens):
        # A number on its own is a number, not a sum; `what is five` is
        # not a question worth taking from anybody.
        raise NotArithmetic(plain)
    found, at = _sum(tokens, 0)
    if at != len(tokens):
        raise NotArithmetic(plain)
    return found


def _listed(text: str) -> list:
    """`3, 4 and 5` -> the three of them."""
    parts = re.split(r"\s*,\s*|\s+and\s+", text)
    return [one for one in parts if one]


def _root(number: Fraction, degree: int) -> Fraction | None:
    if number < 0:
        return None
    top = round(number.numerator ** (1 / degree))
    bottom = round(number.denominator ** (1 / degree))
    for one in (top - 1, top, top + 1):
        for two in (bottom - 1, bottom, bottom + 1):
            if one >= 0 and two > 0 and Fraction(one, two) ** degree == \
                    number:
                return Fraction(one, two)
    return None


def _tokens(plain: str) -> list:
    """(kind, value) pairs: numbers, operators and brackets. A run of words
    that together are one number -- `twenty one` -- is one token."""
    text = f" {plain} "
    text = text.replace("(", " ( ").replace(")", " ) ")
    text = re.sub(r"(?<=\d)%(?=\s)", " % ", text)
    text = f" {' '.join(text.split())} "
    for words, symbol in SAID_AS:
        if symbol is None:
            continue
        text = re.sub(rf"(?<=\s){re.escape(words)}(?=\s)", f" {symbol} ",
                      text)
    text = re.sub(r"(?<=\d)\s*([+*/^%-])\s*(?=[\d(])", r" \1 ", text)
    out, words = [], []

    def flush() -> None:
        if not words:
            return
        found = value(" ".join(words))
        if found is None:
            raise NotArithmetic(" ".join(words))
        out.append(("number", found))
        words.clear()

    for word in text.split():
        if word in ("+", "-", "*", "/", "^", "(", ")", "%of", "%"):
            # `-3` at the start or after an operator is a negative number,
            # which `value` reads; here it is subtraction.
            flush()
            out.append(("op", word) if word not in "()" else (word, word))
        elif word in ("^2", "^3"):
            flush()
            out.append(("op", "^"))
            out.append(("number", Fraction(int(word[1]))))
        elif word == "and":
            # Inside a number (`two hundred and five`) or a joiner the
            # prefixed forms already took: either way it is not an operator.
            words.append(word)
        else:
            words.append(word)
    flush()
    return out


def _sum(tokens: list, at: int) -> tuple:
    left, at = _product(tokens, at)
    while at < len(tokens) and tokens[at] in (("op", "+"), ("op", "-")):
        op = tokens[at][1]
        right, at = _product(tokens, at + 1)
        left = left + right if op == "+" else left - right
    return left, at


def _product(tokens: list, at: int) -> tuple:
    left, at = _power(tokens, at)
    while at < len(tokens) and tokens[at] in (("op", "*"), ("op", "/"),
                                              ("op", "%of")):
        op = tokens[at][1]
        right, at = _power(tokens, at + 1)
        if op == "*":
            left = left * right
        elif op == "%of":
            left = left / 100 * right
        else:
            if right == 0:
                raise NotArithmetic("division by zero")
            left = left / right
    return left, at


def _power(tokens: list, at: int) -> tuple:
    base, at = _atom(tokens, at)
    if at < len(tokens) and tokens[at] == ("op", "^"):
        exponent, at = _power(tokens, at + 1)
        if exponent.denominator != 1 or abs(exponent) > 64:
            raise NotArithmetic("only whole powers")
        if base == 0 and exponent < 0:
            raise NotArithmetic("division by zero")
        base = base ** int(exponent)
    if at < len(tokens) and tokens[at] == ("op", "%"):
        base, at = base / 100, at + 1
    return base, at


def _atom(tokens: list, at: int) -> tuple:
    if at >= len(tokens):
        raise NotArithmetic("ends early")
    kind, found = tokens[at]
    if kind == "number":
        return found, at + 1
    if kind == "op" and found == "-":
        inner, at = _atom(tokens, at + 1)
        return -inner, at
    if kind == "(":
        inner, at = _sum(tokens, at + 1)
        if at >= len(tokens) or tokens[at][0] != ")":
            raise NotArithmetic("unclosed bracket")
        return inner, at + 1
    raise NotArithmetic(f"unexpected {found}")
