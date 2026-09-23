"""Symbols to an object: sympy's own parser, behind a whitelist.

What the encoder reads an utterance as is a string of symbols
(`reading.py`): `x ^2 - 5 x + 6`. Turning that into an object is sympy's
job, not ours -- but `sympy.parse_expr` works by `eval`, so what reaches it
is checked first: only the characters mathematics is written in, and only
the names the encoder's labels can say (`NAMES`, single letters, Greek). A
string that is anything else is refused, so the worst the page can be asked
to do is read something that is not mathematics.

Relations are split before parsing -- `=` is assignment to Python -- and
braces and brackets become sets and matrices after it.

Besides numbers and variables a string may name units (`5 kilometer`, the
names of `sympy.physics.units`), functions `f`, `g` and `h` applied to
something (`f(3)`), and propositions joined by `&`, `|` and `~` (`p & q`).
Written lower case -- the encoder's words are -- `abs(`, `implies(` and
the like are their sympy names, and a lone `i` or `e` is the number.
"""
from __future__ import annotations

import re

import sympy as S
from sympy.parsing.sympy_parser import (convert_xor,
                                        implicit_multiplication_application,
                                        parse_expr, standard_transformations)

#: Function and constant names a string may use.
NAMES = frozenset({"sqrt", "cbrt", "sin", "cos", "tan", "sec", "csc", "cot",
                   "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp",
                   "log", "ln", "Abs", "factorial", "binomial", "gcd", "lcm",
                   "floor", "ceiling", "pi", "E", "I", "oo", "Matrix",
                   "Min", "Max", "Implies", "Equivalent", "EmptySet"})
#: Names as the encoder's lower-case words write them.
ALIASES = {"abs": "Abs", "implies": "Implies", "equivalent": "Equivalent",
           "emptyset": "EmptySet",
           "matrix": "Matrix", "min": "Min", "max": "Max"}
#: Units a string may name, as `sympy.physics.units` does.
UNIT_NAMES = ("meter", "centimeter", "millimeter", "kilometer", "inch",
              "foot", "yard", "mile", "gram", "kilogram", "pound", "second",
              "minute", "hour", "day", "liter", "milliliter")
#: Letters that name a function where something is applied to them.
FUNCTION_LETTERS = ("f", "g", "h")
GREEK = frozenset("""alpha beta gamma delta epsilon zeta eta theta iota
kappa lamda mu nu xi rho sigma tau upsilon phi chi psi omega""".split())
#: The characters mathematics is written in.
ALLOWED = re.compile(r"^[0-9A-Za-z+\-*/^().,=<>!\s\[\]{}|%&~]*$")
RELATIONS = (("<=", S.Le), (">=", S.Ge), ("!=", S.Ne), ("=", S.Eq),
             ("<", S.Lt), (">", S.Gt))
TRANSFORMS = standard_transformations + (implicit_multiplication_application,
                                         convert_xor)


class Unreadable(ValueError):
    """Not a string of mathematics."""


def safe(text: str) -> bool:
    if not ALLOWED.match(text or "") or "__" in text:
        return False
    for name in re.findall(r"[A-Za-z_]+", text):
        if name in NAMES or name in GREEK or name in ALIASES or \
                name in UNIT_NAMES or len(name) == 1:
            continue
        # `xy`, `abc`: letters written together, which implicit
        # multiplication reads as a product.
        if re.fullmatch(r"[a-z]{2,3}", name):
            continue
        return False
    return True


def _depth_split(text: str, op: str) -> list:
    """`text` split on `op` where no bracket is open."""
    parts, depth, start, at = [], 0, 0, 0
    while at < len(text):
        char = text[at]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith(op, at):
            # `<` is not the start of `<=`, and `=` not the end of `<=`.
            longer = any(text.startswith(one, at) and len(one) > len(op)
                         for one, _ in RELATIONS)
            after_other = op == "=" and at and text[at - 1] in "<>!"
            if not longer and not after_other:
                parts.append(text[start:at])
                start = at + len(op)
                at += len(op)
                continue
        at += 1
    parts.append(text[start:])
    return parts


def parsed(text: str):
    """The object a string of symbols is: an expression, a relation, a set,
    a matrix, or a tuple. Raises `Unreadable`."""
    text = (text or "").strip()
    if not text or not safe(text):
        raise Unreadable(text)
    for op, relation in RELATIONS:
        parts = _depth_split(text, op)
        if len(parts) == 2:
            # Not evaluated: `2 + 2 = 5` is a claim to check, not `False`.
            return relation(parsed(parts[0]), parsed(parts[1]),
                            evaluate=False)
        if len(parts) > 2:
            raise Unreadable(text)
    text = text.replace("ln(", "log(")
    text = re.sub(r"\b(" + "|".join(ALIASES) + r")\b",
                  lambda found: ALIASES[found.group(1)], text)
    # A letter on its own, or after a number: `4i` is 4 times i.
    local = {name: S.Symbol(name) for name in
             set(re.findall(r"(?<![A-Za-z_])[a-zA-Z](?![A-Za-z_])", text))
             - {"E", "I"}}
    # A lone `i` or `e` is the number, as it is said.
    local.update({name: value for name, value in (("i", S.I), ("e", S.E))
                  if name in local})
    local.update({name: S.Function(name) for name in FUNCTION_LETTERS
                  if re.search(rf"\b{name}\s*\(", text)})
    local.update({name: S.Symbol(name) for name in GREEK
                  if re.search(rf"\b{name}\b", text)})
    local.update({name: _unit(name) for name in UNIT_NAMES
                  if re.search(rf"\b{name}\b", text)})
    try:
        found = parse_expr(text, local_dict=local,
                           transformations=TRANSFORMS, evaluate=True)
    except Exception as trouble:                     # noqa: BLE001
        raise Unreadable(f"{text}: {trouble}") from None
    if isinstance(found, (set, frozenset)):
        return S.FiniteSet(*found)
    if isinstance(found, list):
        try:
            return S.Matrix(found)
        except Exception as trouble:                 # noqa: BLE001
            raise Unreadable(f"{text}: {trouble}") from None
    if isinstance(found, tuple):
        return S.Tuple(*found)
    if not isinstance(found, S.Basic) and not isinstance(found, S.MatrixBase):
        raise Unreadable(text)
    return found


def _unit(name: str):
    from sympy.physics import units
    return getattr(units, name)


def same(one, other) -> bool:
    """Whether two objects are the same mathematics: equal after
    simplification, for expressions, sides of relations, sets and
    matrices."""
    try:
        if isinstance(one, S.MatrixBase) or isinstance(other, S.MatrixBase):
            return (isinstance(one, S.MatrixBase)
                    and isinstance(other, S.MatrixBase)
                    and one.shape == other.shape
                    and all(S.simplify(a - b) == 0
                            for a, b in zip(one, other)))
        if isinstance(one, S.Rel) or isinstance(other, S.Rel):
            return (type(one) is type(other)
                    and same(one.lhs, other.lhs) and same(one.rhs, other.rhs))
        if isinstance(one, S.Set) or isinstance(other, S.Set):
            return one == other
        return one == other or S.simplify(one - other) == 0
    except Exception:                                # noqa: BLE001
        return False
