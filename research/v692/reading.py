"""Mathematics read by the encoder: what is asked, and of what.

The shared encoder (`research/encoder.py`) has three heads for mathematics,
taught from `corpus.py`:

    math_act      value, solve, factor, is kind, ... or none
    math_role     each word's part: EXPR VAR A B LIST KIND OTHER, or O
    math_symbol   the symbols each word stands for

So `what is the derivative of x squared with respect to x` is read as the
act `derivative`, the words `x squared` as the expression, standing for
`x ^2`, and the last `x` as the variable. The symbols of each part are put
together and handed to sympy (`symbols.parsed`), and what comes back is the
object -- no rule anywhere reads the words. A kind (`prime`, `a perfect
square`) is the curriculum's (`curriculum.kind_said`).

`read` returns None when the encoder says there is no mathematics to do,
which it was taught to say of everything else (`corpus.negatives`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research import encoder
from research.v692 import curriculum as C
from research.v692.corpus import NONE, words
from research.v692.saying import DROP, KEEP
from research.v692.symbols import Unreadable, parsed

HEADS = ("math_act", "math_role", "math_symbol")
#: How sure the encoder must be that something asks for mathematics.
FLOOR = 0.6


@dataclass
class Reading:
    act: str
    chance: float
    words: list
    roles: list
    labels: list
    #: part -> the object read (EXPR, VAR, A, B, LIST, OTHER)
    parts: dict = field(default_factory=dict)
    #: part -> its symbols, as read
    symbols: dict = field(default_factory=dict)
    kind: C.Kind | None = None
    #: why a part could not be read, where one could not
    trouble: str = ""


def enabled() -> bool:
    """Whether the encoder in use was taught mathematics."""
    if not encoder.enabled():
        return False
    try:
        return "math_act" in encoder.LOADED.get().heads
    except RuntimeError:
        return False


def _symbols(word: str, label: str) -> str:
    if label == DROP:
        return ""
    if label == KEEP:
        return word
    if label.startswith(KEEP + " "):
        return word + label[len(KEEP):]
    return label


#: A measure's parts: each given's quantity and value, what is wanted, and
#: of what shape. The quantity and the shape are the labels the encoder
#: gives their last words (`corpus.measure_record`), not the words.
MEASURE_ROLES = ("NAME", "VALUE", "WANTED", "SHAPE")


def _measure(said, roles, labels) -> dict:
    """The measure parts in order: each run of VALUE words is the value of
    the quantity NAMEd last before it -- `legs 3 and 4` names two values,
    the leg and the other leg."""
    out: dict = {"givens": []}
    name, value = None, []
    for word, role, label in zip(said, roles, labels):
        if role != "VALUE" and value:
            out["givens"].append((name, " ".join(value)))
            value = []
        if role in ("SHAPE", "WANTED", "NAME") and label not in (DROP,
                                                                 KEEP):
            if role == "NAME":
                name = label
            else:
                out[role] = label
        elif role == "VALUE":
            symbols = _symbols(word, label)
            if symbols:
                value.append(symbols)
    if value:
        out["givens"].append((name, " ".join(value)))
    return out if (out["givens"] or len(out) > 1) else {}


def _measure_parts(found: dict) -> dict:
    parts = {}
    if "SHAPE" in found:
        parts["SHAPE"] = found["SHAPE"].replace("-", " ")
    if "WANTED" in found:
        parts["WANTED"] = found["WANTED"]
    givens: dict = {}
    for name, symbols in found["givens"]:
        if name is None:
            continue
        if name in givens and name + "2" not in givens:
            # The second of two: `legs 3 and 4`.
            name = name + "2"
        givens[name] = parsed(symbols)
    if givens:
        parts["GIVENS"] = givens
    return parts


def read(text: str) -> Reading | None:
    """What an utterance asks of mathematics, or None."""
    if not enabled():
        return None
    said = words(text)
    if not said:
        return None
    found = encoder.read(said, heads=HEADS)
    act, chance = found["math_act"][0]
    if act == NONE or chance < FLOOR:
        return None
    return assembled(act, chance, said, found["math_role"],
                     found["math_symbol"])


def assembled(act: str, chance: float, said: list, roles: list,
              labels: list) -> Reading:
    """What each word was read as, put together into the parts: the
    symbols of each part handed to sympy."""
    out = Reading(act, chance, said, roles, labels)
    pieces: dict = {}
    kind_words = []
    measure = _measure(said, roles, labels)
    for word, role, label in zip(said, roles, labels):
        if role == "O" or role in MEASURE_ROLES:
            continue
        if role == "KIND":
            kind_words.append(word)
            continue
        pieces.setdefault(role, []).append(_symbols(word, label))
    if kind_words:
        out.kind = C.kind_said(" ".join(kind_words))
    if measure:
        try:
            out.parts.update(_measure_parts(measure))
        except Unreadable as trouble:
            out.trouble = f"VALUE: {trouble}"
    for role, parts in pieces.items():
        symbols = " ".join(one for one in parts if one)
        # A bracket still open where the part ends closes there: `sin of pi
        # over 2` opens the sine and says nothing more after the 2.
        symbols += " )" * max(0, symbols.count("(") - symbols.count(")"))
        out.symbols[role] = symbols
        try:
            out.parts[role] = parsed(symbols)
        except Unreadable as trouble:
            out.trouble = f"{role}: {trouble}"
    return out
