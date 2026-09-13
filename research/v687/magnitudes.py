"""R31: bigger, smaller, heavier, lighter -- compared on a scale people rated.

    is an elephant bigger than a mouse     yes: 376 against 191
    which is heavier, a feather or a brick  a brick: 5.5 of 7 against 1.1

R18 refused every comparative because nothing here had a magnitude: AwA2
records `big` as a yes or no, and XCSLB records `is the largest animal` as a
string. THINGSplus does have one. About 45 people placed each of 1,854 objects
on a real-world size scale anchored from a grain of sand to an aircraft
carrier, and about 40 rated each 1-7 on how heavy it is. Two objects on the
same scale can be compared; anything else is still refused by R18, and faster,
older and better have no scale at all.

Close is not an answer. Two readings 20 units of size or one point of weight
apart are called too close, rather than ordered by the noise between them.
"""
from __future__ import annotations

from . import pins
from .rated import ANCHORS, HEAVY_TIE, SIZE_TIE, with_article

#: The comparative -> the scale, and which end of it is asked for.
COMPARATIVES = {"bigger": ("size", 1), "larger": ("size", 1),
                "smaller": ("size", -1), "heavier": ("heavy", 1),
                "lighter": ("heavy", -1)}

LEADING = ("a", "an", "the")


def _phrase(words: list[str]) -> str:
    while words and words[0] in LEADING:
        words = words[1:]
    return " ".join(words)


def read(question: str) -> tuple[str, str, str, str] | None:
    """(polar | choice, comparative, first, second), or None.

    String operations rather than a pattern: three regular expressions in this
    repository have had their escapes mangled in transit.
    """
    words = (question or "").lower().replace(",", " ").rstrip("?").split()
    at = [index for index, word in enumerate(words) if word in COMPARATIVES]
    if len(at) != 1 or len(words) < 5:
        return None
    index = at[0]
    word = words[index]
    if (words[0] in ("is", "are") and index + 1 < len(words)
            and words[index + 1] == "than"):
        first, second = _phrase(words[1:index]), _phrase(words[index + 2:])
        if first and second:
            return "polar", word, first, second
        return None
    if words[0] in ("which", "what"):
        rest = words[index + 1:]
        if "or" in rest:
            split = rest.index("or")
            first, second = _phrase(rest[:split]), _phrase(rest[split + 1:])
            if first and second:
                return "choice", word, first, second
    return None


def _resolve(engine, word: str, dimension: str) -> tuple[str, str] | None:
    """(concept, the form of the word that found it) on this scale."""
    ratings = engine.profiles.ratings
    forms = [word]
    if word.endswith("es"):
        forms.append(word[:-2])
    if word.endswith("s"):
        forms.append(word[:-1])
    for form in forms:
        senses = [sense["id"] for sense in engine.reasoner.senses_of(form)
                  if sense.get("pos") == "n"]
        if not senses:
            continue
        pinned = pins.of(form)
        chosen = pinned or senses[0]
        concept = ratings.magnitude(dimension, chosen,
                                    [pinned] if pinned else senses)
        if concept is not None:
            return concept, form
    return None


def answer(engine, question: str) -> dict | None:
    read_as = read(question)
    if read_as is None:
        return None
    shape, word, first, second = read_as
    dimension, direction = COMPARATIVES[word]
    left = _resolve(engine, first, dimension)
    right = _resolve(engine, second, dimension)
    if left is None or right is None:
        return None                      # R18 refuses it, by name
    ratings = engine.profiles.ratings
    (a, a_word), (b, b_word) = left, right
    if dimension == "size":
        (a_value, a_start, a_end), (b_value, b_start, b_end) = (
            ratings.size[a], ratings.size[b])
        tie = SIZE_TIE
        apart = a_start > b_end or b_start > a_end
        scale = f"a real-world size scale with {ANCHORS}"
        shown = (f"{with_article(a_word)} at {a_value:.0f} and "
                 f"{with_article(b_word)} at {b_value:.0f}")
    else:
        a_value, b_value = ratings.heavy[a], ratings.heavy[b]
        tie, apart = HEAVY_TIE, abs(a_value - b_value) >= 3.0
        scale = "how heavy it is, rated 1 to 7"
        shown = (f"{with_article(a_word)} at {a_value:.1f} and "
                 f"{with_article(b_word)} at {b_value:.1f}")
    difference = a_value - b_value
    source = (f"THINGSplus asked people where each sits on {scale}: {shown}")
    steps = [
        engine._step(0, "R31", a, a_word,
                     f"{a_word}: {a} — rated {a_value:.1f}"),
        engine._step(1, "R31", b, b_word,
                     f"{b_word}: {b} — rated {b_value:.1f}"),
    ]
    comparison = {"dimension": dimension, "asked": word,
                  "first": {"word": a_word, "concept": a, "value": a_value},
                  "second": {"word": b_word, "concept": b, "value": b_value}}
    def read_as(payload: dict) -> dict:
        # The word the question used as its subject, for v688's sense rank.
        return engine._as_read(payload, a_word, f"{word} than {b_word}")

    if abs(difference) < tie:
        steps.append(engine._step(2, "R31", a, "too close",
                                  f"{abs(difference):.1f} apart: too close "
                                  f"to order", kind="stop"))
        note = (f"Too close to call. {source}, and a difference that small is "
                f"inside what people disagree on.")
        comparison["winner"] = None
        return read_as(engine._shell(question, "UNKNOWN", "R31", concept=a,
                                     note=note, steps=steps,
                                     extra={"comparison": comparison}))
    larger = a if difference > 0 else b
    larger_word = a_word if difference > 0 else b_word
    wanted = larger if direction > 0 else (b if larger == a else a)
    wanted_word = (larger_word if direction > 0
                   else (b_word if larger == a else a_word))
    comparison["winner"] = wanted_word
    clearly = "and their usual ranges do not overlap" if apart else (
        "though their usual ranges overlap")
    steps.append(engine._step(2, "R31", wanted, wanted_word,
                              f"{wanted_word} is {word}", kind="match"))
    if shape == "choice":
        note = f"{with_article(wanted_word).capitalize()}. {source}, {clearly}."
        payload = read_as(engine._shell(question, "LISTING", "R31", concept=a,
                                        note=note, steps=steps,
                                        extra={"comparison": comparison}))
        payload["parse"]["polar"] = False        # a choice, not a yes or no
        return payload
    verdict = "VERIFIED" if wanted == a else "CONTRADICTED"
    note = (f"{'Yes' if verdict == 'VERIFIED' else 'No'}: {source}, {clearly}.")
    return read_as(engine._shell(question, verdict, "R31", concept=a,
                                 note=note, steps=steps,
                                 extra={"comparison": comparison}))
