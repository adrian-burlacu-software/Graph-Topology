"""What colour, what shape, how big: a value, read from what a thing carries.

    what color is a banana            yellow
    what shape is a ball              round
    how big is an elephant            big, large -- and no measurement is stored
    how fast can a cheetah run        fast
    how much does an elephant weigh   heavy

There is no number anywhere in this data, which is why R18 refuses the
comparative and the superlative. A value is not a magnitude, though. The
norms record `is yellow` of a banana and `is big` of an elephant, and
WordNet's gloss of a banana says it is a yellow fruit. The question asks which
of one dimension's words is recorded of the thing, and that is a lookup over
what the concept already carries:

    its trie path    the norms' predicates, walked as R17 walks them
    its gloss        the definition, as written
    its properties   the store's `has_property` rows on the concept itself

Only the concept itself: a quality does not descend (E1 draws the same line
for individuals), and a mammal being `brown` says nothing about a whale.

The dimension decides which words count, and its words are few and fixed. A
value outside them -- `is a crescent` -- is not a colour, whatever else it is.

These were answered before as a listing of everything a banana is, headlined
AMBIGUOUS, because identification took `color` for the class being searched.
"""
from __future__ import annotations

import re

#: A dimension, and the words that are values of it.
DIMENSIONS = {
    "color": frozenset({"red", "orange", "yellow", "green", "blue", "purple",
                        "violet", "pink", "brown", "black", "white", "grey",
                        "gray", "golden", "silver", "tan", "beige", "cream",
                        "colorful", "colourful", "spotted", "striped",
                        "multicolored"}),
    "shape": frozenset({"round", "square", "rectangular", "oval", "spherical",
                        "cylindrical", "triangular", "flat", "long", "thin",
                        "pointed", "curved", "conical", "circular",
                        "elongated", "slender", "crescent-shaped", "coiled"}),
    "size": frozenset({"big", "large", "small", "little", "tiny", "huge",
                       "giant", "enormous", "massive", "miniature", "medium"}),
    "speed": frozenset({"fast", "slow", "quick", "quickly", "slowly", "rapid",
                        "swift"}),
    "weight": frozenset({"heavy", "light", "lightweight"}),
    "height": frozenset({"tall", "short", "high", "low"}),
    "length": frozenset({"long", "short"}),
    "age": frozenset({"old", "young", "ancient"}),
    "temperature": frozenset({"hot", "cold", "warm", "cool"}),
    "texture": frozenset({"soft", "hard", "smooth", "rough", "furry", "fuzzy",
                          "hairy", "scaly", "slimy", "sticky", "prickly",
                          "spiky", "silky", "fluffy", "wet", "dry"}),
}

#: Dimensions a number would measure. A word for one is not a measurement,
#: and the answer says so.
MEASURED = frozenset({"size", "speed", "weight", "height", "length", "age",
                      "temperature"})

#: The word a question asks with, and the dimension it asks about.
ASKED = {"color": "color", "colour": "color", "shape": "shape",
         "size": "size", "texture": "texture",
         "big": "size", "large": "size", "small": "size", "little": "size",
         "tiny": "size", "huge": "size", "fast": "speed", "slow": "speed",
         "quick": "speed", "heavy": "weight", "weigh": "weight",
         "tall": "height", "high": "height", "long": "length",
         "old": "age", "hot": "temperature", "cold": "temperature",
         "warm": "temperature", "soft": "texture", "hard": "texture"}

WHAT = re.compile(r"^what\s+(colou?r|shape|size|texture)\s+(?:is|are)\s+"
                  r"(?:an?\s+|the\s+)?([a-z][a-z ]*?)\s*$")
HOW = re.compile(r"^how\s+(big|large|small|little|tiny|huge|tall|high|long|"
                 r"fast|slow|quick|heavy|old|hot|cold|warm|soft|hard)\s+"
                 r"(?:is|are|can|could)\s+(?:an?\s+|the\s+)?([a-z][a-z ]*?)"
                 r"(?:\s+(?:run|swim|fly|go|move|travel|grow|get))?\s*$")
WEIGH = re.compile(r"^how\s+much\s+(?:does|do)\s+(?:an?\s+|the\s+)?"
                   r"([a-z][a-z ]*?)\s+weigh\s*$")

#: A predicate with one of these is a denial, not a value.
DENYING = frozenset({"not", "never", "cannot", "no", "isn't", "aren't",
                     "can't"})


def read(question: str) -> tuple[str, str] | None:
    """(dimension, the kind's word) for a value question, or None."""
    text = (question or "").strip().lower().rstrip("?").strip()
    found = WHAT.match(text)
    if found:
        return ASKED[found.group(1)], found.group(2).strip()
    found = HOW.match(text)
    if found:
        return ASKED[found.group(1)], found.group(2).strip()
    found = WEIGH.match(text)
    if found:
        return "weight", found.group(1).strip()
    return None


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z-]*", (text or "").lower())


def concept_of(engine, word: str) -> tuple[str | None, str]:
    """The sense a kind word is read as, and the word that found it."""
    reasoner = engine.reasoner
    for candidate in dict.fromkeys((word, word[:-1] if word.endswith("s")
                                    else word)):
        senses = reasoner.senses_of(candidate, "n") or []
        if senses:
            return senses[0]["id"], candidate
    return None, word


def answer(engine, question: str) -> dict | None:
    found = read(question)
    if found is None:
        return None
    dimension, word = found
    concept, word = concept_of(engine, word)
    if concept is None:
        return None
    vocabulary = DIMENSIONS[dimension]
    values: list[tuple[str, str, str]] = []

    def take(text: str, where: str) -> None:
        words = _words(text)
        # A denial is not a value. Nor, for something measured, is a part's:
        # `has small eyes` says nothing about how big an elephant is. A
        # surface's is: a banana is the colour of `has yellow skin`.
        if any(one in DENYING for one in words) or (
                dimension in MEASURED
                and words[:1] in (["has"], ["have"], ["with"])):
            return
        values.extend((one, where, text) for one in words
                      if one in vocabulary)

    profiles = engine.profiles
    name = profiles.named(word)
    if name:
        described = profiles.describe(name)
        for held in (described.path if described else []):
            take(held.predicate, "the norms")
    take(engine.reasoner.gloss(concept) or "", "its definition")
    for fact in engine.reasoner.facts_of(concept, "has_property")[:300]:
        if len(_words(fact.object)) <= 3:
            take(fact.object, fact.source)

    # The best source that says anything is the answer: what the norms and
    # the definition state of it, else what the norms say it only can be,
    # else the store's first few properties. The crawl records a banana
    # green, brown, black and purple as well as yellow, and all seven read as
    # one answer; the norms' `can have green skin` is a banana unripe, and
    # its definition's `yellow fruit` is the colour it has.
    def hedged(one):
        return one[1] == "the norms" and _words(one[2])[:1] == ["can"]

    firm = ("the norms", "its definition")
    tiers = [[one for one in values if one[1] in firm and not hedged(one)],
             [one for one in values if hedged(one)],
             [one for one in values if one[1] not in firm]]
    chosen = next((tier for tier in tiers if tier), [])
    if chosen and chosen is tiers[2]:
        first = list(dict.fromkeys(value for value, _, _ in chosen))[:3]
        chosen = [one for one in chosen if one[0] in first]
    values = chosen
    distinct = list(dict.fromkeys(value for value, _, _ in values))
    sources = list(dict.fromkeys(where for _, where, _ in values))
    measured = dimension in MEASURED
    if distinct:
        note = (f"Recorded of {word}: {', '.join(distinct)} — from "
                f"{', '.join(sources)}.")
    else:
        note = f"Nothing recorded of {word} says what its {dimension} is."
    if measured:
        note += (" Only a word for it is recorded, never a measurement: there "
                 "are no numbers in this data.")
    evidence = [{"value": value, "where": where, "text": text}
                for value, where, text in values[:12]]
    return engine._shell(
        question, "LISTING" if distinct else "UNKNOWN", "attribute",
        concept=concept, note=note,
        extra={"attribute": {"dimension": dimension, "kind": word,
                             "values": distinct, "evidence": evidence,
                             "measured": measured}})
