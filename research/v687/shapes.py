"""Four question shapes the taxonomy answers once they are read as asked.

    what is a dog a kind of               canine, carnivore, placental ... (R1)
    what are the parts of a car           its parts, both columns, and the norms'
    is a tomato a fruit or a vegetable    each put to R1: the one that holds
    is a dolphin like a fish              what the two share (R21), not a yes

Each of these came back as something else: the kind above as AMBIGUOUS from
identification, the parts as a listing about a concept named `part`, the
choice as `is_a` to a concept called `fruit or a vegetable`, and likeness as
VERIFIED because the norms list the word `fish` for a dolphin.
"""
from __future__ import annotations

import re

KIND_OF = re.compile(r"^what\s+(?:is|are)\s+(?:an?\s+|the\s+)?([a-z][a-z ]*?)"
                     r"\s+(?:an?\s+)?(?:kind|type|sort)\s+of\s*$")
KIND_OF_THING = re.compile(r"^what\s+(?:kind|type|sort)\s+of\s+(?:thing|"
                           r"animal|object|creature)\s+(?:is|are)\s+(?:an?\s+|"
                           r"the\s+)?([a-z][a-z ]*?)\s*$")
PARTS = re.compile(r"^what\s+(?:are\s+the\s+parts\s+of|parts\s+(?:does|do)|"
                   r"is\s+(?:an?\s+|the\s+)?[a-z]+\s+made\s+up\s+of)"
                   r"\s+(?:an?\s+|the\s+)?([a-z][a-z ]*?)(?:\s+have)?\s*$")
CHOICE = re.compile(r"^(?:is|are)\s+(?:an?\s+)(.+?)\s+(?:an?\s+)(.+?)\s+or\s+"
                    r"(?:an?\s+)?(.+?)\s*$")
LIKE = re.compile(r"^(?:is|are)\s+(?:an?\s+)?([a-z][a-z ]*?)\s+(?:like|"
                  r"similar\s+to)\s+(?:an?\s+)?([a-z][a-z ]*?)\s*$")

#: Ancestors a `kind of` answer names before it stops.
ABOVE = 8


def _name(concept: str) -> str:
    parts = str(concept or "").rsplit(".", 2)
    if len(parts) == 3 and parts[2].isdigit() and len(parts[1]) == 1:
        return parts[0]
    return str(concept or "")


def _sense(engine, word: str) -> str | None:
    for candidate in dict.fromkeys((word, word[:-1] if word.endswith("s")
                                    else word)):
        senses = engine.reasoner.senses_of(candidate, "n") or []
        if senses:
            return senses[0]["id"]
    return None


def kind_of(engine, question: str, word: str) -> dict | None:
    concept = _sense(engine, word)
    if concept is None:
        return None
    chain = [(node, distance) for node, distance, _ in
             engine.reasoner.ascend(concept) if distance][:ABOVE]
    names = list(dict.fromkeys(_name(node) for node, _ in chain))
    if not names:
        return engine._shell(question, "UNKNOWN", "R1", concept=concept,
                             note=f"Nothing is recorded above {concept}.")
    return engine._shell(
        question, "LISTING", "R1", concept=concept,
        note=(f"{word} is a kind of {', '.join(names)}, nearest first "
              f"(R1)."),
        extra={"above": {"kind": word, "concept": concept, "chain": names}})


def parts_of(engine, question: str, word: str) -> dict | None:
    concept = _sense(engine, word)
    if concept is None:
        return None
    parts: list[str] = []
    # Both columns: the meronym rows are stored inverted relative to their
    # names, so `(car.n.01, part_of, accelerator.n.01)` is a part of a car.
    for relation in ("part_of", "has_part", "has_a"):
        for fact in engine.reasoner.facts_of(concept, relation)[:60]:
            if fact.relation == relation:
                parts.append(_name(fact.object))
    profiles = engine.profiles
    name = profiles.named(word)
    described = profiles.describe(name) if name else None
    for held in (described.path if described else []):
        predicate = held.predicate
        if predicate.startswith(("has a ", "has an ", "has ")):
            parts.append(predicate.split(" ", 1)[1])
    parts = list(dict.fromkeys(one for one in parts if one))
    if not parts:
        return engine._shell(question, "UNKNOWN", "R22", concept=concept,
                             note=f"No parts of {word} are recorded.")
    return engine._shell(
        question, "LISTING", "R22", concept=concept,
        note=f"{len(parts)} parts of {word} recorded.",
        extra={"parts": {"kind": word, "concept": concept, "parts": parts}})


def choice(engine, question: str, subject: str, first: str,
           second: str) -> dict | None:
    options = []
    for option in (first, second):
        payload = engine.ask(f"is {_article(subject)} {subject} "
                             f"{_article(option)} {option}")
        options.append({"option": option, "verdict": payload.get("verdict"),
                        "note": (payload.get("note") or "")[:300]})
    holds = [one["option"] for one in options if one["verdict"] == "VERIFIED"]
    if not holds:
        note = (f"Neither is recorded above {subject}: "
                + "; ".join(f"{one['option']} {one['verdict']}"
                            for one in options))
    else:
        note = (f"{subject} is filed under {' and '.join(holds)} (R1); "
                + "; ".join(f"{one['option']} {one['verdict']}"
                            for one in options))
    return engine._shell(
        question, "LISTING" if holds else "UNKNOWN", "R1", concept=subject,
        note=note, extra={"choice": {"subject": subject, "options": options,
                                     "holds": holds}})


def _article(word: str) -> str:
    return "an" if (word or "")[:1] in "aeiou" else "a"


def like(engine, question: str, left: str, right: str) -> dict | None:
    """Likeness is a degree: the comparison R21 already makes."""
    compared = engine._contrast(
        f"what do {_article(left)} {left} and {_article(right)} {right} "
        f"have in common")
    if compared is None:
        return None
    compared["question"] = question
    return compared


def answer(engine, question: str) -> dict | None:
    text = (question or "").strip().lower().rstrip("?").strip()
    found = KIND_OF.match(text) or KIND_OF_THING.match(text)
    if found:
        return kind_of(engine, question, found.group(1).strip())
    found = PARTS.match(text)
    if found:
        return parts_of(engine, question, found.group(1).strip())
    found = CHOICE.match(text)
    if found:
        return choice(engine, question, *(part.strip()
                                          for part in found.groups()))
    found = LIKE.match(text)
    if found:
        return like(engine, question, found.group(1).strip(),
                    found.group(2).strip())
    padded = f" {text} "
    if (" is to " in padded or " are to " in padded) and " as " in padded:
        # R24 reads one shape, and anything else it did not take was answered
        # as a listing about its first word: `wing is to bird as fin is to
        # what` came back with everything a wing is.
        return engine._shell(
            question, "UNKNOWN", "R24",
            note=("An analogy is read in one shape: `fins are to fish as what "
                  "are to birds`, with what is asked for put as `what`, before "
                  "the last kind. Put that way, it is answered over the norms."))
    return None
