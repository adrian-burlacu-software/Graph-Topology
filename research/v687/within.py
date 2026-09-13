"""Which of a class do something: the kinds beneath it, asked one by one.

    which birds cannot fly         chicken, emu, penguin
    which dogs bark                collie, dalmatian, german shepherd
    what mammals lay eggs          platypus
    which animals can swim         the norms' kinds, and the store's beyond

`do all birds fly` already puts its predicate to every kind of bird the norms
describe and counts (`Profiles._quantified`, R20): 24 fly, and chicken, emu and
penguin do not. That count *is* the answer to `which birds cannot fly` -- the
kinds on each side are the list -- but the question went to identification,
which looks for one unnamed thing carrying every property named, and came
back AMBIGUOUS.

Two readings, over the same taxonomy:

    norms   the kinds of the class the feature norms describe, each asked
            down its trie path (R17) and HELD, INHERITED or DENIED
    store   for a claim, never a denial: the backwards reading (R22) over the
            whole store, kept where the subject sits under the class (R1)

A denial is read only from the norms. They put a fixed question to every
concept they cover, so a no in them is a no; a crawl records what things do,
and silence in it is not.

Only a plural class is taken: `what animal has a trunk` asks for one thing,
and identification (R16) answers it.
"""
from __future__ import annotations

from . import logic, profile

OPENERS = ("which", "what")

#: Words after the opener that are not a class: `what kinds of dog`, `which is
#: heavier`, `which one is black`.
NOT_A_CLASS = frozenset({"is", "are", "was", "were", "do", "does", "did",
                         "can", "could", "has", "have", "kind", "kinds",
                         "type", "types", "sort", "sorts", "one", "ones",
                         "of", "if"})

#: Verbs a backwards question opens with, which end in `s` like a plural.
VERBS = frozenset({"causes", "makes", "eats", "lives", "needs", "wants",
                   "contains", "includes", "has", "uses", "happens",
                   "explains", "means", "does", "goes", "comes", "grows",
                   "holds", "carries", "produces", "lays", "hunts", "kills"})

#: What turns a claim into a denial.
NEGATORS = {"not": "", "cannot": "can", "can't": "can", "don't": "do",
            "doesn't": "does", "aren't": "are", "isn't": "is",
            "never": "", "no": ""}

#: The kinds verdicts that do, and that do not.
DOES = ("HELD", "INHERITED")
DOES_NOT = ("DENIED",)

#: How many subjects the backwards reading may bring back before the class
#: filters them.
BACKWARDS = 200


def _name(concept: str) -> str:
    parts = str(concept or "").rsplit(".", 2)
    if len(parts) == 3 and parts[2].isdigit() and len(parts[1]) == 1:
        return parts[0]
    return str(concept or "")


def read(question: str) -> tuple[str, str, bool] | None:
    """(class word, predicate, negative) for `which birds cannot fly`."""
    words = (question or "").strip().lower().rstrip("?").split()
    if len(words) < 3 or words[0] not in OPENERS:
        return None
    word = words[1]
    if word in NOT_A_CLASS or not word.endswith("s") or word.endswith("ss"):
        return None
    tail = words[2:]
    # `what causes fire`, `what eats a mouse`: a verb and its object, read
    # backwards by R22 -- `causes` is not a class of things that fire.
    if word in VERBS or tail[:1] in (["a"], ["an"], ["the"]):
        return None
    negative = any(one in NEGATORS for one in tail)
    kept = [NEGATORS.get(one, one) for one in tail]
    predicate = " ".join(one for one in kept if one)
    return (word, predicate, negative) if predicate else None


def store_kinds(engine, name: str, predicate: str) -> list[str]:
    """What the store says does this, among the kinds of the class."""
    reasoner = engine.reasoner
    classes = {sense["id"] for sense in reasoner.senses_of(name, "n") or []}
    if not classes:
        return []
    found = engine.inverse.answer(f"what {predicate}", limit=BACKWARDS)
    kept: list[str] = []
    for subject in (getattr(found, "subjects", None) or []):
        concept = subject.get("concept") if isinstance(subject, dict) else ""
        if not concept:
            continue
        candidates = ([concept] if ".n." in concept else
                      [sense["id"] for sense in
                       (reasoner.senses_of(concept, "n") or [])[:3]])
        if any(node in classes
               for one in candidates
               for node, distance, _ in reasoner.ascend(one) if distance):
            kept.append(_name(concept))
    return list(dict.fromkeys(kept))


def answer(engine, question: str) -> dict | None:
    """The kinds of a class that do something, or None to leave it alone."""
    found = read(question)
    if found is None:
        return None
    word, predicate, negative = found
    profiles = engine.profiles
    name = profiles.subject(f"all {word}")
    if name is None or not profiles.subtypes(name):
        return None
    query = logic.parse(predicate, profile.ASIDE)
    verdict = profiles.assess(name, logic.Query("some", query.tree))
    members = list(getattr(verdict, "members", None) or [])
    fits = [one["name"] for one in members
            if one.get("verdict") in (DOES_NOT if negative else DOES)]
    others = [one["name"] for one in members
              if one.get("verdict") in (DOES if negative else DOES_NOT)]
    # The store is read only where the norms decided nothing. Their kinds
    # were each asked the same question; the crawl, read backwards, adds
    # rabbits and goats to the mammals that lay eggs.
    store = [] if negative or members else [
        one for one in store_kinds(engine, name, predicate) if one not in fits]
    kinds = len(profiles.subtypes(name))
    said = "do not" if negative else "do"
    note = (f"Asked of the kinds of {name} the norms describe: "
            f"{len(members)} of {kinds} decide, and {len(fits)} {said} "
            f"{predicate}.")
    if store:
        note += (f" The store records {len(store)} more under {name} that "
                 f"{predicate}, read backwards (R22).")
    if negative:
        note += (" A denial is read from the norms alone: a crawl's silence "
                 "is not a no.")
    within = {"class": name, "predicate": predicate, "negative": negative,
              "norms": fits, "others": others, "store": store,
              "decided": len(members), "kinds": kinds}
    return engine._shell(question, "LISTING" if fits or store else "UNKNOWN",
                         "R20", concept=name, note=note,
                         extra={"within": within})
