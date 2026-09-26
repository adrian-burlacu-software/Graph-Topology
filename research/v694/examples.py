"""Examples asked for: which ones there are, for a purpose.

    do you know any jokes                     kinds of joke I know of
    do you have any tools for gardening       none of mine; some I know of
    can you suggest some games for kids       what the store says fits
    do you know any good books for children

Read off v691's parse: the one asked is `you`, the verb is knowing or
having (or a request -- suggest, recommend, name), and the thing is put
with `any` or `some`. *Do you have a dog* asks whether I have one, and is
v689's (`participants.py`); *any tools for gardening* asks for tools.

The candidates are the kinds of the thing: WordNet's, beneath its first
sense, and the store's own -- a pet is a role WordNet has no kinds under,
and the store says a cat *makes a good pet*. A purpose narrows them the
way the designer finds means (`knowing.serving`): tools for gardening are
the kinds of tool the store says serve gardening. What is said is only
what was found, and nothing is said to fit a purpose the store is silent
on.
"""
from __future__ import annotations

from dataclasses import dataclass

from research.v691 import hearing
from research.v694 import knowing as K

#: What asks for examples, said to the one asked: *do you know any*, and a
#: request -- *can you suggest some*.
KNOWING = frozenset({"know"})
HAVING = frozenset({"have", "got", "own"})
REQUESTS = frozenset({"suggest", "recommend", "name", "list", "give",
                      "tell", "share"})
#: What puts a thing as some of them: `any jokes`, `a few games`.
SOME = frozenset({"any", "some", "few", "several"})
#: A number asked for: `name three birds`.
NUMBERS = {word: number for number, word in enumerate(
    "one two three four five six seven eight nine ten".split(), start=1)}
#: How many are said, and how many kinds a thing may have below it before
#: a purpose is needed to pick among them.
SAID = 5
#: The verbs a row uses to say a thing *is* one: `make a good pet`.
BEING = frozenset({"be", "make", "become"})


@dataclass(frozen=True)
class Wanted:
    kind: str
    said: str
    #: what they are for: `gardening`, `children`, or ""
    purpose: str
    #: know | have | request
    verb: str
    plural: bool = True
    #: how many were asked for, or 0 for some
    many: int = 0


def _noun(word) -> str:
    try:
        from nltk.corpus import wordnet
        forms = wordnet._morphy(word.text, wordnet.NOUN)
    except Exception:                              # noqa: BLE001
        return word.lemma
    if word.tag == "NNS":
        other = [one for one in forms if one != word.text]
        if other:
            return other[0]
    return forms[0] if forms else word.lemma


def _purpose(words, thing) -> str:
    for prep in hearing.children(words, thing.index, {"prep"}):
        if prep.text != "for":
            continue
        for obj in hearing.children(words, prep.index, {"pobj", "pcomp"}):
            return obj.text
    return ""


def read(text: str) -> Wanted | None:
    """Examples an utterance asks for, or None."""
    words = hearing.parse(text)
    if not words:
        return None
    root = next((one for one in words if one.dep == "ROOT"), None)
    # `name three birds`: tagged a noun, and read as the verb it is by the
    # object it takes.
    if root is None or not (root.tag.startswith("VB") or hearing.children(
            words, root.index, {"dobj"})):
        return None
    subject = hearing.children(words, root.index, {"nsubj"})
    verb = root.lemma
    if verb in REQUESTS:
        if subject and subject[0].text != "you":
            return None
        how = "request"
    elif verb in KNOWING | HAVING:
        if not subject or subject[0].text != "you":
            return None
        asked = text.rstrip().endswith("?") or any(
            one.dep == "aux" and one.index < subject[0].index
            for one in hearing.children(words, root.index))
        if not asked:
            return None
        how = "know" if verb in KNOWING else "have"
    else:
        return None
    thing = next(iter(hearing.children(words, root.index, {"dobj"})), None)
    if thing is None or thing.tag not in ("NN", "NNS"):
        return None
    quantity = {one.text for one in hearing.children(
        words, thing.index, {"det", "amod", "nummod"})}
    many = next((NUMBERS[one] for one in quantity if one in NUMBERS), 0)
    if many:
        quantity.add("some")
    if not quantity & SOME and not (how == "request" and thing.tag == "NNS"):
        return None
    purpose = _purpose(words, thing)
    if how == "have" and not purpose:
        # `do you have any pets` asks what I have (`participants.py`).
        return None
    said = thing.text + (f" for {purpose}" if purpose else "")
    return Wanted(_noun(thing), said, purpose, how, thing.tag == "NNS",
                  many)


def _beneath(kind: str) -> list:
    """Kinds of it, WordNet's, nearest first, beneath its first sense."""
    first = K.first_sense(kind)
    if first is None:
        return []
    out, level = [], [first]
    while level:
        below = []
        for one in level:
            for child in one.hyponyms():
                name = K.name_of(child.lemma_names()[0].lower())
                if name not in out:
                    out.append(name)
                below.append(child)
        level = below
    return out


def _told_as(kind: str) -> list:
    """What the store says is one, where WordNet has no kinds of it: a
    row whose words are only being one -- `make a good pet`, `be a pet`,
    `is_a pet` -- of something that is what one of them is: a pet is an
    animal, so a tenant that `makes a good pet` of nothing is not one."""
    forms = K.lemmas(kind)
    first = K.first_sense(kind)
    above = [one.name() for one in first.hypernyms()] if first else []
    out = []
    for concept, relation, said in K.connection().execute(
            "SELECT concept, relation, object FROM facts WHERE relation IN "
            "('is_a', 'capable_of', 'has_property') AND (object = ? OR "
            "object LIKE ?) ORDER BY confidence DESC LIMIT 400",
            (kind, f"% {kind}")):
        words = [one for one in said.lower().split() if one not in K.FILLER]
        if not words or not (K.lemmas(words[-1]) & forms):
            continue
        rest = words[:-1]
        if relation != "is_a" and (not rest or rest[0] not in BEING):
            continue
        if any(K.first_sense(one) is not None and not _describing(one)
               for one in rest[1:]):
            continue
        name = K.name_of(concept)
        if above and not K.is_a(name, *above):
            continue
        if name not in out and name != kind:
            out.append(name)
    return out


def _is_one(name: str, kind: str) -> bool:
    """Whether a word's common sense is a kind of this one's: a taxpayer
    is not a book, whatever else `taxpayer` names."""
    first = K.first_sense(kind)
    return first is not None and K.is_a(name, first.name())


def _purposes(word: str) -> frozenset:
    """A purpose's words: its lemmas and its common sense's synonyms --
    games for kids are games for children."""
    out = set(K.lemmas(word))
    for form in list(out):
        first = K.first_sense(form)
        if first is not None:
            out.update(one.lower() for one in first.lemma_names()
                       if "_" not in one)
    return frozenset(out)


#: How many kinds are weighed for which are best known.
WEIGHED = 400


def _familiar(names: list) -> list:
    """The best known first: the ones the store says most about. Birds
    one has heard of are parrots and hens, not apodiform birds."""
    said = {}
    for name in names[:WEIGHED]:
        word = name.replace("-", " ")
        # A range, not LIKE, so the concept index is used: `robin.n.01`
        # and every other noun sense of it sort between these two.
        said[name] = K.connection().execute(
            "SELECT COUNT(*) FROM facts WHERE concept = ? OR (concept >= ? "
            "AND concept < ?)", (word, word + ".n.", word + ".n/")
        ).fetchone()[0]
    return sorted(names[:WEIGHED], key=lambda one: -said[one])


def _describing(word: str) -> bool:
    try:
        from nltk.corpus import wordnet
        return bool(wordnet.synsets(word, wordnet.ADJ))
    except Exception:                              # noqa: BLE001
        return False


def examples(wanted: Wanted) -> list:
    """(name, why) for kinds of the thing that fit the purpose, best
    first."""
    candidates = [one for one in _beneath(wanted.kind)
                  if _is_one(one, wanted.kind)]
    candidates += [one for one in _told_as(wanted.kind)
                   if one not in candidates]
    if not wanted.purpose:
        return [(one, "") for one in
                _familiar(candidates)[:wanted.many or SAID]]
    within = set(candidates)
    verb = K.verb_of(wanted.purpose)
    found: dict = {}
    if verb and verb != wanted.purpose:
        # `for gardening`: what serves the doing, as the designer asks.
        for means in K.serving(verb=verb):
            if means.name in within and means.name not in found:
                found[means.name] = (means.score, means.why())
        for name in K.named_tools(verb):
            if name in within and name not in found:
                found[name] = (0.5, f"the word {verb} names it")
    else:
        # `for children`: what the store says is for them.
        forms = _purposes(wanted.purpose)
        for row in K.index().mentioning(forms):
            name = K.name_of(row.thing)
            if name not in within or not row.words:
                continue
            if row.words[0][1] & forms:
                continue                     # `children` doing it
            if name not in found or found[name][0] < row.confidence:
                relation = {"used_for": "is used for", "capable_of": "can",
                            "has_property": "is"}.get(row.relation)
                found[name] = (row.confidence,
                               f"{_a(name)} {relation} {row.said}")
    ranked = sorted(found.items(), key=lambda one: -one[1][0])
    return [(name, why) for name, (_, why) in
            ranked[:wanted.many or SAID]]


def _a(name: str) -> str:
    said = name.replace("-", " ")
    return ("an " if said[:1] in "aeiou" else "a ") + said


def _listed(names) -> str:
    names = list(names)
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def answer(wanted: Wanted) -> str:
    """Some I know of, and what says they fit -- or that I know of none."""
    found = examples(wanted)
    said = wanted.said
    if not found:
        if wanted.verb == "have":
            return ""
        return f"I don't know of any {said}."
    names = _listed(name.replace("-", " ") for name, _ in found)
    why = next((one for _, one in found if one), "")
    because = f" -- {why}" if why else ""
    if wanted.verb == "have":
        return (f"I have no {wanted.kind.replace('-', ' ')}s of my own, but "
                f"I know of some {said}: {names}{because}.")
    return f"Some {said} I know of: {names}{because}."
