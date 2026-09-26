"""Asking how a thing is made: a recipe, asked for.

    do you have any good recipes for cake     a recipe asked for by name
    is there a recipe for bread
    do you know a good cake recipe
    how do you make a torch                    asked as how one is made
    how is a raft built

Read off v691's parse, by what the words are and not which words they
are: a noun is a recipe when a sense of it is a *direction* -- WordNet's
"a message describing how something is to be done" (recipe, formula,
instructions); a verb makes a thing when one of its first senses is a
kind of making (`make.v.03`, create): make, bake, build, brew, knit. *How
can I make the milk cold* is not one -- the milk is left in a state, and
that is the designer's (`page.how`).

What is said back is what is known, and what kind of knowing it is. A
recipe taught or seen (`carrying.Recipe`) is one. The store's `made_of`
is not: it says what a thing *can* be made of -- flour, eggs, butter --
with no steps, and is said as exactly that.
"""
from __future__ import annotations

from dataclasses import dataclass

from research.v691 import hearing
from research.v694 import knowing as K

#: What a recipe is, as WordNet has it: a message describing how
#: something is to be done.
DIRECTION = "direction.n.06"
#: What a verb that makes a thing is a kind of.
CREATE = "make.v.03"
#: How many of a word's senses are read: its common ones.
SENSES = 3
#: How many of the store's parts are said.
MOST_PARTS = 5
#: Requests put as an order: *give me a recipe for bread*.
REQUESTING = frozenset({"give", "tell", "show", "share", "suggest",
                        "recommend", "send"})


@dataclass(frozen=True)
class Asked:
    #: the thing to be made, as a name (`cake`)
    product: str
    #: as it was said (`cake`, `a torch`, `cookies`)
    said: str
    #: a recipe asked for by name, or how one is made
    by_name: bool
    plural: bool = False


def _senses(word: str, pos: str) -> list:
    try:
        from nltk.corpus import wordnet
        return wordnet.synsets(word.replace("-", "_"), pos)[:SENSES]
    except Exception:                              # noqa: BLE001
        return []


def _above(synset) -> set:
    return {one.name() for one in synset.closure(lambda s: s.hypernyms())}


def a_recipe(word: str) -> bool:
    """Whether a noun names directions for doing something: `recipe`,
    `formula`, `instructions`."""
    return any(DIRECTION in _above(one) or one.name() == DIRECTION
               for one in _senses(word, "n"))


def makes(verb: str) -> bool:
    """Whether a verb makes a thing: a kind of creating."""
    if verb in ("do", "be", "have"):
        return False
    return any(CREATE in _above(one) or one.name() == CREATE
               for one in _senses(verb, "v"))


def _lemma(word) -> str:
    """spaCy lemmatises *cookies* as `cooky`; WordNet's own morphology
    does not."""
    try:
        from nltk.corpus import wordnet
        return wordnet.morphy(word.text, wordnet.NOUN) or word.lemma
    except Exception:                              # noqa: BLE001
        return word.lemma


def _product(words, word) -> Asked | None:
    """The noun phrase headed by `word` as the thing to be made."""
    if word is None or word.tag not in ("NN", "NNS"):
        return None
    name = _lemma(word)
    if not K.is_a(name, "physical_entity.n.01") or K.lives(name):
        return None
    det = [one.text for one in hearing.children(words, word.index, {"det"})
           if one.text in ("a", "an", "the")]
    return Asked(K.name_of(name), " ".join(det[:1] + [word.text]), True,
                 word.tag == "NNS")


def _made_by(words, verb) -> Asked | None:
    """What a making verb makes: its object, or its passive subject."""
    for one in hearing.children(words, verb.index, {"dobj", "nsubjpass"}):
        return _product(words, one)
    return None


def _asking(words, text: str) -> bool:
    """A question, or a request for something said: not a statement."""
    first = words[0]
    return (text.rstrip().endswith("?")
            or first.tag in ("MD", "VBZ", "VBP", "VBD", "WRB", "WP", "WDT")
            or first.lemma in REQUESTING)


def read(text: str) -> Asked | None:
    """What an utterance asks to be told how to make, or None."""
    words = hearing.parse(text)
    if not words or not _asking(words, text):
        return None
    for word in words:
        if word.tag not in ("NN", "NNS") or not a_recipe(_lemma(word)):
            continue
        # `a recipe for cake`, `a recipe for making cookies`
        for prep in hearing.children(words, word.index, {"prep"}):
            if prep.text != "for":
                continue
            for obj in hearing.children(words, prep.index, {"pobj"}):
                found = _product(words, obj)
                if found is not None:
                    return found
            for doing in hearing.children(words, prep.index, {"pcomp"}):
                if makes(doing.lemma):
                    found = _made_by(words, doing)
                    if found is not None:
                        return found
        # `a cake recipe`
        for one in hearing.children(words, word.index, {"compound"}):
            found = _product(words, one)
            if found is not None:
                return found
        # `a recipe to make bread`
        for doing in hearing.children(words, word.index, {"acl", "relcl"}):
            if makes(doing.lemma):
                found = _made_by(words, doing)
                if found is not None:
                    return found
    if words[0].tag != "WRB" or words[0].text != "how":
        return None
    verb = next((one for one in words if one.index == words[0].head), None)
    if verb is None or not verb.tag.startswith("VB") or not makes(
            verb.lemma):
        return None
    if hearing.children(words, verb.index, {"ccomp", "oprd", "acomp",
                                            "xcomp"}):
        # `how can I make the milk cold`: a state, not a thing made.
        return None
    found = _made_by(words, verb)
    if found is None:
        return None
    return Asked(found.product, found.said, False, found.plural)


def parts(product: str) -> list:
    """What the store says one can be made of, best first: things one
    could have at hand -- a substance or an object -- and not a word for
    being a part (*ingredients*) or a misspelling WordNet does not have."""
    out, seen = [], set()
    for said, _ in K.rows_about(product, "made_of"):
        words = [one for one in said.lower().split() if one not in K.FILLER]
        if not words:
            continue
        head = K.lemmas(words[-1])
        name = next((one for one in sorted(head) if K.first_sense(one)),
                    "")
        if not name or name == product:
            continue
        if not K.is_a(name, "physical_entity.n.01") or K.is_a(
                name, "part.n.02"):
            continue
        if name not in seen:
            seen.add(name)
            out.append(" ".join(words))
        if len(out) == MOST_PARTS:
            break
    return out


def _listed(names) -> str:
    names = [one.replace("-", " ") for one in names]
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _a(name: str) -> str:
    said = name.replace("-", " ")
    return ("an " if said[:1] in "aeiou" else "a ") + said


def answer(asked: Asked, memory=None) -> str:
    """What is known of making it, said as the kind of knowing it is."""
    what = asked.said or asked.product
    known = memory.recipes_for(asked.product) if memory is not None else []
    exact = [one for one in known if one.product == asked.product]
    if exact:
        one = exact[0]
        how = f" -- {one.said}" if one.said else ""
        return (f"Yes: {_a(one.product)} can be made from "
                f"{_listed(_a(part) for part in one.parts)}{how}.")
    from_store = parts(asked.product)
    be = "are" if asked.plural else "is"
    if known:
        one = known[0]
        like = (f"I know how to make {_a(one.product)}, which is like "
                f"{_a(asked.product)}: from "
                f"{_listed(_a(part) for part in one.parts)} -- "
                f"{one.said or 'you told me how'}.")
        return (f"I have no recipe for {what}, but " + like[2:]
                if asked.by_name else like)
    if from_store:
        if asked.by_name:
            return (f"No, I have no recipe for {what} -- I only know what "
                    f"{what} can be made of: {_listed(from_store)}.")
        return (f"I don't know the steps, only what {what} can be made "
                f"of: {_listed(from_store)}.")
    if asked.by_name:
        return (f"No, I have no recipe for {what}, and nothing I know says "
                f"what {what} {be} made of.")
    return ""


def knows(asked: Asked, memory=None) -> bool:
    """Whether there is anything to say about making it."""
    return bool(answer(asked, memory))
