"""What an utterance says about a world, read off its dependency parse.

`openworld.py` read English with five regular expressions and a list of the
verbs that move things: `get|put|move|take|bring|carry|send|place`. That was
enough to show the planner is general, and it made the planner's reach the
reader's: `how would a pig fly?` was not read at all, `carry the box into
the garden` only because `carry` was on the list, and `the door is still
closed` came out as a door in the state *still*.

This reads the parse instead, and asks VerbNet what a verb does rather than
keeping a list of verbs:

    the book is in the shop            at book shop       be + place
    the door is closed / locked        closed door        be + state
    john has the book                  with book john     have
    there is a cup on the table        at cup table       there is
    put the pig on the plane           at pig plane       VerbNet: put-9.1
    give the cup to mary               with cup mary      VerbNet: give-13.1
    open the door                      open door          VerbNet: open
    how would a pig fly?               fly pig            a doing, asked
    make a pig fly                     fly pig            a doing, caused

An order's goal is **what VerbNet says the verb changes** in a sentence of
that shape (`change.effects`, the same reading v689 does for T4): a verb
that moves its object to the place after the preposition asks for the
object to be there, whatever the verb is. That is the reader and the
planner agreeing on what a verb means, from one source.

## The parse

The one model every layer reads with (`v687.language.load`): spaCy's
transformer where it is installed, the small one where it is not. The small
one tags `fly` in *how would a pig fly* as a noun and `mary` in *give the cup
to mary* as a verb; the transformer reads both.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.v689.clauses import Word

#: Prepositions that put a thing somewhere, for `is in/at/on the X`.
PLACES = frozenset({"in", "at", "on", "inside", "within", "aboard", "into",
                    "onto", "under", "beside", "behind"})
#: What a subject hangs off a verb by.
SUBJECTS = frozenset({"nsubj", "nsubjpass", "expl"})
#: Possessive determiners and articles: said before a common noun.
ARTICLES = frozenset({"the", "a", "an", "some", "my", "your", "his", "her",
                      "their", "its", "our", "this", "that", "these",
                      "those"})
#: A clause that asks for something done rather than saying what is:
#: `can you`, `could you`, `would you`, `will you`.
REQUESTS = frozenset({"can", "could", "would", "will"})
#: The people talking, who are not things in the scene: `how do i open the
#: door` is about the door.
PERSONS = frozenset({"i", "me", "we", "us", "you"})

_NLP = None


def nlp():
    """The parser, loaded once: the transformer model if it is there."""
    global _NLP
    if _NLP is None:
        from research.v687.language import load
        _NLP = load() or False
    return _NLP or None


_CACHE: dict = {}


def parse(text: str) -> list[Word]:
    """`Word`s -- v689's own type -- with lemmas, for one utterance."""
    plain = " ".join(text.lower().split())
    if plain in _CACHE:
        return _CACHE[plain]
    parser = nlp()
    if parser is None:
        return []
    words = [Word(token.i, token.text.lower(), token.tag_, token.dep_,
                  token.head.i, token.head.text.lower(),
                  token.lemma_.lower()) for token in parser(plain)]
    if len(_CACHE) > 512:
        _CACHE.clear()
    _CACHE[plain] = words
    return words


def children(words: list[Word], index: int, deps=None) -> list[Word]:
    return [one for one in words if one.head == index
            and one.index != index and (deps is None or one.dep in deps)]


def thing(words: list[Word], word: Word | None) -> str:
    """The name a noun phrase is kept under: its head noun, or the name
    it is. `the red block` is `block`; `my pig` is `pig`."""
    if word is None:
        return ""
    if word.dep == "expl" or word.tag in ("WP", "WDT", "WRB", "WP$"):
        # `there`, and `what` in `what would it take`: no thing's name.
        return ""
    if word.text in PERSONS:
        return ""
    return word.text if word.text.isalpha() else ""


@dataclass
class Heard:
    """What one utterance says: facts, what it asks for, and how."""

    #: facts it states, as literals
    facts: list = field(default_factory=list)
    #: facts it denies: `the door did not open` -> `open door`
    denied: list = field(default_factory=list)
    #: facts said to *still* hold
    still: list = field(default_factory=list)
    #: what it asks for, as literals
    wants: list = field(default_factory=list)
    #: predicates asked for as what a thing does, not a state to leave it in
    doings: set = field(default_factory=set)
    #: an order (`open the door`, `can you open the door`)
    order: bool = False
    #: a question about what it would take (`how would a pig fly`)
    asked: bool = False
    #: the verbs an order or question was said with: `carry` in `carry the
    #: box into the garden`, tried first however VerbNet ranks it
    verbs: list = field(default_factory=list)
    #: (predicate, verb) for what a statement says was done: `sam went to
    #: the park` is a way `at` comes about (`done`)
    done: list = field(default_factory=list)
    #: the preposition each place of an order was said with: `into` for the
    #: garden in `carry the box into the garden`, so it is said back so
    preps: dict = field(default_factory=dict)


def hear(text: str, stated=None, it: str = "") -> Heard:
    """Read one utterance. `stated(verb)` says whether a verb names a state
    (`verbs.stated`), for `make X V`; `it` is what `it` refers to."""
    words = parse(text)
    out = Heard()
    if not words:
        return out
    name = {one.index: (it if one.text in ("it", "they") and it else
                        thing(words, one)) for one in words}
    # A question states nothing: `what steps are required` is not news
    # about steps, and `does a table have legs` is v688's.
    question = (words[0].tag in ("WRB", "WP", "WDT")
                or (words[0].tag in ("MD", "VBZ", "VBP", "VBD")
                    and words[0].lemma in ("be", "do", "have", "can",
                                           "could", "would", "will",
                                           "should", "may", "might"))
                or text.rstrip().endswith("?"))
    for verb in words:
        if not (verb.tag.startswith("VB") or verb.tag in ("JJ", "MD")):
            continue
        if verb.tag == "JJ" and verb.dep not in ("ROOT", "ccomp", "conj"):
            continue
        subject = next(iter(children(words, verb.index, SUBJECTS)), None)
        negated = bool(children(words, verb.index, {"neg"}))
        still = any(one.lemma == "still" for one in
                    children(words, verb.index, {"advmod"}))
        auxiliaries = children(words, verb.index, {"aux", "auxpass"})
        is_be = verb.lemma == "be"
        if is_be or (auxiliaries and verb.tag in ("VBN", "JJ")
                     and any(one.lemma == "be" for one in auxiliaries)):
            if not question:
                _stated(words, verb, subject, name, negated, still, out)
            continue
        if verb.lemma in ("have", "hold") and subject is not None \
                and not question:
            for held in children(words, verb.index, {"dobj"}):
                out.facts.append(f"with {name[held.index]} "
                                 f"{name[subject.index]}")
            continue
        _asked_for(words, verb, subject, name, negated, question,
                   auxiliaries, stated, out)
        if (not question and not negated and subject is not None
                and verb.tag in ("VBD", "VBZ", "VBN")
                and verb.lemma not in ("be", "have", "do")):
            out.done += _done(words, verb)
    return out


def _done(words, verb) -> list:
    """What a statement says was brought about, and by which verb: `sam
    went to the park`, `she carried the bag home` -- a location VerbNet
    says changed. What people say they did is how the verbs they would use
    are learned (`learned.prefer`)."""
    from research.v689 import change

    obj = next(iter(children(words, verb.index, {"dobj"})), None)
    preposition = next((one.text for one in children(
        words, verb.index, {"prep"}) if one.text in PLACES | {"to"}), "")
    try:
        found = change.effects(verb.lemma, obj is not None, preposition)
    except Exception:                              # noqa: BLE001
        return []
    if any(one.kind == "location" and one.after for one in found):
        return [("at", verb.lemma)]
    return []


def _stated(words, verb, subject, name, negated, still, out) -> None:
    """`X is Y`, `X is in Y`, `X is locked`, `there is X on Y`."""
    if subject is not None and subject.dep == "expl":
        subject = next(iter(children(words, verb.index, {"attr"})), None)
    if subject is None or not name.get(subject.index):
        return
    who = name[subject.index]
    found = []
    if verb.lemma != "be":
        # `the door is locked`: the participle is the state.
        found.append(f"{verb.text} {who}")
    for one in children(words, verb.index, {"acomp", "attr"}):
        if one.index != subject.index and one.tag.startswith(("JJ", "VBN")):
            found.append(f"{one.text} {who}")
    places = children(words, verb.index, {"prep"})
    if subject.dep == "attr":
        places += children(words, subject.index, {"prep"})
    for prep in places:
        if prep.text not in PLACES:
            continue
        for where in children(words, prep.index, {"pobj"}):
            if name.get(where.index):
                found.append(f"at {who} {name[where.index]}")
    for fact in found:
        (out.denied if negated else out.facts).append(fact)
        if still and not negated:
            out.still.append(fact)


def _asked_for(words, verb, subject, name, negated, question, auxiliaries,
               stated, out) -> None:
    """An order, a request, or a question about what it would take."""
    from research.v689 import change

    lemma = verb.lemma
    request = (subject is not None and subject.text == "you"
               and any(one.text in REQUESTS for one in auxiliaries))
    imperative = (subject is None and verb.dep == "ROOT"
                  and verb.tag == "VB")
    # Asking what it would take: `how would a pig fly`, `what would it
    # take to get the book home`. A question that only asks whether --
    # `does a table have legs` -- is not asking for a plan, and nor is
    # `what do you need to bake a cake`, which asks what a cake takes and
    # is v688's (the probe caught it).
    asking = (question and verb.tag == "VB" and not request
              and ((_how(words, verb) and _would(auxiliaries, subject))
                   or words[verb.head].lemma == "take"))
    causing = (verb.dep in ("ccomp", "xcomp") and verb.tag == "VB"
               and words[verb.head].lemma in ("make", "let", "have", "help")
               and subject is not None)
    if negated:
        # `the door did not open`: what should have happened, denied.
        if subject is not None and name.get(subject.index):
            out.denied.append(f"{lemma} {name[subject.index]}")
        return
    if causing:
        who = name.get(subject.index, "")
        if who:
            out.wants.append(f"{lemma} {who}")
            if stated is None or not stated(lemma):
                out.doings.add(lemma)
            out.asked = out.asked or question
            out.order = out.order or not question
        return
    if lemma == "make" and not causing:
        # `make the door open`: the complement is the state.
        obj = next(iter(children(words, verb.index, {"dobj"})), None)
        for one in children(words, verb.index, {"ccomp", "xcomp", "oprd",
                                                "acomp"}):
            if obj is not None and one.tag.startswith("JJ"):
                out.wants.append(f"{one.text} {name[obj.index]}")
        if out.wants:
            out.order = out.order or not question
            out.asked = out.asked or question
        return
    if not (request or imperative or asking):
        return

    obj = next(iter(children(words, verb.index, {"dobj"})), None)
    place, preposition = None, ""
    for prep in children(words, verb.index, {"prep", "dative"}):
        target = next(iter(children(words, prep.index, {"pobj"})), None)
        if target is not None:
            place, preposition = target, prep.text
    wanted = _effects(lemma, obj, place, preposition, name)
    if place is not None and name.get(place.index) and preposition:
        out.preps[name[place.index]] = preposition
    if not wanted and obj is not None and name.get(obj.index):
        wanted = [f"{lemma} {name[obj.index]}"]
    if not wanted and obj is None and subject is not None \
            and subject.text != "you" and name.get(subject.index):
        # `how would a pig fly`: something the subject does.
        wanted = [f"{lemma} {name[subject.index]}"]
        out.doings.add(lemma)
    out.wants += wanted
    if wanted:
        out.verbs.append(lemma)
    out.order = out.order or request or (imperative and not asking)
    out.asked = out.asked or asking


def _how(words, verb) -> bool:
    """`how` asking the manner of this verb -- not `how many`, which asks
    a count and hangs off `many`."""
    return any(one.lemma == "how" and one.head == verb.index
               for one in words)


#: What makes `how ... V` ask what it would take rather than what happened:
#: `how would a pig fly`, `how can i open it`; not `how did the dog bark`.
WOULD = frozenset({"would", "could", "can", "should", "might", "may",
                   "will"})


def _would(auxiliaries, subject) -> bool:
    for one in auxiliaries:
        if one.text in WOULD:
            return True
        if one.lemma == "do" and one.tag != "VBD" and subject is not None                 and subject.text in PERSONS:
            return True
    return False


def _effects(lemma, obj, place, preposition, name) -> list:
    """What an order asks of its object, when a place is said: that it be
    there -- or, where VerbNet says the verb takes it *from the doer* to
    whoever is named (`give`, `hand`, `pass`: give-13.1's Theme leaves the
    Agent), that they have it."""
    from research.v689 import change

    if obj is None or not name.get(obj.index):
        return []
    what = name[obj.index]
    where = name.get(place.index, "") if place is not None else ""
    if not where or preposition not in PLACES | {"to", "toward",
                                                  "towards"}:
        return []
    try:
        found = change.effects(lemma, True, preposition)
    except Exception:                              # noqa: BLE001
        found = []
    handed = any(one.kind == "location" and one.position == "object"
                 and one.at == "subject" and one.after is False
                 for one in found)
    if handed and preposition == "to":
        return [f"with {what} {where}"]
    return [f"at {what} {where}"]
