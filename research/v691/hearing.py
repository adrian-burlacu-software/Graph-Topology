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
from research.v691 import numbers, quantities as Q

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
    #: how counts changed, by what a statement says was done: `i gave 2
    #: apples to mary` is `[("with apple you", -2), ("with apple mary", 2)]`
    changes: list = field(default_factory=list)
    #: a question about a count (`how many apples does mary have`), taken
    #: apart: {kind, holders, where, compare, than, who, yesno, total}
    count: dict | None = None
    #: the kind of thing last counted, for `she gave 2 to john`
    counted: str = ""
    #: counts something was done to that could not be read as a change --
    #: `3 balloons popped` -- as (kind, holder or ""): not to be trusted
    #: any more, because a count stated after it would be a guess
    unsure: list = field(default_factory=list)
    #: what the parse tags as a proper noun: `john` in `give john 4
    #: apples`, where no article or position says it is a name
    names: list = field(default_factory=list)


def hear(text: str, stated=None, it: str = "", kind: str = "",
         who: str = "") -> Heard:
    """Read one utterance. `stated(verb)` says whether a verb names a state
    (`verbs.stated`), for `make X V`; `it` is what `it` refers to; `kind`
    is what a bare number counts (`she gave 2 to john`) and `who` is who
    `he` or `she` is."""
    words = parse(text)
    out = Heard(counted=kind)
    if not words:
        return out
    name = {one.index: (it if one.text in ("it", "they") and it else
                        thing(words, one)) for one in words}
    out.names = [one.text for one in words if one.tag in ("NNP", "NNPS")
                 and one.text.isalpha()]
    counts = _counts(words, kind)
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
                if held.index in counts:
                    continue
                out.facts.append(f"with {name[held.index]} "
                                 f"{name[subject.index]}")
            continue
        _asked_for(words, verb, subject, name, negated, question,
                   auxiliaries, stated, out)
        if (not question and not negated and subject is not None
                and verb.tag in ("VBD", "VBZ", "VBN")
                and verb.lemma not in ("be", "have", "do")):
            out.done += _done(words, verb)
    if counts or (question and _asks_how_many(words)):
        _amounts(words, counts, name, question, out, who)
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


# -- counts ----------------------------------------------------------------
#
# `mary has 3 apples`, `i gave 2 apples to mary`, `give mary 2 apples`, `how
# many apples does mary have`. The same parse, read for how many: a noun with
# a number on it is a count of a kind, and what a verb does to a count is
# what VerbNet says it does to one of them (`change.effects`) -- so there is
# no list here of verbs that give, take, lose or find.

#: Who a pronoun is, as the scene keeps them: the person talking is `you`
#: to the one listening, and the one listening is `me`.
SPEAKER = frozenset({"i", "me", "my", "we", "us", "our", "mine"})
LISTENER = frozenset({"you", "your", "yours"})
THIRD = frozenset({"he", "she", "him", "her", "his", "hers", "they",
                   "them", "their"})
#: Two prepositions that together say where from: `out of the bowl`.
SOURCES = frozenset({"out of", "off of", "away from", "down from",
                     "out from"})
FROM = "from"
#: What makes a count a sum of several: `together`, `in total`.
TOTAL = frozenset({"together", "altogether", "total", "combined",
                   "overall", "both"})
#: Comparatives that ask by how much, and which way.
MORE = frozenset({"more", "greater", "bigger", "larger"})
FEWER = frozenset({"fewer", "less", "smaller"})
#: Where something is when a place is not said: whichever place the scene
#: knows has some of that kind (`2 birds flew away`).
SOMEWHERE = "?"


def _subtree(words, index: int) -> list:
    out, frontier = [index], [index]
    while frontier:
        at = frontier.pop()
        for one in words:
            if one.head == at and one.index != at and one.index not in out:
                out.append(one.index)
                frontier.append(one.index)
    return sorted(out)


def _kind(word) -> str:
    """The kind a counted noun names: the noun WordNet knows it as, not
    the tagger's lemma -- the transformer tags `pears` in `how many pears
    does sam have` as a plural name and leaves it as it is."""
    from research.v689.asker import _noun_lemma
    return _noun_lemma(word.text, word.lemma) or word.lemma or word.text


def _counts(words, kind: str = "") -> dict:
    """noun index -> (how many, of what) for every counted noun: `3 apples`,
    `a dozen eggs`, and a number standing where a noun would -- `she gave 2
    to john` -- which counts whatever was counted last."""
    out = {}
    for noun in words:
        if noun.tag.startswith("NN"):
            parts = [one for one in words if one.head == noun.index
                     and one.dep in ("nummod", "quantmod")
                     and one.index != noun.index]
            if not parts:
                continue
            indices = sorted({at for one in parts
                              for at in _subtree(words, one.index)})
            value = numbers.value(" ".join(words[at].text for at in indices))
            if value is not None:
                out[noun.index] = (value, _kind(noun))
        elif (noun.tag == "CD" and kind and noun.dep in (
                "dobj", "nsubj", "pobj", "attr", "conj")):
            value = numbers.value(" ".join(
                words[at].text for at in _subtree(words, noun.index)))
            if value is not None:
                out[noun.index] = (value, kind)
    return out


def _asks_how_many(words) -> bool:
    """`how many`, `how much`, or `who has more`."""
    for word in words:
        if word.lemma in ("many", "much") and any(
                one.lemma == "how" and one.head == word.index
                for one in words):
            return True
    return any(word.tag == "JJR" for word in words) and (
        words[0].tag in ("WP", "VBZ", "VBP", "VBD"))


def _holder(words, word, name: dict, who: str) -> str:
    if word is None:
        return ""
    if word.text in SPEAKER:
        return "you"
    if word.text in LISTENER:
        return "me"
    if word.text in THIRD:
        return who
    found = name.get(word.index, "")
    if found and word.tag in ("NN", "NNS"):
        # `the red box` and `the blue box` are two boxes: a holder of a
        # count is told apart by what is said of it.
        said = [one.text for one in words if one.head == word.index
                and one.dep == "amod" and one.tag == "JJ"
                and one.text.isalpha() and one.lemma not in ("many",
                                                             "much")]
        if said:
            found = "-".join(said + [found])
    return found


def _does_something(verb: str, has_object: bool, preposition: str) -> bool:
    """Whether VerbNet says the verb changes anything about what it is done
    to, in a sentence of this shape: `pop`, `burst`, `paint`. Seeing and
    counting change nothing, and leave a count as it was."""
    from research.v689 import change
    try:
        found = change.effects(verb, has_object, preposition)
    except Exception:                              # noqa: BLE001
        return False
    return bool(found)


def _consumes(verb: str) -> bool:
    """Whether doing it uses a thing up: VerbNet's `take_in` (eat-39.1,
    drink), where nothing is said to go anywhere and the thing is gone."""
    from research.v689 import change
    return any(predicate == "take_in"
               for frame, _ in change.frames().get(verb, ())
               for predicate, *_ in frame.semantics)


#: VerbNet's words for a thing ceasing to be what it was.
CEASING = frozenset({"destroyed", "degradation_material_integrity",
                     "disappear"})


def _ceases(verb: str, found) -> bool:
    """Whether doing it ends the thing: VerbNet says it is destroyed,
    falls apart or disappears, or that it was alive and is not."""
    from research.v689 import change
    if any(one.word == "alive" and one.before and one.after is False
           for one in found):
        return True
    return any(predicate in CEASING
               for frame, _ in change.frames().get(verb, ())
               for predicate, *_ in frame.semantics)


def _moved(verb: str, dative: bool, preposition: str,
           has_object: bool) -> dict:
    """party -> +1 or -1: who gains and who loses one of what the verb
    moves, as VerbNet says -- read, not listed. The parties are positions
    (`subject`, `place`, and `from`: wherever it was); the caller puts
    names to them."""
    from research.v689 import change
    try:
        found = change.effects(verb, has_object, preposition)
    except Exception:                              # noqa: BLE001
        found = []
    moved = "object" if has_object else "subject"
    out: dict = {}
    for one in found:
        if one.kind != "location" or one.position != moved:
            continue
        if moved == "subject" and one.at == "subject":
            continue
        party = {"subject": "subject", "place": "place",
                 "": "subject" if moved == "object" else "from"}.get(
            one.at, "")
        if not party:
            continue
        if one.before is True and one.after is False:
            out[party] = -1
        elif one.after is True and out.get(party) != -1:
            out[party] = 1
    if moved == "object" and out.get("place") == 1 and "subject" not in out:
        # `peter handed 3 coins to kate`: VerbNet's hand says where they
        # went and not where from. Of coins, from the one handing them --
        # the rule plans keep (`quantities.lifted`).
        out["subject"] = -1
    if moved == "object" and dative and "place" not in out and \
            out.get("subject") == -1:
        # `mary gave john three apples`: VerbNet's give without a
        # preposition says only that the apples left mary. The one in the
        # dative is who has them now.
        out["place"] = 1
    if not out and moved == "object" and _consumes(verb):
        out["subject"] = -1
    if not out and _ceases(verb, found):
        # `2 plates broke`, `3 fish died`: what stops being is not counted.
        out["from"] = -1
    if moved == "subject" and "from" not in out and change.moves(verb) \
            and "place" not in out:
        # `2 birds flew away`: a thing that moves itself leaves wherever
        # it was.
        out["from"] = -1
    return out


def _amounts(words, counts: dict, name: dict, question: bool, out: Heard,
             who: str) -> None:
    """What the utterance says about counts: amounts it states, how a
    counted thing moved, an order to move some, or a question of how many.
    The plain facts the rest of `hear` read off the same nouns -- `at eggs
    basket` -- are taken back, because they said less."""
    texts = {words[index].text for index in counts}
    for index in sorted(counts):
        out.counted = counts[index][1]
    out.facts = [fact for fact in out.facts
                 if not texts & set(fact.split()[1:])]
    out.wants = [fact for fact in out.wants
                 if not texts & set(fact.split()[1:])]
    if question:
        _how_many(words, counts, name, out, who)
        return
    for verb in words:
        if not verb.tag.startswith("VB"):
            continue
        subject = next(iter(children(words, verb.index, SUBJECTS)), None)
        if subject is not None and subject.dep == "expl":
            subject = next((one for one in children(words, verb.index,
                                                    {"attr"})
                            if one.index in counts), None)
        if children(words, verb.index, {"neg"}):
            continue
        objects = [one for one in children(words, verb.index, {"dobj"})
                   if one.index in counts]
        place, preposition = None, ""
        # `there are 9 birds in the tree`: the place may hang off what is
        # counted rather than off the verb, as `_stated` reads it too.
        preps = children(words, verb.index, {"prep", "dative"})
        if subject is not None and subject.index in counts:
            preps += children(words, subject.index, {"prep"})
        for prep in preps:
            target = next(iter(children(words, prep.index, {"pobj"})), None)
            said = prep.text
            if target is None:
                # `out of the bowl`, `off of the shelf`: the place hangs off
                # a second preposition, and the two say where from.
                inner = next(iter(children(words, prep.index, {"prep"})),
                             None)
                if inner is not None:
                    target = next(iter(children(words, inner.index,
                                                {"pobj"})), None)
                    said = FROM if f"{prep.text} {inner.text}" in \
                        SOURCES else inner.text
            if target is not None and said not in ("than", "of"):
                place, preposition = target, said
        # `her mom gave her 4 books`: the one given to may be a pronoun.
        dative = next((one for one in children(words, verb.index, {"dative"})
                       if one.tag.startswith("NN") or one.tag == "PRP"),
                      None)
        if dative is not None and place is None:
            place, preposition = dative, "to"
        holder = _holder(words, subject, name, who)
        where = _holder(words, place, name, who) if place is not None else ""
        if verb.lemma in ("have", "hold", "own", "keep") \
                and verb.tag != "VB":
            if subject is not None and any(
                    one.lemma in ("each", "every")
                    for one in children(words, subject.index, {"det"})):
                # `each box has 6 eggs` is not what one box has: it is a
                # rate, and nothing here multiplies yet.
                continue
            for held in objects:
                if holder:
                    count, kind = counts[held.index]
                    out.facts.append(Q.amount(f"with {kind} {holder}", count))
            continue
        if verb.lemma == "be":
            if subject is not None and subject.index in counts and where \
                    and preposition in PLACES:
                count, kind = counts[subject.index]
                out.facts.append(Q.amount(f"at {kind} {where}", count))
            continue
        order = ((subject is None and verb.dep == "ROOT"
                  and verb.tag == "VB")
                 or (subject is not None and subject.text == "you"
                     and any(one.text in REQUESTS for one in children(
                         words, verb.index, {"aux"}))))
        if objects:
            count, kind = counts[objects[0].index]
            moved = _moved(verb.lemma, dative is not None, preposition, True)
        elif subject is not None and subject.index in counts:
            count, kind = counts[subject.index]
            moved = _moved(verb.lemma, False, preposition, False)
            holder = ""
        else:
            continue
        if not moved:
            if not order and _does_something(verb.lemma, bool(objects),
                                             preposition):
                # Something was done to them that is not a count moving:
                # the count is not what it was known to be.
                out.unsure.append((kind, holder))
            continue
        doer = "me" if order else holder
        parties = {"subject": doer, "place": where, "from": SOMEWHERE}
        changes = []
        for party, sign in sorted(moved.items()):
            named = parties.get(party, "")
            if not named:
                continue
            # A place said with `in`, `on`, `into` holds things; a person
            # said with `to` has them; wherever it was, was a place.
            relation = ("at" if party == "from" or (
                party == "place" and preposition not in ("to", ""))
                else "with")
            changes.append((f"{relation} {kind} {named}", sign * count))
        if order:
            gains = [(fluent, delta) for fluent, delta in changes
                     if delta > 0 and fluent.split()[-1] != doer]
            for fluent, delta in gains or changes:
                out.wants.append(Q.condition(
                    fluent, "+=" if delta > 0 else "-=", abs(delta)))
            if out.wants:
                out.order = True
                if verb.lemma not in out.verbs:
                    out.verbs.append(verb.lemma)
        else:
            out.changes += changes


def _how_many(words, counts, name, out: Heard, who: str) -> None:
    """`how many apples does mary have`, `how many are in the basket`, `how
    many more does john have than mary`, `who has more apples`."""
    kind, noun = "", None
    for word in words:
        if word.tag.startswith("NN") and any(
                one.lemma in ("many", "much") and one.head == word.index
                for one in words):
            kind, noun = _kind(word), word
            break
    compare = ""
    for word in words:
        if word.text in MORE:
            compare = "more"
        elif word.text in FEWER:
            compare = "fewer"
        else:
            continue
        if noun is None and words[word.head].tag.startswith("NN"):
            noun = words[word.head]
            kind = _kind(noun)
    if noun is None and out.counted:
        # `how many are left`: whatever was counted last.
        kind = out.counted
    if not kind:
        return
    root = next((one for one in words if one.dep == "ROOT"), None)
    if root is None:
        return
    subject = next(iter(children(words, root.index, SUBJECTS)), None)
    holders, than, where = [], [], []
    if subject is not None and subject.lemma in ("many", "much"):
        subject = None
    if subject is not None and (noun is None or subject.index != noun.index):
        listed = [subject]
        for two in words:
            # `ann, bob and cal`: each joined to the one before it.
            if two.dep == "conj" and two.head in {one.index
                                                  for one in listed}:
                listed.append(two)
        for one in listed:
            found = _holder(words, one, name, who)
            if found:
                holders.append(found)
    for prep in children(words, root.index, {"prep"}):
        if prep.text in PLACES:
            for target in children(words, prep.index, {"pobj"}):
                found = _holder(words, target, name, who)
                if found:
                    where.append(found)
    for at, word in enumerate(words):
        if word.text == "than" and at + 1 < len(words):
            found = _holder(words, words[at + 1], name, who)
            if found:
                than.append(found)
    asked_who = subject is not None and subject.tag == "WP"
    if asked_who:
        holders = [_holder(words, one, name, who) for one in words
                   if (noun is None or one.index > noun.index)
                   and one.tag in ("NNP", "NN", "PRP")
                   and one.index != getattr(noun, "index", -1)]
        holders = [one for one in holders if one]
    yesno = words[0].lemma in ("do", "be", "have") and not asked_who
    total = any(one.text in TOTAL for one in words) or (
        any(one.text == "all" and one.head == root.index for one in words))
    out.count = {"kind": kind, "holders": holders, "where": where,
                 "compare": compare, "than": than, "who": asked_who,
                 "yesno": yesno, "total": total or len(holders) > 1}
