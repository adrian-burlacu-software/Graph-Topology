"""One grammar: what a question asks, of which relation.

A wh-question about this conversation's individuals is read into the goal
cell it fills -- `(asked, relation)` -- and the session answers the cell
(`Session._cells`), whichever words filled it. The question word says what is
asked, and what follows it says of which relation:

    when did the dog bark                 AUX NP VP              time      occurrence
    how many times / how often did it     AUX NP VP              times     occurrence
    who chased the cat, who did not bark  [AUX] [not] VP         subject   occurrence
    who did Fred give the ball to         AUX NP ... to|from     recipient occurrence
    what did the dog chase                AUX NP VP              object    occurrence
    what was the dog doing                COPULA NP doing        verb      occurrence
    where is the key (before the garden)  COPULA NP [before X]   place     located
    what is the cat on                    COPULA NP PLACE        place     located
    what is Mary carrying                 COPULA NP V-ing        object    holding
    how many objects is Mary carrying     N COPULA NP V-ing      count     holding
    what is north of the office           COPULA RELATION NP     subject   dimension
    what is the kitchen north of          COPULA NP RELATION     object    dimension
    how do you go from the hall to it     AUX ... from NP to NP  path      dimension
    what color is Greg                    N COPULA NP            value     attribute
    what is Gertrude afraid of            COPULA NP QUALITY PREP object    attribute
    where will Sumit go                   will NP go             place     motive
    how many dogs are there               N are there            count     is_a
    which dog is black                    [N] [AUX] [not] VP     which     is_a
    what kind of dog is it                kind of N is NP        kind      is_a
    what happened (to the vase), what did it do (first)          events    story
    what did i tell you (first)                                  events    told
    what will happen tomorrow                                    events    future
    what do you know about it, what can it do, tell me about it  facts     any
    how do you know that, how sure are you                       grounds   answer
    what about a cat, and a fish                                 again     question

Each shape is tried in turn after the question word, first match reading.
The reading given for each cell is the one its old pattern in `reading.py`
gave -- the handlers read `aux`, `rest` and `obj` as they did -- and the act
it was named for is kept as the reading's name, which the page shows and
nothing dispatches on (`ACTS`).

A question about a kind -- `what can a dog do`, `how many legs does a spider
have` -- fills no cell here, and is v688's.
"""
from __future__ import annotations

from .reading import (AUX, COPULA, PLACES, QUESTION_WORDS, SEQUENCE, Mention,
                      Reading, _with_object, read_mention)
from .relations import phrase as relation_phrase

#: What a count of individuals ends with: `how many dogs are there`.
COUNT_ENDS = (("are", "there"), ("is", "there"), ("there", "are"),
              ("have", "come", "up"), ("has", "come", "up"))

#: Words after `how many` that count kinds of a thing, which R25 answers.
KIND_WORDS = frozenset({"kind", "kinds", "type", "types", "sort", "sorts",
                        "breed", "breeds", "species"})

#: What a question about a quality toward something ends with: `what is
#: Gertrude afraid of`.
TOWARD = frozenset({"of", "to", "about", "with", "for", "at", "by"})

#: Asking what was said, in the order it was said: telling time, not story
#: time. `what did i tell you` asked whether you told yourself things.
TOLD = (("what", "did", "i", "tell", "you"), ("what", "have", "i", "told",
                                              "you"),
        ("what", "did", "i", "say"), ("what", "have", "i", "said"))

#: Asking how an answer was reached. `how do you know that` was refused as a
#: question about method.
GROUNDS = (("how", "do", "you", "know"), ("how", "do", "you", "know", "that"),
           ("how", "sure", "are", "you"), ("are", "you", "sure"),
           ("why", "do", "you", "think", "so"))

#: The last question, of another kind: `what about a cat`, `and a fish?`.
AGAIN = (("what", "about"), ("how", "about"), ("and",))

#: The act each cell was a pattern for, kept as the reading's name.
ACTS = {("time", "occurrence"): "when",
        ("times", "occurrence"): "how_many_times",
        ("subject", "occurrence"): "who",
        ("recipient", "occurrence"): "to_whom",
        ("object", "occurrence"): "what_did",
        ("verb", "occurrence"): "doing",
        ("place", "located"): "where",
        ("object", "holding"): "carrying",
        ("count", "holding"): "carrying",
        ("subject", "dimension"): "related",
        ("object", "dimension"): "related",
        ("path", "dimension"): "route",
        ("value", "attribute"): "attribute",
        ("object", "attribute"): "toward",
        ("place", "motive"): "where_going",
        ("count", "is_a"): "how_many",
        ("which", "is_a"): "which",
        ("kind", "is_a"): "what",
        ("events", "story"): "happened",
        ("events", "told"): "happened",
        ("events", "future"): "happened",
        ("facts", "any"): "about",
        ("grounds", "answer"): "meta",
        ("again", "question"): "ellipsis"}


def _individual(found) -> bool:
    """A phrase naming one of this conversation's individuals, rather than a
    kind or a new one."""
    return found is not None and found.form not in ("indefinite", "another",
                                                    "kind", "plural", "group")


class _Question:
    """The words of one question, and the reading of a cell from them."""

    def __init__(self, tokens: list[str], lexicon, names: frozenset,
                 said: str) -> None:
        self.tokens, self.lexicon, self.names, self.said = (tokens, lexicon,
                                                            names, said)

    def reading(self, cell: tuple, *args, **kwargs) -> Reading:
        found = Reading(ACTS[cell], *args, said=self.said, **kwargs)
        found.cell = cell
        return found

    def with_object(self, reading: Reading) -> Reading:
        return _with_object(reading, self.lexicon, self.names)

    def mention(self, at: int, opener: str = "is", upto: int | None = None,
                final_ok: bool = True):
        """The phrase at `at`, read over the words up to `upto`."""
        span = self.tokens if upto is None else self.tokens[:upto]
        return read_mention(span, at, self.lexicon, opener,
                            final_ok=final_ok, names=self.names)


# -- occurrence ------------------------------------------------------------------
def _time(q: _Question) -> Reading | None:
    """`when did the dog bark`, `how many times did it bark`, `how often`."""
    tokens = q.tokens
    if tokens[0] == "when":
        at, slot = 1, "time"
    elif tokens[:3] == ["how", "many", "times"]:
        at, slot = 3, "times"
    elif tokens[:2] == ["how", "often"]:
        at, slot = 2, "times"
    else:
        return None
    if len(tokens) < at + 3 or tokens[at] not in AUX:
        return None
    head = tokens[at]
    found = q.mention(at + 1, head, final_ok=False)
    if _individual(found) and found.end < len(tokens):
        return q.with_object(q.reading((slot, "occurrence"), found, head,
                                       tokens[found.end:]))
    return None


def _subject(q: _Question) -> Reading | None:
    """`who chased the cat`, `who did not bark`; `who did Fred give it to`
    asks the recipient."""
    tokens = q.tokens
    head = tokens[1]
    if len(tokens) >= 5 and head in AUX and tokens[-1] in ("to", "from"):
        found = q.mention(2, head, final_ok=False)
        if _individual(found) and found.end < len(tokens) - 1:
            asked = q.with_object(q.reading(("recipient", "occurrence"), found,
                                            head, tokens[found.end:-1]))
            asked.rest = asked.rest + [tokens[-1]]
            return asked
    if head in COPULA:
        return None
    aux = head if head in AUX else None
    rest = tokens[2:] if aux else tokens[1:]
    holds = rest[:1] != ["not"]
    rest = rest if holds else rest[1:]
    if not rest:
        return None
    return q.with_object(q.reading(("subject", "occurrence"), None, aux, rest,
                                   holds=holds))


def _did(q: _Question) -> Reading | None:
    """`what did the dog chase`: its object. What it did (first) is the
    story's events, and what it can do or has is what was told of it."""
    tokens = q.tokens
    head = tokens[1]
    if head not in AUX or head in COPULA:
        return None
    start, holds = (3, False) if tokens[2:3] == ["not"] else (2, True)
    found = q.mention(start, head, final_ok=False)
    # `what do you need to bake a cake`: `you` is anyone, not me.
    if not _individual(found) or found.form in ("speaker", "addressee"):
        return None
    rest = tokens[found.end:]
    if rest[:1] == ["do"] and rest[1:2] and rest[1] in SEQUENCE:
        return q.reading(("events", "story"), found, head, rest[1:])
    # `what did the dog do`: what it did, not what it can do.
    if rest == ["do"] and head == "did":
        return q.reading(("events", "story"), found, head, [])
    if rest in (["do"], ["have"]):
        return q.reading(("facts", "any"), found, head, rest, holds=holds)
    if not rest:
        return None
    return q.reading(("object", "occurrence"), found, head, rest, holds=holds)


def _doing(q: _Question) -> Reading | None:
    """`what was the dog doing`, `what is it doing`."""
    tokens = q.tokens
    if len(tokens) < 4 or tokens[1] not in COPULA or tokens[-1] != "doing":
        return None
    found = q.mention(2, tokens[1], upto=len(tokens) - 1)
    if _individual(found) and found.end == len(tokens) - 1:
        return q.reading(("verb", "occurrence"), found, tokens[1], ["doing"])
    return None


# -- located, and holding ----------------------------------------------------------
def _where(q: _Question) -> Reading | None:
    """`where is the key`, `where was the football before the bathroom`
    (`story.where_around`)."""
    tokens = q.tokens
    head = tokens[1]
    if head not in COPULA:
        return None
    found = q.mention(2)
    if _individual(found) and found.end == len(tokens):
        return q.reading(("place", "located"), found)
    if (_individual(found) and len(tokens) > found.end + 1
            and tokens[found.end] in ("before", "after")):
        return q.reading(("place", "located"), found, head,
                         tokens[found.end:])
    return None


def _on(q: _Question) -> Reading | None:
    """`what is the cat on`."""
    tokens = q.tokens
    if tokens[1] not in COPULA or tokens[-1] not in PLACES:
        return None
    found = q.mention(2, upto=len(tokens) - 1)
    if _individual(found) and found.end == len(tokens) - 1:
        return q.reading(("place", "located"), found)
    return None


def _carrying(q: _Question) -> Reading | None:
    """`what is Mary carrying`, `how many objects is Mary carrying` -- if the
    verb means having it with you, which the handler asks VerbNet."""
    tokens = q.tokens
    if tokens[0] == "what":
        at = 1
    elif tokens[:2] == ["how", "many"] and tokens[2:3] != ["times"]:
        at = 3
    else:
        return None
    if (len(tokens) < at + 3 or tokens[at] not in COPULA
            or not tokens[-1].endswith("ing")
            or not hasattr(q.lexicon, "progressive")):
        return None
    head = tokens[at]
    found = q.mention(at + 1, head, upto=len(tokens) - 1)
    verb = q.lexicon.progressive(tokens[-1])
    if _individual(found) and found.end == len(tokens) - 1 and verb:
        counting = at == 3
        return q.reading(("count" if counting else "object", "holding"),
                         found, head, [verb], count=counting,
                         obj=(Mention("kind", tokens[2], text=tokens[2])
                              if counting else None))
    return None


# -- dimension: S1-S4's relations ------------------------------------------------
def _related(q: _Question) -> Reading | None:
    """`what is north of the office`, `what is the kitchen north of`: one side
    of a relation asked, the other named (`relations.py`). `?` in `rest`
    marks the side asked."""
    tokens = q.tokens
    if len(tokens) < 4 or tokens[1] not in COPULA:
        return None
    opened = relation_phrase(tokens[2:])
    if opened is not None:
        found = q.mention(2 + opened.length)
        if _individual(found) and found.end == len(tokens):
            return q.reading(("subject", "dimension"), found, tokens[1],
                             ["?"] + tokens[2:2 + opened.length])
    found = q.mention(2)
    closed = relation_phrase(tokens[found.end:]) if found else None
    if (_individual(found) and closed is not None
            and found.end + closed.length == len(tokens)):
        return q.reading(("object", "dimension"), found, tokens[1],
                         tokens[found.end:] + ["?"])
    return None


def _route(q: _Question) -> Reading | None:
    """`how do you go from the kitchen to the garden`: a way over the
    compass."""
    tokens = q.tokens
    if (tokens[1] not in AUX or "from" not in tokens
            or "to" not in tokens[tokens.index("from"):]):
        return None
    start = tokens.index("from")
    there = q.mention(start + 1)
    if (_individual(there) and there.end < len(tokens)
            and tokens[there.end] == "to"):
        goal = q.mention(there.end + 1)
        if _individual(goal) and goal.end == len(tokens):
            asked = q.reading(("path", "dimension"), there, tokens[1],
                              tokens[3:start])
            asked.obj = goal
            return asked
    return None


# -- attribute, and motive -------------------------------------------------------
def _attribute(q: _Question) -> Reading | None:
    """`what color is Greg`: one of its attributes, the kind of value
    asked."""
    tokens = q.tokens
    if (len(tokens) < 4 or tokens[2] not in COPULA or tokens[1] in AUX
            or tokens[1] in QUESTION_WORDS):
        return None
    found = q.mention(3)
    if _individual(found) and found.end == len(tokens):
        return q.reading(("value", "attribute"), found, tokens[2],
                         [tokens[1]])
    return None


def _toward(q: _Question) -> Reading | None:
    """`what is Gertrude afraid of`: what a quality toward something is
    toward."""
    tokens = q.tokens
    if len(tokens) < 5 or tokens[1] not in COPULA or tokens[-1] not in TOWARD:
        return None
    found = q.mention(2, upto=len(tokens) - 2)
    if _individual(found) and found.end == len(tokens) - 2:
        return q.reading(("object", "attribute"), found, tokens[1],
                         tokens[-2:])
    return None


def _where_going(q: _Question) -> Reading | None:
    """`where will Sumit go`: where someone is going, which nothing has
    told."""
    tokens = q.tokens
    if len(tokens) != 4 or tokens[1] != "will":
        return None
    found = q.mention(2, "will")
    if _individual(found) and found.end == 3:
        return q.reading(("place", "motive"), found, "will", tokens[3:])
    return None


# -- is_a: the individuals of a kind ---------------------------------------------
def _how_many(q: _Question) -> Reading | None:
    """`how many dogs are there`."""
    tokens = q.tokens
    if tokens[:2] != ["how", "many"]:
        return None
    end = next((len(one) for one in COUNT_ENDS
                if tuple(tokens[-len(one):]) == one), 0)
    middle = tokens[2:len(tokens) - end] if end else []
    if not end or (middle and middle[0] in KIND_WORDS):
        return None
    kind = (" ".join(middle[:-1] + [q.lexicon.lemma(middle[-1])])
            if middle else "")
    return q.reading(("count", "is_a"), Mention("kind", kind,
                                                text=" ".join(middle),
                                                end=2 + len(middle)))


def _which(q: _Question) -> Reading | None:
    """`which dog is black`, `which one can swim`: one identified by
    description."""
    tokens = q.tokens
    if len(tokens) < 3:
        return None
    at, kind = 1, ""
    if tokens[1] not in AUX:
        kind = "" if tokens[1] == "one" else q.lexicon.lemma(tokens[1])
        at = 2
    aux = tokens[at] if tokens[at] in AUX else None
    rest = tokens[at + 1:] if aux else tokens[at:]
    holds = rest[:1] != ["not"]
    rest = rest if holds else rest[1:]
    if not rest:
        return None
    return q.with_object(q.reading(
        ("which", "is_a"), Mention("kind", kind, text=" ".join(tokens[1:at]),
                                   end=at), aux, rest, holds=holds))


def _kind_of(q: _Question) -> Reading | None:
    """`what kind of dog is it`."""
    tokens = q.tokens
    if tokens[:3] != ["what", "kind", "of"] or "is" not in tokens[3:]:
        return None
    found = q.mention(tokens.index("is", 3) + 1)
    if _individual(found) and found.end == len(tokens):
        return q.reading(("kind", "is_a"), found)
    return None


# -- events: the story's order, telling's, and what is to come -------------------
def _happened(q: _Question) -> Reading | None:
    """`what happened`, `what happened first`, `what happened to the vase`."""
    tokens = q.tokens
    if tokens[:2] != ["what", "happened"] and tokens[:3] != ["what", "has",
                                                             "happened"]:
        return None
    at = 2 if tokens[1] == "happened" else 3
    # `what happened to the vase`: what it took part in.
    if tokens[at:at + 1] == ["to"]:
        found = q.mention(at + 1)
        if _individual(found) and found.end == len(tokens):
            return q.reading(("events", "story"), found, None, ["to"])
    return q.reading(("events", "story"),
                     rest=[word for word in tokens[at:] if word in SEQUENCE])


def _future(q: _Question) -> Reading | None:
    """`what will happen tomorrow`, `what is going to happen`."""
    tokens = q.tokens
    if (tokens[:3] not in (["what", "will", "happen"], ["what", "is", "going"])
            or (tokens[1] != "will" and tokens[3:5] != ["to", "happen"])):
        return None
    at = 3 if tokens[1] == "will" else 5
    return q.reading(("events", "future"), rest=["future"] + [
        word for word in tokens[at:] if word in SEQUENCE])


def _told(q: _Question) -> Reading | None:
    """`what did i tell you`, `what did i say first`."""
    tokens = q.tokens
    for told in TOLD:
        after = tokens[len(told):]
        if tuple(tokens[:len(told)]) == told and (not after or (
                len(after) == 1 and after[0] in SEQUENCE)):
            return q.reading(("events", "told"), rest=["told"] + after)
    return None


# -- about the conversation --------------------------------------------------------
def _about(q: _Question) -> Reading | None:
    """`what do you know about it`, `tell me about the dog`."""
    tokens = q.tokens
    at = (5 if tokens[:5] == ["what", "do", "you", "know", "about"] else
          3 if tokens[:3] == ["tell", "me", "about"] else None)
    if at is None:
        return None
    found = q.mention(at)
    if _individual(found) and found.end == len(tokens):
        return q.reading(("facts", "any"), found)
    return None


def _grounds(q: _Question) -> Reading | None:
    """`how do you know that`, `how sure are you`: the last answer's
    grounds."""
    return (q.reading(("grounds", "answer")) if tuple(q.tokens) in GROUNDS
            else None)


def _again(q: _Question) -> Reading | None:
    """`what about a cat`, `and a fish?`: the last question, of another
    kind."""
    tokens = q.tokens
    for opener in AGAIN:
        if tuple(tokens[:len(opener)]) == opener and len(tokens) > len(opener):
            found = q.mention(len(opener))
            if (found is not None and found.end == len(tokens)
                    and found.form in ("indefinite", "kind")):
                return q.reading(("again", "question"), found)
    return None


#: The shapes tried after each question word, in order.
SHAPES = {
    "what": (_doing, _related, _toward, _attribute, _carrying, _on, _did,
             _future, _happened, _about, _told, _again, _kind_of),
    "how": (_route, _time, _carrying, _how_many, _grounds, _again),
    "when": (_time,),
    "where": (_where_going, _where),
    "who": (_subject,),
    "whom": (_subject,),
    "which": (_which,),
    "tell": (_about,),
    "and": (_again,),
    "are": (_grounds,),
    "why": (_grounds,),
}


def cell_reading(tokens: list[str], lexicon, names: frozenset,
                 said: str) -> Reading | None:
    """The reading of the cell a question fills, or None."""
    if len(tokens) < 2:
        return None
    question = _Question(tokens, lexicon, names, said)
    for shape in SHAPES.get(tokens[0], ()):
        found = shape(question)
        if found is not None:
            return found
    return None
