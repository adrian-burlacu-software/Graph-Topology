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

Every shape is in one table (`SHAPES`), tried in turn after the question
word, and each proposes a goal. A question is proposed at most two: the first
shape to read it straight into slots, and the first to read it as a reading
of its own.
The reading given for each cell is the one its old pattern in `reading.py`
gave -- the handlers read `aux`, `rest` and `obj` as they did -- with the act
`question`, and one goal: the cell, with its slots read off the reading
(`_own_goal`).

Some shapes read a question straight into slots, the relation and what is
asked of it read off the words:

    who is in the kitchen, is anyone in it, how many people are in it
    when was Mary in the kitchen
    who has the football, is anyone carrying it, does Mary have it
    what does Mary have, how many things does she have
    did anyone go to the garden, how many people went to the kitchen
    where did Mary go (first), where did she drop the football

That goal is tried first, and composed from memory by its relation's
operator (`goals.py`); when it finds nothing, the goal the question's own
words state is tried, or the act the rest of `reading.py` read. So `what does
Mary have` is what is with her, and failing that what was told she has.

A question about a kind -- `what can a dog do`, `how many legs does a spider
have` -- fills no cell here, and is v688's.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps

from .reading import (AUX, COPULA, PLACES, QUESTION_WORDS, SEQUENCE, Mention,
                      Reading, _with_object, read_mention, tags_of, tokens_of)
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

#: The cells this grammar's shapes read, into slots or as readings of their
#: own. Each is answered by one method of its relation's operator
#: (`goals.Answering.cells`).
CELLS = frozenset({
    ("subject", "located"), ("any", "located"), ("count", "located"),
    ("time", "located"), ("any", "holding"), ("whether", "holding"),
    ("subject", "holding"), ("any", "occurrence"), ("count", "occurrence"),
    ("place", "occurrence"),
    ("time", "occurrence"), ("times", "occurrence"), ("subject", "occurrence"),
    ("recipient", "occurrence"), ("object", "occurrence"),
    ("verb", "occurrence"), ("place", "located"), ("object", "holding"),
    ("count", "holding"), ("subject", "dimension"), ("object", "dimension"),
    ("path", "dimension"), ("value", "attribute"), ("object", "attribute"),
    ("place", "motive"), ("count", "is_a"), ("which", "is_a"),
    ("kind", "is_a"), ("events", "story"), ("events", "told"),
    ("events", "future"), ("facts", "any"), ("grounds", "answer"),
    ("again", "question")})


# -- goals ------------------------------------------------------------------------
#: Being at a place: `in the kitchen`, `at school`.
PLACING = frozenset({"in", "inside", "at", "on"})

#: Anyone at all, and anything at all.
PEOPLE = frozenset({"anyone", "anybody", "someone", "somebody"})
THINGS = frozenset({"anything", "something"})

#: Counting people: `how many people`, `how many persons`.
PERSONS = frozenset({"people", "person", "persons"})

#: A verb of having said bare, before its object: `who has the football`.
HAS = frozenset({"has", "have", "had", "holds", "owns", "keeps"})

#: What `does Mary ___ it` asks of her when it is having something.
HAVE = frozenset({"have", "hold", "own", "keep", "carry"})

#: Auxiliaries a question about an occurrence or a having opens with.
DID = frozenset({"did", "does", "do", "has", "have", "had"})

#: Mention forms that name no one here.
NO_ONE = frozenset({"indefinite", "another", "kind", "plural", "group"})


@dataclass
class Goal:
    asked: str
    relation: str
    who: str = ""
    subject: Mention | None = None
    object: Mention | None = None
    verb: str = ""
    sequence: int | None = None
    clause: Reading | None = None
    said: str = ""
    #: stated by the question's own reading, which is its clause and the
    #: reading `reading.read` gives the question
    own: bool = False
    #: the exemplar a question nothing else read was read like, and how near
    #: (`exemplars.py`)
    like: str = ""

    @property
    def cell(self) -> tuple:
        """What is asked, of which relation: what the session answers by."""
        return (self.asked, self.relation)

    def as_dict(self) -> dict:
        return {"asked": self.asked, "relation": self.relation,
                "who": self.who,
                "subject": self.subject.text if self.subject else None,
                "object": self.object.text if self.object else None,
                "verb": self.verb, "sequence": self.sequence,
                "clause": " ".join(self.clause.rest) if self.clause else None,
                "like": self.like}


def _individual(found) -> bool:
    """A phrase naming one of this conversation's individuals, rather than a
    kind or a new one."""
    return found is not None and found.form not in NO_ONE


class _Question:
    """The words of one question, and the reading of a cell from them."""

    def __init__(self, tokens: list[str], lexicon, names: frozenset,
                 said: str) -> None:
        self.tokens, self.lexicon, self.names, self.said = (tokens, lexicon,
                                                            names, said)
        #: the cell of the last reading made
        self.cell: tuple | None = None

    def reading(self, cell: tuple, *args, **kwargs) -> Reading:
        self.cell = cell
        return Reading("question", *args, said=self.said, **kwargs)

    def with_object(self, reading: Reading) -> Reading:
        return _with_object(reading, self.lexicon, self.names)

    def mention(self, at: int, opener: str = "is", upto: int | None = None,
                final_ok: bool = True):
        """The phrase at `at`, read over the words up to `upto`."""
        span = self.tokens if upto is None else self.tokens[:upto]
        return read_mention(span, at, self.lexicon, opener,
                            final_ok=final_ok, names=self.names)

    def named(self, at: int, upto: int | None = None, opener: str = "is"):
        """The individual named from `at` to the end (or to `upto`)."""
        found = self.mention(at, opener, upto)
        end = len(self.tokens) if upto is None else upto
        return found if _individual(found) and found.end == end else None

    @staticmethod
    def filling(word: str) -> str:
        """Who may fill the slot a word asks: people, or things."""
        return ("people" if word in ("who", "whom") or word in PEOPLE
                else "things")

    def kind_of(self, words: list[str]) -> str:
        who = " ".join(words[:-1] + [self.lexicon.lemma(words[-1])])
        return "people" if who in PERSONS else who

    def clause(self, aux, rest, subject=None) -> Reading:
        return self.with_object(Reading("question", subject, aux, list(rest),
                                        said=self.said))

    def progressive(self, word: str) -> str:
        if not word.endswith("ing") or not hasattr(self.lexicon,
                                                   "progressive"):
            return ""
        return self.lexicon.progressive(word) or ""


def _stated(shape):
    """A shape that reads a question as a reading of its own: the question's
    reading, stating one goal (`_own_goal`)."""
    @wraps(shape)
    def propose(q: _Question) -> Goal | None:
        found = shape(q)
        if found is None:
            return None
        goal = _own_goal(q.cell, found, q.lexicon)
        found.goals = [goal]
        return goal
    propose.family = "stated"
    return propose


def _slots(shape):
    """A shape that reads a question straight into slots: a goal its
    relation's operator composes from memory."""
    @wraps(shape)
    def propose(q: _Question) -> Goal | None:
        return shape(q) if len(q.tokens) >= 3 else None
    propose.family = "slots"
    return propose


# -- occurrence ------------------------------------------------------------------
@_stated
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


@_stated
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


@_stated
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


@_stated
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
@_stated
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


@_stated
def _on(q: _Question) -> Reading | None:
    """`what is the cat on`."""
    tokens = q.tokens
    if tokens[1] not in COPULA or tokens[-1] not in PLACES:
        return None
    found = q.mention(2, upto=len(tokens) - 1)
    if _individual(found) and found.end == len(tokens) - 1:
        return q.reading(("place", "located"), found)
    return None


@_stated
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
@_stated
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


@_stated
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
@_stated
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


@_stated
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


@_stated
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
@_stated
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


@_stated
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


@_stated
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
@_stated
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


@_stated
def _future(q: _Question) -> Reading | None:
    """`what will happen tomorrow`, `what is going to happen`."""
    tokens = q.tokens
    if (tokens[:3] not in (["what", "will", "happen"], ["what", "is", "going"])
            or (tokens[1] != "will" and tokens[3:5] != ["to", "happen"])):
        return None
    at = 3 if tokens[1] == "will" else 5
    return q.reading(("events", "future"), rest=["future"] + [
        word for word in tokens[at:] if word in SEQUENCE])


@_stated
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
@_stated
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


@_stated
def _grounds(q: _Question) -> Reading | None:
    """`how do you know that`, `how sure are you`: the last answer's
    grounds."""
    return (q.reading(("grounds", "answer")) if tuple(q.tokens) in GROUNDS
            else None)


@_stated
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


# -- read straight into slots ------------------------------------------------------
@_slots
def _in(q: _Question) -> Goal | None:
    """`who is in the kitchen`, `what is in the box`."""
    tokens = q.tokens
    if tokens[1] in COPULA and tokens[2] in PLACING:
        place = q.named(3)
        if place is not None:
            return Goal("subject", "located", q.filling(tokens[0]),
                        object=place, said=q.said)
    return None


@_slots
def _when_in(q: _Question) -> Goal | None:
    """`when was Mary in the kitchen`."""
    tokens = q.tokens
    if tokens[1] not in COPULA:
        return None
    found = q.mention(2, tokens[1], final_ok=False)
    if (_individual(found) and found.end + 1 < len(tokens)
            and tokens[found.end] in PLACING):
        place = q.named(found.end + 1)
        if place is not None:
            return Goal("time", "located", subject=found, object=place,
                        said=q.said)
    return None


@_slots
def _anyone(q: _Question) -> Goal | None:
    """`is anyone in the kitchen`, `is anyone carrying the football`."""
    tokens = q.tokens
    if tokens[1] not in PEOPLE | THINGS or len(tokens) <= 3:
        return None
    if tokens[2] in PLACING:
        place = q.named(3)
        if place is not None:
            return Goal("any", "located", q.filling(tokens[1]), object=place,
                        said=q.said)
    verb = q.progressive(tokens[2])
    if verb:
        thing = q.named(3)
        if thing is not None:
            return Goal("any", "holding", q.filling(tokens[1]), object=thing,
                        verb=verb, said=q.said)
    return None


@_slots
def _whether_carrying(q: _Question) -> Goal | None:
    """`is Mary carrying the football`."""
    tokens = q.tokens
    if len(tokens) <= 3:
        return None
    holder = q.mention(1, tokens[0], final_ok=False)
    if _individual(holder) and holder.end + 1 < len(tokens):
        verb = q.progressive(tokens[holder.end])
        thing = q.named(holder.end + 1) if verb else None
        if thing is not None:
            return Goal("whether", "holding", subject=holder, object=thing,
                        verb=verb, said=q.said)
    return None


@_slots
def _did_anyone(q: _Question) -> Goal | None:
    """`does anyone have the football`, `did anyone go to the garden`, `does
    Mary have the football`."""
    tokens = q.tokens
    if len(tokens) <= 3:
        return None
    if tokens[1] in PEOPLE and tokens[2] in HAVE:
        thing = q.named(3)
        if thing is not None:
            return Goal("any", "holding", "people", object=thing,
                        verb=tokens[2], said=q.said)
    if tokens[1] in PEOPLE | THINGS:
        return Goal("any", "occurrence", q.filling(tokens[1]),
                    clause=q.clause(tokens[0], tokens[2:]), said=q.said)
    holder = q.mention(1, tokens[0], final_ok=False)
    if (_individual(holder) and holder.end + 1 < len(tokens)
            and tokens[holder.end] in HAVE):
        thing = q.named(holder.end + 1)
        if thing is not None:
            return Goal("whether", "holding", subject=holder, object=thing,
                        verb=tokens[holder.end], said=q.said)
    return None


@_slots
def _how_many_in(q: _Question) -> Goal | None:
    """`how many people are in the kitchen`, `how many things does Mary
    have`, `how many people went to the kitchen`."""
    tokens = q.tokens
    if tokens[:2] != ["how", "many"] or len(tokens) <= 3:
        return None
    at = next((index for index in range(3, len(tokens))
               if tokens[index] in COPULA), None)
    if at is not None and at + 2 < len(tokens) and tokens[at + 1] in PLACING:
        place = q.named(at + 2)
        if place is not None:
            return Goal("count", "located", q.kind_of(tokens[2:at]),
                        object=place, said=q.said)
    at = next((index for index in range(3, len(tokens))
               if tokens[index] in DID), None)
    if at is not None and tokens[-1] in HAVE and at + 1 < len(tokens) - 1:
        holder = q.named(at + 1, len(tokens) - 1, opener=tokens[at])
        if holder is not None:
            return Goal("count", "holding", q.kind_of(tokens[2:at]),
                        subject=holder, verb=tokens[-1], said=q.said)
    tags = tags_of(tokens, q.lexicon)
    if tokens[3] not in AUX and tags and tags[3].startswith("VB"):
        return Goal("count", "occurrence", q.kind_of(tokens[2:3]),
                    clause=q.clause(None, tokens[3:]), said=q.said)
    return None


@_slots
def _has(q: _Question) -> Goal | None:
    """`who has the football`, `who is carrying the football`."""
    tokens = q.tokens
    verb, at = "", 0
    if tokens[1] in HAS:
        verb, at = q.lexicon.lemma(tokens[1]), 2
    elif tokens[1] in COPULA and len(tokens) > 3:
        verb, at = q.progressive(tokens[2]), 3
    if verb:
        thing = q.named(at)
        if thing is not None:
            return Goal("subject", "holding", q.filling(tokens[0]),
                        object=thing, verb=verb, said=q.said)
    return None


@_slots
def _does_have(q: _Question) -> Goal | None:
    """`what does Mary have`."""
    tokens = q.tokens
    if tokens[1] in DID and tokens[-1] in HAVE:
        holder = q.named(2, len(tokens) - 1, opener=tokens[1])
        if holder is not None:
            return Goal("object", "holding", subject=holder,
                        verb=tokens[-1], said=q.said)
    return None


@_slots
def _where_did(q: _Question) -> Goal | None:
    """`where did Mary go`, `where did Mary go first`, `where did Mary drop
    the football`."""
    tokens = q.tokens
    if tokens[1] not in DID:
        return None
    found = q.mention(2, tokens[1], final_ok=False)
    if _individual(found) and found.end < len(tokens):
        rest = tokens[found.end:]
        sequence = None
        if len(rest) == 2 and rest[-1] in SEQUENCE:
            sequence, rest = SEQUENCE[rest[-1]], rest[:-1]
        clause = q.clause(tokens[1], rest, found)
        if len(rest) == 1 or clause.obj is not None:
            return Goal("place", "occurrence", subject=found,
                        verb=q.lexicon.lemma(rest[0]), sequence=sequence,
                        clause=clause, said=q.said)
    return None


# -- the table -----------------------------------------------------------------------
#: The shapes tried after each question word, in order: slots and readings
#: alike, each proposing a goal.
SHAPES = {
    **{word: (_anyone, _whether_carrying) for word in COPULA},
    **{word: (_did_anyone,) for word in DID},
    "are": (_anyone, _whether_carrying, _grounds),
    "what": (_in, _has, _does_have, _doing, _related, _toward, _attribute,
             _carrying, _on, _did, _future, _happened, _about, _told, _again,
             _kind_of),
    "who": (_in, _has, _subject),
    "whom": (_subject,),
    "when": (_when_in, _time),
    "where": (_where_did, _where_going, _where),
    "how": (_how_many_in, _route, _time, _carrying, _how_many, _grounds,
            _again),
    "which": (_which,),
    "tell": (_about,),
    "and": (_again,),
    "why": (_grounds,),
}


def propose(tokens: list[str], lexicon, names: frozenset = frozenset(),
            said: str = "") -> list[Goal]:
    """The goals a question states, in the order they are tried: the first
    shape to read it as slots, then the first to read it as a reading of its
    own, which is the question's reading (`reading.read`)."""
    if len(tokens) < 2:
        return []
    question = _Question(tokens, lexicon, names, said)
    found: dict = {}
    for shape in SHAPES.get(tokens[0], ()):
        if shape.family in found:
            continue
        goal = shape(question)
        if goal is not None:
            found[shape.family] = goal
    return [found[family] for family in ("slots", "stated")
            if family in found]


def read_goal(text: str, lexicon, names: frozenset = frozenset()
              ) -> Goal | None:
    """The goal a question states as slots, read from its text."""
    return next((one for one in propose(tokens_of(text)[0], lexicon, names,
                                        text) if not one.own), None)


def _own_goal(cell: tuple, reading: Reading, lexicon) -> Goal:
    """The goal a question's own words state: the cell its shape read, with
    the slots read off its reading, which is the goal's clause."""
    mention = reading.mention
    kind = mention is not None and mention.form == "kind"
    who = mention.kind if kind else ""
    if reading.count and reading.obj is not None:
        who = lexicon.lemma(reading.obj.kind)
        who = "people" if who in PERSONS else who
    verb = ""
    if cell[1] in ("occurrence", "holding"):
        verb = next((lexicon.lemma(word) for word in reading.rest
                     if word.isalpha()), "")
    sequence = next((SEQUENCE[word] for word in reading.rest
                     if word in SEQUENCE), None)
    return Goal(cell[0], cell[1], who, None if kind else mention, reading.obj,
                verb, sequence, reading, reading.said, own=True)
