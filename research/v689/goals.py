"""Goals: what a question asks, as slots, and the operators that answer it by
composing what memory already does.

`reading.py` reads a question by matching it against a pattern written for
it, and names the act; the session runs the operator for that act. So a
question nobody wrote a pattern for is not asked at all, even when every
piece of its answer is already here: v689 could say where Mary is, and who
was in the garden was a question about kinds; it could say what Mary is
carrying, and `who has the football` came back "nobody here was said to a
football" (`v690/DESIGN.md` §4.6, §4.7).

A goal is the question taken apart:

    asked      what the answer fills: the subject, the object, the place, a
               count, or whether there is any
    relation   located (being at a place), holding (having with one), or an
               occurrence of a verb
    who        what may fill the subject: people, things, or a kind
    subject    the one it is about, when that is named
    object     the thing held, or the place
    verb       the verb said, for VerbNet to read

and the operators here propose on those slots, not on a pattern's name:

    located            who is in the kitchen, is anyone in the kitchen, how
                       many people are in the kitchen, what is in the kitchen
                       -- where each one is as the story stands (T4, T3)
    holding            who has the football, who is carrying it, what does
                       Mary have -- what is with whom, where VerbNet reads the
                       verb as having it with you (T4)
    occurrence place   where did Mary go, where did Mary go first -- the
                       occurrences of the verb, in story order, and where
                       each went (T5, T1)

They run in front of the act operators and **decline when they find
nothing**, so what a pattern answered before it still answers -- `what does
it have` of a beagle told to have a tail is what was told -- unless nothing
else would read the question at all, when "not told" is the answer rather
than a question about kinds.
"""
from __future__ import annotations

from dataclasses import dataclass

from research.v687.executive import ANSWERED, DECLINED, Operator

from . import change as changes
from .discourse import SPEAKER_KIND
from .reading import COPULA, SEQUENCE, Mention, read_mention, tokens_of

#: Being at a place: `in the kitchen`, `at school`.
PLACING = frozenset({"in", "inside", "at", "on"})

#: Anyone at all, and anything at all.
PEOPLE = frozenset({"anyone", "anybody", "someone", "somebody"})
THINGS = frozenset({"anything", "something"})

#: Counting people: `how many people`, `how many persons`.
PERSONS = frozenset({"people", "person", "persons"})

#: A verb of having said bare, before its object: `who has the football`.
HAS = frozenset({"has", "have", "had", "holds", "owns", "keeps"})

#: What `what does Mary ___` asks of her when it is having something.
HAVE = frozenset({"have", "hold", "own", "keep", "carry"})

#: Auxiliaries a question about an occurrence opens with.
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
    said: str = ""

    def as_dict(self) -> dict:
        return {"asked": self.asked, "relation": self.relation,
                "who": self.who,
                "subject": self.subject.text if self.subject else None,
                "object": self.object.text if self.object else None,
                "verb": self.verb, "sequence": self.sequence}


def read_goal(text: str, lexicon, names: frozenset = frozenset()
              ) -> Goal | None:
    """The goal a question states, or None when it is not one of these."""
    tokens, _ = tokens_of(text)
    if len(tokens) < 3:
        return None
    first = tokens[0]

    def named(at: int, upto: int | None = None, opener: str = "is"):
        """The individual named from `at` to the end (or to `upto`)."""
        span = tokens if upto is None else tokens[:upto]
        found = read_mention(span, at, lexicon, opener, final_ok=True,
                             names=names)
        if found is None or found.end != len(span) or found.form in NO_ONE:
            return None
        return found

    def filling(word: str) -> str:
        return "people" if word in ("who", "whom") or word in PEOPLE \
            else "things"

    # `who is in the kitchen`, `what is in the box`
    if first in ("who", "what") and tokens[1] in COPULA \
            and tokens[2] in PLACING:
        place = named(3)
        if place is not None:
            return Goal("subject", "located", filling(first), object=place,
                        said=text)

    # `is anyone in the kitchen`
    if first in COPULA and tokens[1] in PEOPLE | THINGS \
            and len(tokens) > 3 and tokens[2] in PLACING:
        place = named(3)
        if place is not None:
            return Goal("any", "located", filling(tokens[1]), object=place,
                        said=text)

    # `how many people are in the kitchen`
    if tokens[:2] == ["how", "many"]:
        at = next((index for index in range(3, len(tokens))
                   if tokens[index] in COPULA), None)
        if at is not None and at + 2 < len(tokens) \
                and tokens[at + 1] in PLACING:
            place = named(at + 2)
            kind = tokens[2:at]
            if place is not None and kind:
                who = " ".join(kind[:-1] + [lexicon.lemma(kind[-1])])
                if who in PERSONS:
                    who = "people"
                return Goal("count", "located", who, object=place, said=text)

    # `who has the football`, `who is carrying the football`
    if first in ("who", "what"):
        verb, at = "", 0
        if tokens[1] in HAS:
            verb, at = lexicon.lemma(tokens[1]), 2
        elif (tokens[1] in COPULA and len(tokens) > 3
              and tokens[2].endswith("ing")
              and hasattr(lexicon, "progressive")):
            verb, at = lexicon.progressive(tokens[2]) or "", 3
        if verb:
            thing = named(at)
            if thing is not None:
                return Goal("subject", "holding", filling(first),
                            object=thing, verb=verb, said=text)

    # `what does Mary have`
    if first == "what" and tokens[1] in DID and tokens[-1] in HAVE:
        holder = named(2, len(tokens) - 1, opener=tokens[1])
        if holder is not None:
            return Goal("object", "holding", subject=holder,
                        verb=tokens[-1], said=text)

    # `where did Mary go`, `where did Mary go first`
    if first == "where" and tokens[1] in DID:
        found = read_mention(tokens, 2, lexicon, tokens[1], names=names)
        if found is not None and found.form not in NO_ONE \
                and found.end < len(tokens):
            rest = tokens[found.end:]
            sequence = None
            if len(rest) == 2 and rest[-1] in SEQUENCE:
                sequence, rest = SEQUENCE[rest[-1]], rest[:-1]
            if len(rest) == 1:
                return Goal("place", "occurrence", subject=found,
                            verb=lexicon.lemma(rest[0]), sequence=sequence,
                            said=text)
    return None


class Answering:
    """The operators, over one session's memory."""

    def __init__(self, session) -> None:
        self.session = session

    @property
    def story(self):
        return self.session.story

    @property
    def discourse(self):
        return self.session.discourse

    def operators(self) -> list[Operator]:
        def on(relation: str):
            return lambda memory: (memory.get("goal") is not None
                                   and memory["goal"].relation == relation)

        return [
            Operator("located", self._step(self.located), on("located"),
                     rule="T3"),
            Operator("holding", self._step(self.holding), on("holding"),
                     rule="T4"),
            Operator("occurrence place", self._step(self.occurrence_place),
                     on("occurrence"), rule="T5"),
        ]

    @staticmethod
    def _step(answer):
        def apply(memory: dict) -> str:
            return answer(memory["goal"], memory["reading"], memory["turn"])
        return apply

    # -- what may fill a slot --------------------------------------------------
    def candidates(self, who: str) -> list:
        session = self.session
        people = session._individuals(SPEAKER_KIND)
        if who == "people":
            return people
        if who == "things" or who in self.story.ANYTHING:
            ids = {one.id for one in people}
            return [one for one in session._individuals() if one.id not in ids]
        return session._individuals(who)

    def _nothing(self, reading, turn, text: str) -> str:
        """Nothing found: another operator's to answer, unless nothing else
        would read the question."""
        if reading.act != "generic":
            return DECLINED
        turn.answer = {"outcome": "unknown", "source": "conversation",
                       "text": f"not told — {text}"}
        return ANSWERED

    def _names(self, found: list) -> str:
        names = [self.discourse.describe(one) for one, _ in found]
        return (", ".join(names[:-1]) + " and " + names[-1]
                if len(names) > 1 else "".join(names))

    def _why(self, found: list) -> str:
        return "; ".join(f"{self.discourse.describe(one)}: "
                         f"{self.story.quote(basis)}" for one, basis in found)

    # -- located ------------------------------------------------------------------
    def located(self, goal: Goal, reading, turn) -> str:
        """Who or what is at a place: where each one is as the story stands,
        carried or not (`Timeline.whereabouts`)."""
        story = self.story
        place = story._resolved(goal.object)
        word = goal.object.kind
        there = (self.discourse.describe(place) if place is not None
                 else f"the {word}")
        found = []
        for one in self.candidates(goal.who):
            if place is not None and one.id == place.id:
                continue
            history = story.timeline.whereabouts(one.id)
            if not history or history[-1][0] is None:
                continue
            key, where, basis = history[-1]
            if (place is not None and key == place.id) or where == word:
                found.append((one, basis))
        anyone = ("nobody here" if goal.who == "people" else
                  "nothing here" if goal.who == "things" else
                  f"no {goal.who} here")
        if goal.asked == "count":
            counts = story.COUNTS
            count = (counts[len(found)] if len(found) < len(counts)
                     else str(len(found)))
            text = (f"{count} — {self._why(found)}" if found else
                    f"none — {anyone} was said to be in {there} now")
            turn.answer = {"outcome": "retrieved", "source": "told",
                           "text": text + " (T4, T3)"}
            return ANSWERED
        if not found:
            return self._nothing(reading, turn,
                                 f"{anyone} was said to be in {there} now")
        if goal.asked == "any":
            turn.answer = {"outcome": "verified", "source": "told",
                           "text": f"yes — {self._why(found)} (T4, T3)"}
            return ANSWERED
        turn.answer = {"outcome": "retrieved", "source": "told",
                       "text": f"{self._names(found)} — {self._why(found)} "
                               f"(T4, T3)"}
        return ANSWERED

    # -- holding ------------------------------------------------------------------
    def holding(self, goal: Goal, reading, turn) -> str:
        """What is with whom, where VerbNet reads the verb as having its
        object with you (`change.accompanies`): `has`, `is carrying`,
        `holds`. A verb it does not read so is another operator's."""
        session, story = self.session, self.story
        if not changes.accompanies(goal.verb,
                                   session.asker.verb_senses(goal.verb)):
            return DECLINED
        if goal.asked == "object":
            holder = story._resolved(goal.subject)
            if holder is None:
                return DECLINED
            held = [(self.discourse.by_id(one), basis)
                    for one, basis in story.held_by(holder.id)]
            held = [(one, basis) for one, basis in held if one is not None]
            if not held:
                return self._nothing(
                    reading, turn, f"nothing was said to be with "
                                   f"{self.discourse.describe(holder)} now")
            turn.answer = {"outcome": "retrieved", "source": "told",
                           "text": f"{self._names(held)} — "
                                   f"{self._why(held)} (T4, T3)"}
            return ANSWERED
        thing = story._resolved(goal.object)
        if thing is None:
            return DECLINED
        found = []
        for one in self.candidates(goal.who):
            for held, basis in story.held_by(one.id):
                if held == thing.id:
                    found.append((one, basis))
        described = self.discourse.describe(thing)
        if not found:
            return self._nothing(reading, turn,
                                 f"nothing said puts {described} with anyone "
                                 f"now")
        turn.answer = {"outcome": "retrieved", "source": "told",
                       "text": f"{self._names(found)} — "
                               f"{described}: {self.story.quote(found[-1][1])}"
                               f" (T4, T3)"}
        return ANSWERED

    # -- where an occurrence went ----------------------------------------------
    def occurrence_place(self, goal: Goal, reading, turn) -> str:
        """`where did Mary go`: the occurrences of the verb with her as their
        subject, in story order (T5, T1), and where each one went; the last
        one, or the one the sequence word picks."""
        story = self.story
        referent = story._resolved(goal.subject)
        if referent is None:
            return DECLINED
        wanted = {f"subject {referent.id}", f"is_a {goal.verb}"}
        found = [one for one in story.timeline.identify(wanted)
                 if one.place or one.place_word]
        described = self.discourse.describe(referent)
        if not found:
            return self._nothing(reading, turn,
                                 f"nothing was said of where {described} "
                                 f"would {goal.verb}")
        if goal.sequence in (None, -1):
            chosen = len(found) - 1
        elif 1 <= goal.sequence <= len(found):
            chosen = goal.sequence - 1
        else:
            turn.answer = {"outcome": "unknown", "source": "told",
                           "text": f"not told — {described} did that "
                                   f"{len(found)} time"
                                   f"{'' if len(found) == 1 else 's'}"}
            return ANSWERED

        def where(one) -> str:
            thing = (self.discourse.by_id(one.place) if one.place else None)
            return (self.discourse.describe(thing) if thing is not None
                    else f"the {one.place_word}")

        one = found[chosen]
        text = f"{where(one)} — you told me “{one.said}”"
        before = found[:chosen]
        if goal.sequence is None and before:
            text += "; before that, " + ", then ".join(
                where(earlier) for earlier in before)
        turn.answer = {"outcome": "retrieved", "source": "told",
                       "text": text + " (T5, T1)"}
        return ANSWERED
