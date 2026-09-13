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

    asked      what the answer fills: the subject, the object, the place, the
               time, a count, whether there is any, or whether it holds
    relation   located (being at a place), holding (having with one), or an
               occurrence of a verb
    who        what may fill the subject: people, things, or a kind
    subject    the one it is about, when that is named
    object     the thing held, or the place
    verb       the verb said, for VerbNet to read
    clause     the verb phrase, read as `reading.py` reads one, for an
               occurrence to be identified by (`story._wanted`)

The operators propose on the relation and read the answer out of the slot
asked, so each relation answers every question its slots can make:

    located     who is in the kitchen, is anyone in it, how many people
                are in it, what is in it, when was Mary in it -- where each
                one is as the story stands (T4, T3)
    holding     who has the football, is anyone carrying it, does Mary have
                it, what does Mary have, how many things does she have --
                what is with whom, where VerbNet reads the verb as having it
                with you (T4); one place at a time (T3) is why a football
                with John is not with Mary
    occurrence  where did Mary go (first), where did she drop the football,
                did anyone go to the garden, how many people went to the
                kitchen -- the occurrences of the verb, identified as `who`
                identifies them (T5), in story order (T1)

They run in front of the act operators and **decline when they find
nothing**, so what a pattern answered before it still answers -- `what does
it have` of a beagle told to have a tail is what was told -- unless nothing
else would read the question at all, when "not told" is the answer rather
than a question about kinds.
"""
from __future__ import annotations

from dataclasses import dataclass

from research.v687.executive import ANSWERED, DECLINED, Operator
from research.v687.links import link

from . import change as changes
from .discourse import SPEAKER_KIND
from .reading import (AUX, COPULA, PLACES, SEQUENCE, Mention, Reading, _cell,
                      _individual, _with_object, read_mention, tags_of,
                      tokens_of)

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

    def as_dict(self) -> dict:
        return {"asked": self.asked, "relation": self.relation,
                "who": self.who,
                "subject": self.subject.text if self.subject else None,
                "object": self.object.text if self.object else None,
                "verb": self.verb, "sequence": self.sequence,
                "clause": " ".join(self.clause.rest) if self.clause else None}


# -- one grammar for the cells patterns used to read -----------------------------
#: What a question word asks, before the rest says of which relation.
OPENERS = ((("how", "many", "times"), "times"), (("how", "often"), "times"),
           (("when",), "time"), (("where",), "place"), (("who",), "subject"),
           (("whom",), "subject"), (("what",), "object"))

#: The act each cell was a pattern for, kept as the reading's name: the page
#: shows it, and nothing dispatches on it (`Session.say` answers the cell).
ACTS = {("time", "occurrence"): "when",
        ("times", "occurrence"): "how_many_times",
        ("place", "located"): "where",
        ("subject", "occurrence"): "who",
        ("recipient", "occurrence"): "to_whom",
        ("object", "occurrence"): "what_did",
        ("object", "holding"): "carrying",
        ("count", "holding"): "carrying"}


def cell_reading(tokens: list[str], lexicon, names: frozenset,
                 said: str) -> Reading | None:
    """What a question asks, of which relation, read by one grammar.

    The question word says what is asked -- `when` a time, `how many times`
    and `how often` a count of occurrences, `where` a place, `who` the
    subject, `what` the object, and `how many` before a noun a count of things
    -- and what follows it says of which relation:

        AUX  NP  VERB-PHRASE          an occurrence   when did the dog bark
        COPULA  NP  [before|after X]  located         where is the key
        COPULA  NP  PLACE             located         what is the cat on
        COPULA  NP  V-ing             holding         what is Mary carrying
        AUX  NP  ...  to|from         its recipient   who did Fred give it to
        [AUX]  [not]  VERB-PHRASE     its subject     who chased the cat
        AUX  NP  VERB-PHRASE          its object      what did the dog chase

    Seven patterns in `reading.py` used to read these one shape each. The
    reading given for each cell is the one its pattern gave, so the handlers
    that fill the cell read it as they did; what it does not read is left
    for the patterns that remain (`what did it do`, `what can it do`).
    """
    if len(tokens) < 2:
        return None
    opened = next(((len(words), slot) for words, slot in OPENERS
                   if tuple(tokens[:len(words)]) == words), None)
    if opened is None:
        if tokens[:2] != ["how", "many"]:
            return None
        opened = (3, "count")           # `how many objects`: things counted
    at, slot = opened
    if at >= len(tokens):
        return None
    head = tokens[at]

    def reading(cell: tuple, *args, **kwargs) -> Reading:
        return _cell(Reading(ACTS[cell], *args, said=said, **kwargs), *cell)

    # an occurrence, and its time or how many times: `when did the dog bark`
    if slot in ("time", "times"):
        if len(tokens) < at + 3 or head not in AUX:
            return None
        found = read_mention(tokens, at + 1, lexicon, head, names=names)
        if _individual(found) and found.end < len(tokens):
            return _with_object(reading((slot, "occurrence"), found, head,
                                        tokens[found.end:]), lexicon, names)
        return None

    # holding: `what is Mary carrying`, `how many objects is Mary carrying` --
    # if the verb means having it with you, which the handler asks VerbNet
    if (slot in ("object", "count") and len(tokens) >= at + 3
            and head in COPULA and tokens[-1].endswith("ing")
            and hasattr(lexicon, "progressive")):
        found = read_mention(tokens[:-1], at + 1, lexicon, head,
                             final_ok=True, names=names)
        verb = lexicon.progressive(tokens[-1])
        if _individual(found) and found.end == len(tokens) - 1 and verb:
            counting = slot == "count"
            return reading((slot, "holding"), found, head, [verb],
                           count=counting,
                           obj=(Mention("kind", tokens[2], text=tokens[2])
                                if counting else None))
    if slot == "count":
        return None

    if slot == "subject":
        # its recipient: `who did Fred give the football to`
        if len(tokens) >= 5 and head in AUX and tokens[-1] in ("to", "from"):
            found = read_mention(tokens, 2, lexicon, head, names=names)
            if _individual(found) and found.end < len(tokens) - 1:
                asked = _with_object(
                    reading(("recipient", "occurrence"), found, head,
                            tokens[found.end:-1]), lexicon, names)
                asked.rest = asked.rest + [tokens[-1]]
                return asked
        # its subject: `who chased the cat`, `who did not bark`
        if head in COPULA:
            return None
        aux = head if head in AUX else None
        rest = tokens[2:] if aux else tokens[1:]
        holds = rest[:1] != ["not"]
        rest = rest if holds else rest[1:]
        if not rest:
            return None
        return _with_object(reading(("subject", "occurrence"), None, aux,
                                    rest, holds=holds), lexicon, names)

    # located, a place asked: `where is the key`, `where was the football
    # before the bathroom` (`story.where_around`)
    if slot == "place":
        if head not in COPULA:
            return None
        found = read_mention(tokens, 2, lexicon, "is", final_ok=True,
                             names=names)
        if _individual(found) and found.end == len(tokens):
            return reading(("place", "located"), found)
        if (_individual(found) and len(tokens) > found.end + 1
                and tokens[found.end] in ("before", "after")):
            return reading(("place", "located"), found, head,
                           tokens[found.end:])
        return None

    # `what`: a place (`what is the cat on`) or an occurrence's object
    if head in COPULA:
        if tokens[-1] in PLACES:
            found = read_mention(tokens[:-1], 2, lexicon, "is",
                                 final_ok=True, names=names)
            if _individual(found) and found.end == len(tokens) - 1:
                return reading(("place", "located"), found)
        return None
    if head not in AUX:
        return None
    start, holds = (3, False) if tokens[2:3] == ["not"] else (2, True)
    found = read_mention(tokens, start, lexicon, head, names=names)
    if not _individual(found) or found.form in ("speaker", "addressee"):
        return None
    rest = tokens[found.end:]
    if (not rest or rest in (["do"], ["have"])
            or (rest[:1] == ["do"] and rest[1:2] and rest[1] in SEQUENCE)):
        return None                     # what it did, can do, has: patterns
    return reading(("object", "occurrence"), found, head, rest, holds=holds)


def read_goal(text: str, lexicon, names: frozenset = frozenset()
              ) -> Goal | None:
    """The goal a question states, or None when it is not one of these."""
    tokens, _ = tokens_of(text)
    if len(tokens) < 3:
        return None
    first = tokens[0]

    def someone(found) -> bool:
        return found is not None and found.form not in NO_ONE

    def named(at: int, upto: int | None = None, opener: str = "is"):
        """The individual named from `at` to the end (or to `upto`)."""
        span = tokens if upto is None else tokens[:upto]
        found = read_mention(span, at, lexicon, opener, final_ok=True,
                             names=names)
        return found if someone(found) and found.end == len(span) else None

    def filling(word: str) -> str:
        return ("people" if word in ("who", "whom") or word in PEOPLE
                else "things")

    def kind_of(words: list[str]) -> str:
        who = " ".join(words[:-1] + [lexicon.lemma(words[-1])])
        return "people" if who in PERSONS else who

    def clause(aux, rest, subject=None) -> Reading:
        return _with_object(Reading("who", subject, aux, list(rest),
                                    said=text), lexicon, names)

    def progressive(word: str) -> str:
        if not word.endswith("ing") or not hasattr(lexicon, "progressive"):
            return ""
        return lexicon.progressive(word) or ""

    # `who is in the kitchen`, `what is in the box`
    if first in ("who", "what") and tokens[1] in COPULA \
            and tokens[2] in PLACING:
        place = named(3)
        if place is not None:
            return Goal("subject", "located", filling(first), object=place,
                        said=text)

    # `when was Mary in the kitchen`
    if first == "when" and tokens[1] in COPULA:
        found = read_mention(tokens, 2, lexicon, tokens[1], names=names)
        if someone(found) and found.end + 1 < len(tokens) \
                and tokens[found.end] in PLACING:
            place = named(found.end + 1)
            if place is not None:
                return Goal("time", "located", subject=found, object=place,
                            said=text)

    # `is anyone in the kitchen`, `is anyone carrying the football`
    if first in COPULA and tokens[1] in PEOPLE | THINGS and len(tokens) > 3:
        if tokens[2] in PLACING:
            place = named(3)
            if place is not None:
                return Goal("any", "located", filling(tokens[1]),
                            object=place, said=text)
        verb = progressive(tokens[2])
        if verb:
            thing = named(3)
            if thing is not None:
                return Goal("any", "holding", filling(tokens[1]),
                            object=thing, verb=verb, said=text)

    if first in DID and len(tokens) > 3:
        # `does anyone have the football`
        if tokens[1] in PEOPLE and tokens[2] in HAVE:
            thing = named(3)
            if thing is not None:
                return Goal("any", "holding", "people", object=thing,
                            verb=tokens[2], said=text)
        # `did anyone go to the garden`
        if tokens[1] in PEOPLE | THINGS:
            return Goal("any", "occurrence", filling(tokens[1]),
                        clause=clause(first, tokens[2:]), said=text)
        # `does Mary have the football`
        holder = read_mention(tokens, 1, lexicon, first, names=names)
        if someone(holder) and holder.end + 1 < len(tokens) \
                and tokens[holder.end] in HAVE:
            thing = named(holder.end + 1)
            if thing is not None:
                return Goal("whether", "holding", subject=holder,
                            object=thing, verb=tokens[holder.end], said=text)

    # `is Mary carrying the football`
    if first in COPULA and len(tokens) > 3:
        holder = read_mention(tokens, 1, lexicon, first, names=names)
        if someone(holder) and holder.end + 1 < len(tokens):
            verb = progressive(tokens[holder.end])
            thing = named(holder.end + 1) if verb else None
            if thing is not None:
                return Goal("whether", "holding", subject=holder,
                            object=thing, verb=verb, said=text)

    if tokens[:2] == ["how", "many"] and len(tokens) > 3:
        # `how many people are in the kitchen`
        at = next((index for index in range(3, len(tokens))
                   if tokens[index] in COPULA), None)
        if at is not None and at + 2 < len(tokens) \
                and tokens[at + 1] in PLACING:
            place = named(at + 2)
            if place is not None:
                return Goal("count", "located", kind_of(tokens[2:at]),
                            object=place, said=text)
        # `how many things does Mary have`
        at = next((index for index in range(3, len(tokens))
                   if tokens[index] in DID), None)
        if at is not None and tokens[-1] in HAVE and at + 1 < len(tokens) - 1:
            holder = named(at + 1, len(tokens) - 1, opener=tokens[at])
            if holder is not None:
                return Goal("count", "holding", kind_of(tokens[2:at]),
                            subject=holder, verb=tokens[-1], said=text)
        # `how many people went to the kitchen`
        tags = tags_of(tokens, lexicon)
        if tokens[3] not in AUX and tags and tags[3].startswith("VB"):
            return Goal("count", "occurrence", kind_of(tokens[2:3]),
                        clause=clause(None, tokens[3:]), said=text)

    # `who has the football`, `who is carrying the football`
    if first in ("who", "what"):
        verb, at = "", 0
        if tokens[1] in HAS:
            verb, at = lexicon.lemma(tokens[1]), 2
        elif tokens[1] in COPULA and len(tokens) > 3:
            verb, at = progressive(tokens[2]), 3
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

    # `where did Mary go`, `where did Mary go first`, `where did Mary drop
    # the football`
    if first == "where" and tokens[1] in DID:
        found = read_mention(tokens, 2, lexicon, tokens[1], names=names)
        if someone(found) and found.end < len(tokens):
            rest = tokens[found.end:]
            sequence = None
            if len(rest) == 2 and rest[-1] in SEQUENCE:
                sequence, rest = SEQUENCE[rest[-1]], rest[:-1]
            said = clause(tokens[1], rest, found)
            if len(rest) == 1 or said.obj is not None:
                return Goal("place", "occurrence", subject=found,
                            verb=lexicon.lemma(rest[0]), sequence=sequence,
                            clause=said, said=text)
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
            Operator("occurrence", self._step(self.occurrence),
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
        if who in ("people", ""):
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

    def _count(self, found: list) -> str:
        counts = self.story.COUNTS
        return (counts[len(found)] if len(found) < len(counts)
                else str(len(found)))

    @staticmethod
    def _anyone(who: str) -> str:
        return ("nobody here" if who in ("people", "") else
                "nothing here" if who == "things" else f"no {who} here")

    def _answer(self, turn, outcome: str, text: str, rules: str) -> str:
        turn.answer = {"outcome": outcome, "source": "told",
                       "text": f"{text} ({rules})"}
        return ANSWERED

    # -- located ------------------------------------------------------------------
    def located(self, goal: Goal, reading, turn) -> str:
        """Who or what is at a place, and when one was: where each one is as
        the story stands, carried or not (`Timeline.whereabouts`)."""
        story = self.story
        place = story._resolved(goal.object)
        word = goal.object.kind
        there = (self.discourse.describe(place) if place is not None
                 else f"the {word}")

        def at_place(key, where) -> bool:
            return (place is not None and key == place.id) or where == word

        if goal.asked == "time":
            return self._when_located(goal, reading, turn, at_place, there)
        found = []
        for one in self.candidates(goal.who):
            if place is not None and one.id == place.id:
                continue
            history = story.timeline.whereabouts(one.id)
            if not history or history[-1][0] is None:
                continue
            key, where, basis = history[-1]
            if at_place(key, where):
                found.append((one, basis))
        anyone = self._anyone(goal.who)
        if goal.asked == "count":
            return self._answer(
                turn, "retrieved",
                f"{self._count(found)} — {self._why(found)}" if found else
                f"none — {anyone} was said to be in {there} now", "T4, T3")
        if not found:
            return self._nothing(reading, turn,
                                 f"{anyone} was said to be in {there} now")
        if goal.asked == "any":
            return self._answer(turn, "verified",
                                f"yes — {self._why(found)}", "T4, T3")
        return self._answer(turn, "retrieved",
                            f"{self._names(found)} — {self._why(found)}",
                            "T4, T3")

    def _when_located(self, goal: Goal, reading, turn, at_place,
                      there: str) -> str:
        """`when was Mary in the kitchen`: each time the story put her
        there, situated as `when` situates an occurrence (T1, T2)."""
        story = self.story
        referent = story._resolved(goal.subject)
        if referent is None:
            return DECLINED
        times = []
        for key, where, basis in story.timeline.whereabouts(referent.id):
            if key is None or not at_place(key, where):
                continue
            occurrence = (story.timeline.occurrence(basis.occurrence)
                          if hasattr(basis, "occurrence") else None)
            if occurrence is not None:
                times.append(story.situate(occurrence))
            else:
                episode = story.timeline.episode(getattr(basis, "episode",
                                                         None))
                times.append(f"{story.phrase(episode)} — "
                             f"{story.quote(basis)}")
        described = self.discourse.describe(referent)
        if not times:
            return self._nothing(reading, turn,
                                 f"nothing said put {described} in {there}")
        return self._answer(turn, "retrieved", "; ".join(times), "T1, T3")

    # -- holding ------------------------------------------------------------------
    def holding(self, goal: Goal, reading, turn) -> str:
        """What is with whom, where VerbNet reads the verb as having its
        object with you (`change.accompanies`): `has`, `is carrying`,
        `holds`. A verb it does not read so is another operator's."""
        session, story = self.session, self.story
        if not changes.accompanies(goal.verb,
                                   session.asker.verb_senses(goal.verb)):
            return DECLINED
        if goal.asked in ("object", "count"):
            return self._held(goal, reading, turn)
        thing = story._resolved(goal.object)
        if thing is None:
            return DECLINED
        described = self.discourse.describe(thing)
        holders = [(one, basis) for one in self.candidates(goal.who or
                                                           "people")
                   for held, basis in story.held_by(one.id)
                   if held == thing.id]
        if goal.asked == "whether":
            holder = story._resolved(goal.subject)
            if holder is None:
                return DECLINED
            mine = [(one, basis) for one, basis in holders
                    if one.id == holder.id]
            if mine:
                return self._answer(
                    turn, "verified",
                    f"yes — {described}: {story.quote(mine[-1][1])}",
                    "T4, T3")
            others = [(one, basis) for one, basis in holders
                      if one.id != holder.id]
            if others and link("at_location").exclusive:
                # T3: one place at a time -- with someone else is not here.
                return self._answer(
                    turn, "denied",
                    f"no — {described} is with {self._names(others)}: "
                    f"{story.quote(others[-1][1])}", "T4, T3")
            return self._nothing(
                reading, turn, f"nothing said puts {described} with "
                               f"{self.discourse.describe(holder)} now")
        if not holders:
            return self._nothing(reading, turn,
                                 f"nothing said puts {described} with anyone "
                                 f"now")
        if goal.asked == "any":
            return self._answer(
                turn, "verified",
                f"yes — {self._names(holders)}: "
                f"{story.quote(holders[-1][1])}", "T4, T3")
        return self._answer(turn, "retrieved",
                            f"{self._names(holders)} — {described}: "
                            f"{story.quote(holders[-1][1])}", "T4, T3")

    def _held(self, goal: Goal, reading, turn) -> str:
        """`what does Mary have`, `how many things does Mary have`."""
        story = self.story
        holder = story._resolved(goal.subject)
        if holder is None:
            return DECLINED
        held = [(self.discourse.by_id(one), basis)
                for one, basis in story.held_by(holder.id)]
        held = [(one, basis) for one, basis in held if one is not None]
        if goal.asked == "count" and goal.who not in ("", "things") \
                and goal.who not in story.ANYTHING:
            kind = {one.id for one in self.candidates(goal.who)}
            held = [(one, basis) for one, basis in held if one.id in kind]
        described = self.discourse.describe(holder)
        if goal.asked == "count":
            return self._answer(
                turn, "retrieved",
                f"{self._count(held)} — {self._why(held)}" if held else
                f"none — nothing was said to be with {described} now",
                "T4, T3")
        if not held:
            return self._nothing(reading, turn, f"nothing was said to be "
                                                f"with {described} now")
        return self._answer(turn, "retrieved",
                            f"{self._names(held)} — {self._why(held)}",
                            "T4, T3")

    # -- occurrences -----------------------------------------------------------------
    def _occurrences(self, goal: Goal, subject=None) -> list | None:
        """The occurrences the goal's verb phrase names, in story order,
        identified as `who` identifies them (T5); None when an object it
        names is not one of the individuals here."""
        story = self.story
        clause = goal.clause
        if clause is None:
            return story.timeline.identify(
                {f"subject {subject.id}", f"is_a {goal.verb}"})
        other = None
        if clause.obj is not None and clause.obj.form not in NO_ONE:
            other = story._resolved(clause.obj,
                                    exclude={subject.id} if subject else ())
            if other is None:
                return None
        verb = story._verb_asked(clause)
        return story.timeline.identify(story._wanted(clause, subject, other,
                                                     verb))

    def occurrence(self, goal: Goal, reading, turn) -> str:
        if goal.asked == "place":
            return self._place(goal, reading, turn)
        found = self._occurrences(goal) or []
        allowed = {one.id for one in self.candidates(goal.who)}
        doers, seen = [], set()
        for one in found:
            doer = self.discourse.by_id(one.subject or "")
            if doer is not None and doer.id in allowed \
                    and doer.id not in seen:
                seen.add(doer.id)
                doers.append((doer, one))
        what = " ".join(goal.clause.rest)
        anyone = self._anyone(goal.who)
        if goal.asked == "count":
            return self._answer(
                turn, "retrieved",
                f"{self._count(doers)} — {self._why(doers)}" if doers else
                f"none — {anyone} was said to {what}", "T5")
        if not doers:
            return self._nothing(reading, turn,
                                 f"{anyone} was said to {what}")
        return self._answer(turn, "verified", f"yes — {self._why(doers)}",
                            "T5")

    def _place(self, goal: Goal, reading, turn) -> str:
        """`where did Mary go`: the occurrences of the verb with her as their
        subject, in story order (T5, T1), and where each one went; the last
        one, or the one the sequence word picks."""
        story = self.story
        referent = story._resolved(goal.subject)
        if referent is None:
            return DECLINED
        found = [one for one in (self._occurrences(goal, referent) or [])
                 if one.place or one.place_word]
        described = self.discourse.describe(referent)
        if not found:
            what = (" ".join([goal.verb] + goal.clause.rest[1:])
                    if goal.clause is not None else goal.verb)
            return self._nothing(reading, turn,
                                 f"nothing was said of where {described} "
                                 f"would {what}")
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
        return self._answer(turn, "retrieved", text, "T5, T1")
