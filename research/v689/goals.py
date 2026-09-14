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

Every other cell is answered by its relation's operator too, by the handler
written for it (`Answering.cells`): `where is Mary` by the located operator,
`who chased the cat` by the occurrence operator, `what color is Greg` by the
attribute operator. One operator a relation, one method a cell.
"""
from __future__ import annotations

from research.v687.executive import ANSWERED, DECLINED, Operator
from research.v687.links import link

from . import change as changes
from .discourse import SPEAKER_KIND
from .grammar import NO_ONE, Goal



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

    #: The rule each relation's operator mainly carries out, for its trace.
    RULES = {"located": "T3", "holding": "T4", "occurrence": "T5",
             "dimension": "S2", "attribute": "I1", "is_a": "R1",
             "story": "T1", "future": "T1"}

    def cells(self) -> dict[str, dict]:
        """Every cell, by relation: what answers what is asked of it. The
        cells memory is composed for are this class's; the rest are the
        handlers written for them, which always answer."""
        session, story = self.session, self.story

        def handled(handler):
            def answer(goal: Goal, reading, turn) -> str:
                handler(reading, turn)
                return ANSWERED
            return answer

        happened = handled(session._happened)
        return {
            "located": {"subject": self.located, "any": self.located,
                        "count": self.located, "time": self.located,
                        "place": handled(session._where)},
            "holding": {"any": self.holding, "whether": self.holding,
                        "subject": self.holding, "object": self.holding,
                        "count": self.holding},
            "occurrence": {"any": self.occurrence, "count": self.occurrence,
                           "place": self.occurrence,
                           "time": handled(story.when_asked),
                           "times": handled(story.how_many_times),
                           "subject": handled(session._who),
                           "recipient": handled(story.to_whom),
                           "object": handled(session._what_did),
                           "verb": handled(story.doing)},
            "dimension": {"subject": handled(session._related_to),
                          "object": handled(session._related_to),
                          "path": handled(session._route)},
            "attribute": {"value": handled(session._attribute),
                          "object": handled(session._toward)},
            "motive": {"place": handled(session._where_going)},
            "is_a": {"count": handled(session._how_many),
                     "which": handled(session._which),
                     "kind": handled(session._what)},
            "story": {"events": happened},
            "told": {"events": happened},
            "future": {"events": happened},
            "any": {"facts": handled(session._about)},
            "answer": {"grounds": handled(session._meta)},
            "question": {"again": handled(session._ellipsis)},
        }

    def operators(self) -> list[Operator]:
        """One operator a relation: proposed when a goal of it is, and trying
        each such goal, in the order they were proposed, until one is
        answered."""
        cells = self.cells()

        def goals(memory: dict, relation: str) -> list:
            return [one for one in memory.get("goals", ())
                    if one.relation == relation
                    and one.asked in cells[relation]]

        def operator(relation: str) -> Operator:
            def apply(memory: dict) -> str:
                for goal in goals(memory, relation):
                    answer = cells[relation][goal.asked]
                    if answer(goal, memory["reading"],
                              memory["turn"]) == ANSWERED:
                        return ANSWERED
                return DECLINED
            return Operator(relation, apply,
                            lambda memory: bool(goals(memory, relation)),
                            rule=self.RULES.get(relation, ""))

        return [operator(relation) for relation in cells]

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
        `holds`. A verb it does not read so is another operator's.

        A question whose own words ask what someone is carrying resolves
        them as a question's subject is resolved (`_carried`)."""
        session, story = self.session, self.story
        if goal.clause is reading and goal.asked in ("object", "count"):
            return self._carried(reading, turn)
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

    def _carried(self, reading, turn) -> str:
        """`what is Mary carrying`, `how many objects is Mary carrying`: what
        is with her -- when VerbNet reads the verb as having something with
        you (`change.accompanies`) -- counted, where the question counts. No
        other reading of the question would answer it, so what was not found
        is said."""
        session, story = self.session, self.story
        referent = session._here(reading, turn)
        if referent is None:
            return ANSWERED
        described = self.discourse.describe(referent)
        verb = reading.rest[0] if reading.rest else ""
        if not changes.accompanies(verb, story.asker.verb_senses(verb)):
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"not told — VerbNet does not read "
                                   f"“{verb}” as having something with you, "
                                   f"and nothing was said of what "
                                   f"{described} is {verb}ing"}
            return ANSWERED
        held = story.held_by(referent.id)
        kind = (story.asker.lemma(reading.obj.kind)
                if reading.obj is not None else "")
        if kind and kind not in story.ANYTHING:
            ones = {one.id for one in session._individuals(kind)}
            held = [one for one in held if one[0] in ones]
        names = [self.discourse.describe(self.discourse.by_id(one))
                 for one, _ in held]
        why = "; ".join(f"{name}: {story.quote(basis)}"
                        for name, (_, basis) in zip(names, held))
        if reading.count:
            count = (story.COUNTS[len(held)] if len(held) < len(story.COUNTS)
                     else str(len(held)))
            text = (f"{count} — {why}" if held else
                    f"none — nothing was said to be with {described} now")
        elif held:
            text = (", ".join(names[:-1]) + " and " + names[-1]
                    if len(names) > 1 else names[0]) + f" — {why}"
        else:
            text = f"nothing — nothing was said to be with {described} now"
        turn.answer = {"outcome": "retrieved", "source": "told",
                       "text": text + " (T4, T3)"}
        return ANSWERED

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
