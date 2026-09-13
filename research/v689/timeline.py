"""Story time: episodes, what happened in them, and in what order.

Episodic memory knew who and what, and not when. `the dog barked. then it
slept` stored two abilities, and `what happened first` could only give the
order the two were told in. This is the part of memory that is about time:
a projection of the same stream (`events.py`), kept beside the episodic
tables and planned into a trie of its own.

## Episodes

An **episode** is a stretch of the story at one time. `yesterday`, `this
morning`, `now` and `tomorrow` each name one, and sit in order on the line of
days (`tense.FRAMES`). A story told in the past with no day named is in
`then`, which is before now and in no known order against any day. Every
statement lands in one: the one it names, the present if it is in the
present tense, and the past the story is already in if it is in the past.
Individuals are not in episodes -- the same dog chased the cat yesterday and
sleeps today -- but what happens to them, and what they are like at the time,
is.

## Occurrences

An **occurrence** is something that happened, kept the way an individual is:
a node with predicates. Davidson's event is an individual of its verb, so an
occurrence of chasing is stored under every verb chasing is a kind of --
`is_a chase`, `is_a pursue`, `is_a travel`, WordNet's troponymy, walked up by
R1 exactly as a beagle is walked up to dog -- and under who took part, the
episode and the aspect. Planned by `adaptive_coverage` and identified by
walking the trie down (`episodic.walk_down`), as a description finds an
individual.

## The rules

**T1. Told in order, happened in order.** Narrative progression, as discourse
representation theory has it: a clause in the simple past moves the story on,
so it comes after the occurrence the story was at. A progressive is in
progress at that time and does not move it; a past perfect is before it. A
link says otherwise when there is one -- `before that`, `meanwhile`,
`finally` -- and an anchor clause says exactly where: `after the dog chased
the cat, it slept`. Every placement is an `ordered` event with its reason.

**T2. Before is a partial order, walked like the taxonomy.** `before` is
transitive and asymmetric, so `did the dog bark before it slept` is a walk
along `before` edges, as R1 is a walk along `is_a` edges: yes when the one is
reached from the other, no when the other is reached from the one, and not
told when neither is -- absent, not false. Episodes on the line of days are
in order without being walked.

**T3. A state holds within its episode, until something ends it.** What an
individual is like, where it is and what it is doing were told of a time. At a
moment in the same episode, the latest of them before it answers; nothing told
in one episode answers another -- `yesterday the pig was in an airplane` says
nothing of where it is today, and the answer says what was said and when. What
it is, what it is called, whose it is and what it can do are not in time.

**T4. What an occurrence changes** is read from VerbNet (`change.py`): the
state it begins holds after it and did not hold before, and the state it ends
held before and does not hold after.

**T5. An occurrence is an individual of its verb** (above): `who`, `when`,
`what happened` and `how many times` are identification on the occurrence
trie, with one slot read out.

**T6. Nothing happens by inheritance.** What a kind does is a tendency; that
one of them did it is an occurrence. `did the dog bark`, with nothing told of
the dog, is not answered by dogs barking -- the E1 of events.
"""
from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass, field

from research.v687 import walks
from research.v687.links import named
from research.v687.ordering import adaptive_coverage
from research.v687.trie import PredicateTrie

from .episodic import walk_down

#: Relations told of a time: those `links.py` marks `in_time`. What an
#: individual is, what it is called, whose it is and what it can do are not
#: among them.
IN_TIME = named(lambda one: one.in_time)

#: The keys of the three episodes a story can be in without naming a day.
NOW, THEN, LATER = "now", "then", "later"

#: What a judge says of a state told as one of several: `it is either in the
#: school or the park`, asked `is it in the park`.
MAYBE = "maybe"


def somewhere(record) -> bool:
    """Does a told place put it somewhere in particular? `in the kitchen`
    does; `no longer in the kitchen` and `either in the school or the park`
    do not."""
    return (record.relation == "at_location"
            and not record.object.startswith(("no ", "either ")))


@dataclass
class Episode:
    id: str
    key: str
    label: str
    rank: float | None
    tense: str
    turn: int

    @property
    def sort_key(self) -> float:
        """Where it is drawn: on the line of days, and `then` between the
        days before and now, where it is known to be only before now."""
        if self.rank is not None:
            return self.rank
        return -0.5 if self.tense == "past" else 0.5

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Occurrence:
    id: str
    verb: str
    #: every verb it is a kind of, nearest first, as words
    kinds: list
    subject: str | None
    episode: str
    tense: str
    aspect: str
    said: str
    turn: int = 0
    seq: int = 0
    #: the individual after the verb, or the word when it is no one here
    object: str | None = None
    object_word: str = ""
    #: the individual it happened in, on, into or from
    place: str | None = None
    place_word: str = ""
    preposition: str = ""
    #: what episodic memory stores of it: `chase a cat`
    predicate: str = ""
    again: bool = False

    def predicates(self) -> frozenset:
        found = [f"is_a {kind}" for kind in self.kinds]
        for name in ("subject", "object", "place"):
            value = getattr(self, name)
            if value:
                found.append(f"{name} {value}")
        if self.object_word and not self.object:
            found.append(f"object_kind {self.object_word}")
        found += [f"episode {self.episode}", f"aspect {self.aspect}"]
        return frozenset(found)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Change:
    """T4: a state an occurrence began or ended, for one participant."""

    occurrence: str
    individual: str
    #: state | location | activity
    kind: str
    word: str
    after: bool | None
    before: bool | None
    source: str
    #: for a location, the individual it is at, and the word for it
    place: str | None = None
    place_word: str = ""
    seq: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Record:
    """A state told of an individual, and where in the story it was told."""

    seq: int
    individual: str
    relation: str
    object: str
    said: str
    episode: str
    #: the occurrence the story was at when it was told, if any
    after: str | None
    turn: int
    bound: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Holding:
    """What a state came to at a moment, and what that rests on."""

    value: bool | None
    basis: object = None
    rule: str = ""
    episode: Episode | None = None
    #: (episode, value, basis) for every other episode that settles it
    elsewhere: list = field(default_factory=list)


class Timeline:
    """Episodes, occurrences, their order and what holds when."""

    def __init__(self, log) -> None:
        self.log = log
        self.episodes: list[Episode] = []
        #: the episode the story is in: that of the last thing told
        self.current: str | None = None
        self.occurrences: list[Occurrence] = []
        #: episode -> the occurrence the story has reached in it (T1)
        self.reference: dict[str, str] = {}
        #: a -> {b, ...}: a comes before each b
        self.after: dict[str, set] = {}
        self.during: set[frozenset] = set()
        #: (first, second) -> why they are in that order
        self.reasons: dict[tuple, str] = {}
        self.changes: list[Change] = []
        self.records: list[Record] = []
        self.trie = PredicateTrie()
        self.plan: list = []
        self.members: dict[tuple, list] = {}
        self.growth: list[dict] = []
        for kind, handler in (
                ("episode_opened", self._on_opened),
                ("episode_entered", self._on_entered),
                ("occurred", self._on_occurred), ("ordered", self._on_ordered),
                ("changed", self._on_changed), ("told", self._on_told)):
            log.on(kind, handler)

    # -- looking things up -------------------------------------------------
    def episode(self, episode_id: str | None) -> Episode | None:
        return next((one for one in self.episodes if one.id == episode_id),
                    None)

    def by_key(self, key: str) -> Episode | None:
        return next((one for one in self.episodes if one.key == key), None)

    def occurrence(self, occurrence_id: str | None) -> Occurrence | None:
        return next((one for one in self.occurrences
                     if one.id == occurrence_id), None)

    def in_episode(self, episode_id: str) -> list[Occurrence]:
        return [one for one in self.occurrences if one.episode == episode_id]

    def latest(self, tense: str) -> Episode | None:
        """The episode a question in this tense is about when it names none:
        the present if the story has been in it, otherwise where the story
        is; for the past, the latest past the story has been in."""
        current = self.episode(self.current)
        if tense == "present":
            return self.by_key(NOW) or current
        if tense == "future":
            future = [one for one in self.episodes
                      if one.tense == "future"]
            return future[-1] if future else current
        if current is not None and current.tense == "past":
            return current
        past = [one for one in self.episodes if one.tense == "past"]
        return (max(past, key=lambda one: (one.sort_key, one.turn))
                if past else current)

    # -- commands ----------------------------------------------------------
    def enter(self, key: str, label: str, rank: float | None,
              tense: str) -> Episode:
        """The episode a statement is in: found, or opened."""
        found = self.by_key(key)
        if found is None:
            self.log.record("episode_opened", {
                "id": f"e{len(self.episodes) + 1}", "key": key,
                "label": label, "rank": rank, "tense": tense,
                "turn": self.log.turn})
            return self.episodes[-1]
        if self.current != found.id:
            self.log.record("episode_entered", {"id": found.id})
        return found

    def narrate(self, data: dict, link: str = "", anchor: str | None = None,
                relation: str = "") -> Occurrence:
        """T1: record an occurrence and where it falls.

        `data` is the occurrence's fields. `link` places it against the
        occurrence the story is at; `anchor`, an occurrence already here,
        with `relation` saying how.
        """
        episode = data["episode"]
        at = self.reference.get(episode)
        aspect = data.get("aspect", "simple")
        new = f"o{len(self.occurrences) + 1}"
        orders: list[tuple[str, str, str, str]] = []
        advance = False
        if anchor:
            said = (self.occurrence(anchor).said
                    if self.occurrence(anchor) else anchor)
            if relation == "after":
                orders.append((anchor, new, "before", f"after “{said}”"))
                advance = True
            elif relation == "before":
                orders.append((new, anchor, "before", f"before “{said}”"))
            else:
                orders.append((new, anchor, "during", f"while “{said}”"))
        elif link == "before":
            if at:
                orders.append((new, at, "before", "said to be before it"))
        elif link == "during":
            if at:
                orders.append((new, at, "during", "said to be meanwhile"))
        elif link == "last":
            orders += [(one.id, new, "before", "said to come last")
                       for one in self.in_episode(episode)]
            advance = True
        elif aspect == "perfect":
            if at:
                orders.append((new, at, "before",
                               "a past perfect is before the time the story "
                               "is at"))
        elif aspect == "progressive":
            if at:
                orders.append((new, at, "during",
                               "a progressive is in progress at the time the "
                               "story is at"))
        else:
            if at:
                orders.append((at, new, "before",
                               "said to come after it" if link == "after"
                               else "told after it, in the simple past: "
                                    "narrative progression"))
            advance = True
        self.log.record("occurred", {**data, "id": new, "advance": advance})
        for first, second, how, why in orders:
            self.log.record("ordered", {"first": first, "second": second,
                                        "how": how, "why": why})
        return self.occurrences[-1]

    def change(self, **data) -> Change:
        """T4: record a state an occurrence began or ended."""
        self.log.record("changed", data)
        return self.changes[-1]

    # -- handlers ----------------------------------------------------------
    def _on_opened(self, event) -> None:
        data = event.data
        self.episodes.append(Episode(data["id"], data["key"], data["label"],
                                     data.get("rank"), data["tense"],
                                     int(data.get("turn") or 0)))
        self.current = data["id"]

    def _on_entered(self, event) -> None:
        self.current = event.data["id"]

    def _on_occurred(self, event) -> None:
        data = dict(event.data)
        advance = data.pop("advance", False)
        known = set(Occurrence.__dataclass_fields__)
        occurrence = Occurrence(**{key: value for key, value in data.items()
                                   if key in known})
        occurrence.seq = event.seq
        occurrence.turn = event.turn
        self.occurrences.append(occurrence)
        if advance:
            self.reference[occurrence.episode] = occurrence.id
        self.current = occurrence.episode
        if not self.log.replaying:
            self.replan(f"{occurrence.id} {occurrence.verb}")

    def _on_ordered(self, event) -> None:
        data = event.data
        first, second = data["first"], data["second"]
        if data["how"] == "before":
            self.after.setdefault(first, set()).add(second)
        else:
            self.during.add(frozenset((first, second)))
        self.reasons[(first, second)] = data.get("why", "")

    def _on_changed(self, event) -> None:
        known = set(Change.__dataclass_fields__)
        change = Change(**{key: value for key, value in event.data.items()
                           if key in known})
        change.seq = event.seq
        self.changes.append(change)

    def _on_told(self, event) -> None:
        when = event.data.get("when")
        if not when:
            return
        data = event.data
        self.records.append(Record(
            event.seq, data["node"], data["relation"], data["object"],
            data.get("said", ""), when["episode"], when.get("after"),
            event.turn, [data["bound"]] if data.get("bound") else []))
        self.current = when["episode"]

    # -- T5: the occurrence trie -------------------------------------------
    def replan(self, reason: str) -> dict:
        before = self.trie.node_count
        corpus = tuple((one.id, one.predicates()) for one in self.occurrences)
        self.plan = adaptive_coverage(corpus)
        self.trie, self.members = PredicateTrie(), {}
        for occurrence, path in self.plan:
            self.trie.insert(occurrence, path)
            for cut in range(len(path) + 1):
                self.members.setdefault(tuple(path[:cut]), []).append(
                    occurrence)
        cells = sum(len(predicates) for _, predicates in corpus)
        growth = {"reason": reason, "occurrences": len(corpus),
                  "cells": cells, "nodes": self.trie.node_count,
                  "allocated": self.trie.node_count - before,
                  "shared": (round(1 - self.trie.node_count / cells, 3)
                             if cells else 0.0)}
        self.growth.append(growth)
        return growth

    def identify(self, wanted) -> list[Occurrence]:
        """The occurrences stored beneath a description, in story order."""
        found, _, _ = walk_down(self.trie, self.members, frozenset(wanted))
        return [one for one in self.story() if one.id in found]

    # -- T2: order ---------------------------------------------------------
    def _episode_order(self, one: Episode | None,
                       other: Episode | None) -> str | None:
        if one is None or other is None or one.id == other.id:
            return None
        if one.rank is not None and other.rank is not None:
            return ("before" if one.rank < other.rank else
                    "after" if one.rank > other.rank else None)
        # `then` is before the present and whatever is to come, and `later`
        # after the present and whatever has been; nothing more is known.
        def side(episode: Episode) -> float | None:
            if episode.rank is not None:
                return episode.rank
            return {"past": -0.5, "future": 0.5}.get(episode.tense)

        mine, theirs = side(one), side(other)
        unranked = one.rank is None or other.rank is None
        if mine is None or theirs is None:
            return None
        if unranked and mine < 0 <= theirs or unranked and mine <= 0 < theirs:
            return "before"
        if unranked and theirs < 0 <= mine or unranked and theirs <= 0 < mine:
            return "after"
        return None

    def path(self, start: str, goal: str) -> list[str]:
        """The `before` edges from one occurrence to another, as the
        occurrences on the way, both ends included; [] if none. T2 is the
        one path walk (`walks.path`) along `before`."""
        route = walks.path(start, goal, lambda node: (
            ("before", later) for later in sorted(self.after.get(node, ()))))
        return [start] + [node for _, node in route] if route else []

    def _reaches(self, start: str, goal: str) -> bool:
        return bool(self.path(start, goal))

    def relation(self, first: str, second: str) -> str | None:
        """T2: before | after | during, or None when nothing told says."""
        one, other = self.occurrence(first), self.occurrence(second)
        if one is None or other is None or first == second:
            return None
        by_day = self._episode_order(self.episode(one.episode),
                                     self.episode(other.episode))
        if by_day:
            return by_day
        if self._reaches(first, second):
            return "before"
        if self._reaches(second, first):
            return "after"
        if frozenset((first, second)) in self.during:
            return "during"
        return None

    def positions(self, episode_id: str) -> dict[str, float]:
        """Where each occurrence in an episode falls: T1's edges sorted
        topologically, ties in the order they were told, and what happens
        during another at the same place."""
        ones = self.in_episode(episode_id)
        told = {one.id: index for index, one in enumerate(ones)}
        waiting = {one.id: 0 for one in ones}
        for first in told:
            for second in self.after.get(first, ()):
                if second in waiting:
                    waiting[second] += 1
        ready = [(told[one], one) for one in told if not waiting[one]]
        heapq.heapify(ready)
        order: list[str] = []
        while ready:
            _, node = heapq.heappop(ready)
            order.append(node)
            for second in self.after.get(node, ()):
                if second in waiting:
                    waiting[second] -= 1
                    if not waiting[second]:
                        heapq.heappush(ready, (told[second], second))
        # Told in an order that cannot all hold: the rest as they were told.
        order += [one for one in told if one not in order]
        place = {node: float(index) for index, node in enumerate(order)}
        for pair in self.during:
            first, second = sorted(pair, key=lambda one: told.get(one, -1))
            if first in place and second in place:
                place[second] = place[first]
        return place

    def story(self) -> list[Occurrence]:
        """Every occurrence in story order: episodes on the line of days,
        and within each, T1's order."""
        out: list[Occurrence] = []
        for episode in sorted(self.episodes,
                              key=lambda one: (one.sort_key, one.turn)):
            place = self.positions(episode.id)
            out += sorted(self.in_episode(episode.id),
                          key=lambda one: (place.get(one.id, 0.0), one.seq))
        return out

    # -- T3 and T4: what holds ---------------------------------------------
    def holding(self, individual: str, judge, episode_id: str,
                at: str | None = None, side: str = "end",
                _nested: bool = False) -> Holding:
        """What a state came to at a moment in an episode.

        `judge(kind, relation_or_aspect, word)` says whether a record, a
        change or an occurrence is the state asked about (1), its contrary
        (-1), or neither (0). The moment is the end of the episode, or just
        `before`, `after` or `during` the occurrence `at`.
        """
        episode = self.episode(episode_id)
        place = self.positions(episode_id)
        moment = float("inf")
        if at is not None and at in place:
            moment = place[at] + {"before": -0.1, "during": 0.05,
                                  "after": 0.3}.get(side, 0.3)
        # T3: one place at a time. For a judge of where something is
        # (`judge.exclusive`), being told it is somewhere else, or going
        # somewhere else, ends its being here.
        exclusive = getattr(judge, "exclusive", False)
        found = []      # (position, seq, value, basis, rule, value before)
        for record in self.records:
            if record.individual != individual or record.episode != episode_id:
                continue
            matched = judge("record", record.relation, record.object)
            if not matched and exclusive and somewhere(record):
                matched = -1
            if not matched:
                continue
            position = (place.get(record.after, -1.0) + 0.5 if record.after
                        else -0.5)
            value = MAYBE if matched == MAYBE else matched > 0
            found.append((position, record.seq, value, record, "T3", None))
        for change in self.changes:
            occurrence = self.occurrence(change.occurrence)
            if (change.individual != individual or occurrence is None
                    or occurrence.episode != episode_id):
                continue
            matched = judge("change", change.kind,
                            change.place or change.place_word
                            if change.kind == "location" else change.word)
            if (not matched and exclusive and change.kind == "location"
                    and change.after):
                found.append((place.get(change.occurrence, 0.0) + 0.25,
                              change.seq, False, change, "T3", None))
                continue
            if not matched:
                continue

            def turned(value):
                return None if value is None else (value if matched > 0
                                                   else not value)
            found.append((place.get(change.occurrence, 0.0) + 0.25,
                          change.seq, turned(change.after), change, "T4",
                          turned(change.before)))
        for occurrence in self.in_episode(episode_id):
            if occurrence.subject != individual:
                continue
            matched = judge("doing", occurrence.aspect, occurrence.verb)
            if not matched:
                continue
            # A progressive is in progress; a simple past happened, and
            # whether it goes on was not said.
            value = (matched > 0 if occurrence.aspect == "progressive"
                     else None)
            found.append((place.get(occurrence.id, 0.0), occurrence.seq,
                          value, occurrence, "T5", None))
        earlier = [one for one in found if one[0] <= moment]
        settled = [one for one in earlier if one[2] is not None]
        if settled:
            chosen = max(settled, key=lambda one: (one[0], one[1]))
            return Holding(chosen[2], chosen[3], chosen[4], episode)
        later = sorted((one for one in found
                        if one[0] > moment and one[5] is not None),
                       key=lambda one: (one[0], one[1]))
        if later:
            return Holding(later[0][5], later[0][3], "T4", episode)
        if earlier:
            chosen = max(earlier, key=lambda one: (one[0], one[1]))
            return Holding(None, chosen[3], chosen[4], episode)
        elsewhere = []
        if not _nested:
            for other in self.episodes:
                if other.id == episode_id:
                    continue
                held = self.holding(individual, judge, other.id,
                                    _nested=True)
                if held.basis is not None:
                    elsewhere.append((other, held.value, held.basis))
        return Holding(None, None, "T3", episode, elsewhere)

    def whereabouts(self, individual: str,
                    direct: bool = False) -> list[tuple]:
        """(place, word, basis) for every place an individual came to be, in
        story order: episodes on the line of days, and T1's order within each.

        What is somewhere in or with something else is where that is: a
        football got by Mary is in each room Mary goes to, one after another.
        With `direct`, only what it is itself in or with: Mary. This is the
        order of places (T2), across episodes; whether one still holds is
        `holding`'s question, and T3's.
        """
        steps = []
        places: dict[str, dict] = {}
        episodes = {one.id: one for one in self.episodes}

        def when(episode_id, position, seq):
            episode = episodes.get(episode_id)
            if episode_id not in places:
                places[episode_id] = self.positions(episode_id)
            return ((episode.sort_key, episode.turn) if episode else (0, 0),
                    position, seq)

        for change in self.changes:
            occurrence = self.occurrence(change.occurrence)
            if change.kind == "location" and occurrence is not None:
                at = when(occurrence.episode, 0.0, change.seq)
                at = (at[0], places[occurrence.episode].get(occurrence.id, 0.0)
                      + 0.25, change.seq)
                steps.append((at, change))
        for record in self.records:
            if record.relation != "at_location":
                continue
            at = when(record.episode, 0.0, record.seq)
            position = (places[record.episode].get(record.after, -1.0) + 0.5
                        if record.after else -0.5)
            steps.append(((at[0], position, record.seq), record))
        located: dict[str, tuple] = {}
        history: list[tuple] = []

        def resolved(node):
            found, seen = located.get(node), {node}
            while found and found[0] in located and found[0] not in seen:
                seen.add(found[0])
                found = located[found[0]]
            return found

        for _, step in sorted(steps, key=lambda one: one[0]):
            if isinstance(step, Change):
                key = step.place or step.place_word
                if step.after:
                    located[step.individual] = (key, step.place_word)
                elif located.get(step.individual, (None,))[0] == key:
                    located.pop(step.individual)
            elif somewhere(step):
                located[step.individual] = (
                    step.bound[0] if step.bound else step.object, step.object)
            elif (step.object.startswith("either ")
                  or located.get(step.individual, (None, None))[1]
                  == step.object[3:]):
                located.pop(step.individual, None)
            found = (located.get(individual) if direct
                     else resolved(individual))
            key = found[0] if found else None
            if history and history[-1][0] == key:
                continue
            if found:
                history.append((*found, step))
            elif history:
                # Let go of, somewhere the story does not say: nowhere known.
                history.append((None, "", step))
        return history

    def told_elsewhere(self, episode_id: str) -> set[tuple]:
        """(individual, relation, object) of every fact in time whose latest
        telling is in another episode: what T3 keeps from answering a
        question about this one."""
        latest: dict[tuple, str] = {}
        for record in self.records:
            latest[(record.individual, record.relation, record.object)] = \
                record.episode
        return {key for key, episode in latest.items()
                if episode != episode_id}

    # -- the page ----------------------------------------------------------
    def as_dict(self) -> dict:
        episodes = []
        for episode in sorted(self.episodes,
                              key=lambda one: (one.sort_key, one.turn)):
            place = self.positions(episode.id)
            episodes.append({
                **episode.as_dict(), "current": episode.id == self.current,
                "occurrences": [
                    {**one.as_dict(), "position": place.get(one.id, 0.0)}
                    for one in sorted(self.in_episode(episode.id),
                                      key=lambda one: (place.get(one.id, 0.0),
                                                       one.seq))],
                "states": [record.as_dict() for record in self.records
                           if record.episode == episode.id],
                "changes": [change.as_dict() for change in self.changes
                            if (self.occurrence(change.occurrence)
                                or Occurrence("", "", [], None, "", "", "",
                                              "")).episode == episode.id]})
        return {"current": self.current, "episodes": episodes,
                "order": [{"first": first, "second": second, "why": why,
                           "how": ("during" if frozenset((first, second))
                                   in self.during else "before")}
                          for (first, second), why in self.reasons.items()],
                "nodes": self.trie.node_count,
                "growth": self.growth[-1] if self.growth else None}
