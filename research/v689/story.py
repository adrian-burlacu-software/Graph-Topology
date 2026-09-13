"""What a conversation does with time: statements placed, questions answered.

`timeline.py` is the projection -- episodes, occurrences, their order and what
holds -- `tense.py` reads when, and `change.py` is T4's VerbNet. This is the
domain service a session hands time to. It decides, and records what it
decided through the timeline's commands, so everything here is replayed from
the stream and nothing is asked twice.

    said                               recorded
    yesterday the dog chased the cat   episode yesterday; o1 chase, the dog,
                                       the cat
    then it slept                      o2 sleep; o1 before o2 (T1)
    the cat broke the vase             o3 break; T4: the vase broken after o3
    it is empty now                    has_property empty, in now (T3)

    asked                              answered by
    did the dog chase the cat          T5: an occurrence under is_a chase
    did the dog move                   T5: chasing is a kind of moving
    did the dog bark, nothing told     T6: not told; that dogs bark is a
                                       tendency of the kind
    did the dog bark before it slept   T2: a walk along before
    is the vase broken                 T4, then T3
    was it full this morning           T3, in that episode
    when did the dog chase the cat     T5, and where it falls
    what happened after it barked      T2, from that occurrence
    how many times did it bark         T5: the occurrences, counted
"""
from __future__ import annotations

from . import change as changes
from .discourse import OBJECT_WEIGHT
from .episodic import DID_NOT
from .reading import ARTICLES, SEQUENCE, article, read, words
from .tense import When, occurs, tense_of
from .timeline import (IN_TIME, LATER, NOW, THEN, Change, Episode, Holding,
                       Occurrence, Record)

#: What ends a verb's object and begins where it happened.
PREPOSITIONS = changes.DESTINATION | changes.SOURCE | {"with", "by", "of"}

#: Introducers in the past tense: `there was a pig`, `i saw a dog`.
PAST_INTRODUCERS = frozenset({"was", "were", "saw", "met", "found", "got",
                              "had"})


def _base(relation: str) -> tuple[str, bool]:
    """(the relation, denied): `not_has_property` -> (has_property, True)."""
    return ((relation[4:], True) if relation.startswith("not_")
            else (relation, False))


class Story:
    """Time, for one session: T1 to T6 over its timeline."""

    def __init__(self, session, timeline) -> None:
        self.session = session
        self.timeline = timeline
        #: (episode, tense, aspect) of the statement being told, for the
        #: occurrence `narrate` records after the fact is told
        self._pending: tuple | None = None

    # -- shortcuts ---------------------------------------------------------
    @property
    def asker(self):
        return self.session.asker

    @property
    def memory(self):
        return self.session.memory

    @property
    def discourse(self):
        return self.session.discourse

    def when(self) -> When:
        return getattr(self.session, "_when", None) or When()

    def phrase(self, episode: Episode | None) -> str:
        """How an episode is said in a reply."""
        if episode is None:
            return "at no time told"
        if episode.key == THEN:
            return "back then"
        return episode.label

    def _several(self) -> bool:
        return len(self.timeline.episodes) > 1

    # -- a clause's time ---------------------------------------------------
    def clause_time(self, aux, rest) -> tuple[str, str]:
        rest = list(rest)
        if not rest:
            return tense_of(aux, rest)
        prefix = ["it"] + ([aux] if aux else [])
        tags = self.asker.tags(prefix + rest)
        return tense_of(aux, rest, tags[len(prefix):] if tags else None)

    def verb_of(self, aux, rest, aspect: str) -> str:
        """The verb a clause is about: `was flying` -> fly, `had eaten` ->
        eat, `chased` -> chase."""
        if not rest:
            return ""
        word = rest[0]
        if aspect == "progressive":
            return self.asker.progressive(word) or self.asker.verb(word)
        if aspect == "perfect":
            return self.asker.participle(word) or self.asker.verb(word)
        return self.asker.verb(word)

    # -- episodes ----------------------------------------------------------
    def episode_for(self, tense: str) -> Episode:
        """The episode a statement is in: the one it names; the present for
        the present tense; for the past, the past the story is already in,
        or `then`."""
        timeline, frame = self.timeline, self.when().frame
        if frame is not None:
            kind = ("past" if frame.rank < 0 else
                    "present" if frame.rank == 0 else "future")
            return timeline.enter(frame.key, frame.label, frame.rank, kind)
        if tense == "future":
            return timeline.enter(LATER, "later", None, "future")
        if tense == "past":
            current = timeline.episode(timeline.current)
            # The past the story is in; and `today`, said outright, has
            # a past of its own: `today there is a cat. it slept.`
            if current is not None and (current.tense == "past"
                                        or current.label == "today"):
                return timeline.enter(current.key, current.label,
                                      current.rank, current.tense)
            return timeline.enter(THEN, "then", None, "past")
        return timeline.enter(NOW, "now", 0.0, "present")

    def asked_episode(self, tense: str) -> tuple[Episode | None, str]:
        """(episode, why none) a question is about: the one it names, or the
        one its tense points at."""
        frame = self.when().frame
        if frame is not None:
            found = self.timeline.by_key(frame.key)
            return found, ("" if found else
                           f"nothing was told of {frame.label}")
        return self.timeline.latest(tense), ""

    def introduced(self, reading) -> None:
        """`yesterday there was a dog` puts the story at yesterday. With no
        time named an introduction places nothing: that there is a dog is not
        a state of a moment, and `there is a dog. it barked.` is one story."""
        if self.when().frame is None:
            return
        said = words(reading.said)[:4]
        self.episode_for("past" if PAST_INTRODUCERS & set(said)
                         else "present")

    # -- statements --------------------------------------------------------
    def before_telling(self, relation: str | None, mode: str, aux, rest,
                       holds: bool) -> dict | None:
        """Where in the story a told fact holds, as `tell` records it.

        A fact in time -- a quality, a place, not doing -- is told with its
        episode and the occurrence the story is at. A doing is placed by its
        occurrence instead (`narrate`). What an individual is, is called, owns
        or can do is in no episode, and a habit (`it barks`) is not an
        occurrence.
        """
        self._pending = None
        doing = relation in ("capable_of", DID_NOT) and mode == "does"
        if relation is None or (relation not in IN_TIME and not doing):
            return None
        tense, aspect = self.clause_time(aux, rest)
        if doing and not occurs(tense, aspect):
            return None
        episode = self.episode_for(tense)
        self._pending = (episode, tense, aspect)
        if relation not in IN_TIME:
            return None
        return {"episode": episode.id,
                "after": self.timeline.reference.get(episode.id),
                "tense": tense, "aspect": aspect}

    def narrate(self, reading, referent, other, relation: str, obj: str,
                holds: bool, said: str) -> str:
        """T1 and T4 for a told doing: the occurrence, where it falls, and
        what it changes. Returns what the reply adds."""
        pending, self._pending = self._pending, None
        if (pending is None or relation != "capable_of" or not holds
                or reading is None):
            return ""
        episode, tense, aspect = pending
        rest = list(reading.rest)
        verb = self.verb_of(reading.aux, rest, aspect)
        if not verb:
            return ""
        parts = self._participants(reading, referent, other)
        when = self.when()
        anchor = self.find_clause(when.anchor) if when.anchor else None
        occurrence = self.timeline.narrate(
            {"verb": verb, "kinds": self.asker.verb_kinds(verb),
             "subject": referent.id, "episode": episode.id, "tense": tense,
             "aspect": aspect, "said": _said(when.main or said),
             "predicate": obj,
             "again": when.again, **parts["data"]},
            when.link, anchor.id if anchor else None, when.relation)
        notes = [self.placed(occurrence)]
        who = {"subject": referent.id, "object": parts["data"]["object"]}
        for effect in changes.effects(verb, parts["has_object"],
                                      parts["data"]["preposition"],
                                      parts["clause"],
                                      self.asker.verb_senses(verb)):
            individual = who.get(effect.position)
            if not individual:
                continue
            data = parts["data"]
            if effect.kind == "location":
                if not (data["place"] or data["place_word"]):
                    continue
                change = self.timeline.change(
                    occurrence=occurrence.id, individual=individual,
                    kind="location", word="", after=effect.after,
                    before=effect.before, source=effect.source,
                    place=data["place"], place_word=data["place_word"])
            else:
                if effect.kind == "state" and effect.word == verb and not (
                        changes.adjective(self.participle(
                            verb, rest[0] if rest else verb))
                        or changes.adjective(verb)):
                    continue
                change = self.timeline.change(
                    occurrence=occurrence.id, individual=individual,
                    kind=effect.kind, word=effect.word, after=effect.after,
                    before=effect.before, source=effect.source)
            notes.append(self.changed(change, occurrence, rest))
        return "; ".join(note for note in notes if note)

    def _participants(self, reading, referent, other) -> dict:
        """Who and what took part, by position: the object after the verb,
        the place after a preposition, or a clause (`started barking`)."""
        rest = list(reading.rest)
        tags = None
        if len(rest) > 1:
            prefix = ["it"] + ([reading.aux] if reading.aux else [])
            found = self.asker.tags(prefix + rest)
            tags = found[len(prefix):] if found else None
        data = {"object": None, "object_word": "", "place": None,
                "place_word": "", "preposition": ""}
        clause = ""
        at = next((index for index in range(1, len(rest) - 1)
                   if rest[index] in PREPOSITIONS), None)
        if at is not None and rest[at] in ("with", "by", "of"):
            at = None
        middle = rest[1:at] if at is not None else rest[1:]
        if at is not None:
            start = at + 1 + (1 if rest[at + 1:at + 2] == ["of"] else 0)
            data["preposition"] = rest[at]
            data["place_word"] = " ".join(word for word in rest[start:]
                                          if word not in ARTICLES)
            if other is not None and reading.obj_at >= start:
                data["place"] = other.id
                data["place_word"] = other.kind
        if len(rest) > 1 and not (other is not None and at is None):
            if rest[1] == "to" and len(rest) > 2 and at is None:
                clause, middle = self.asker.verb(rest[2]), []
            elif (rest[1].endswith("ing") and at is None
                  and self.asker.progressive(rest[1])):
                clause, middle = self.asker.progressive(rest[1]), []
        if other is not None and at is None:
            data["object"], data["object_word"] = other.id, other.kind
        elif middle:
            nouns = (tags[1:1 + len(middle)] if tags else None)
            if nouns is None or any(tag.startswith(("NN", "PRP", "DT"))
                                    for tag in nouns):
                data["object_word"] = " ".join(word for word in middle
                                               if word not in ARTICLES)
                found = self._existing(middle, exclude={referent.id})
                if found is not None:
                    data["object"], data["object_word"] = found.id, found.kind
        return {"data": data, "clause": clause,
                "has_object": bool(data["object"] or data["object_word"])}

    def _existing(self, phrase: list[str], exclude=frozenset()):
        """The individual a phrase names among those here, never a new one."""
        from .reading import read_mention
        mention = read_mention(list(phrase), 0, self.session.lexicon(),
                               final_ok=True, names=self.discourse.names())
        if mention is None or mention.end != len(phrase):
            return None
        return self._resolved(mention, exclude)

    def _resolved(self, mention, exclude=frozenset()):
        if mention is None or mention.form in ("indefinite", "another",
                                               "kind"):
            return None
        if (mention.form in ("definite", "demonstrative", "possessive",
                             "ordinal", "other")
                and mention.kind and not self.session._individuals(
                    mention.kind)):
            return None
        return self.discourse.resolve(mention, exclude=set(exclude),
                                      weight=OBJECT_WEIGHT).referent

    def placed(self, occurrence: Occurrence) -> str:
        """Where T1 put an occurrence, as the reply says it."""
        episode = self.timeline.episode(occurrence.episode)
        text = (f"it {'will happen' if occurrence.tense == 'future' else 'happened'}"
                f" {self.phrase(episode)}")
        for (first, second), why in self.timeline.reasons.items():
            if occurrence.id not in (first, second):
                continue
            other = self.timeline.occurrence(second if first == occurrence.id
                                             else first)
            if other is None:
                continue
            how = ("during" if frozenset((first, second))
                   in self.timeline.during else
                   "before" if first == occurrence.id else "after")
            text += f", {how} “{other.said}” (T1: {why})"
        return text

    def participle(self, verb: str, said: str) -> str:
        """The form a state brought about by a verb is said in: `broke` ->
        broken, `closed` -> closed. Found by asking which spelling reads back
        as the verb's participle, since nothing here inflects."""
        for candidate in (said + "n", said, verb + "ed", verb + "d",
                          verb + "en"):
            if self.asker.participle(candidate) == verb:
                return candidate
        return said

    def changed(self, change: Change, occurrence: Occurrence,
                rest: list[str]) -> str:
        """What T4 recorded, as the reply says it."""
        who = self.discourse.by_id(change.individual)
        described = self.discourse.describe(who) if who else "it"
        if change.kind == "location":
            place = self.discourse.by_id(change.place) if change.place else None
            where = (self.discourse.describe(place) if place is not None
                     else f"{article(change.place_word)} {change.place_word}")
            # Where it ends up, said as being there: onto the table, on it.
            preposition = {"onto": "on", "into": "in", "to": "at",
                           "upon": "on", "towards": "toward"}.get(
                occurrence.preposition, occurrence.preposition or "at")
            if change.after:
                return (f"T4 ({change.source}): {described} is {preposition} "
                        f"{where} after it, and was not before")
            return (f"T4 ({change.source}): {described} was at {where} "
                    f"before it, and is not after")
        if change.kind == "activity":
            doing = rest[1] if len(rest) > 1 else change.word
            return (f"T4 ({change.source}): {described} is "
                    f"{'' if change.after else 'not '}{doing} after it")
        word = (self.participle(change.word, rest[0] if rest else "")
                if change.word == occurrence.verb else change.word)
        text = (f"T4 ({change.source}): {described} is "
                f"{'' if change.after else 'not '}{word} after it")
        if change.before is not None:
            text += (f", and was {'' if change.before else 'not '}before")
        return text

    def told_note(self, when: dict | None) -> str:
        """`; told of yesterday (T3)`, for a state told of a named time or
        when the story has more than one."""
        episode = self.timeline.episode((when or {}).get("episode"))
        if episode is None or (not self._several()
                               and episode.key in (THEN, NOW)):
            return ""
        return f"; told of {self.phrase(episode)} (T3)"

    def verb_words(self, occurrence: Occurrence) -> list[str]:
        """An occurrence's words from its verb on, as said: `broke the
        vase`, `stopped barking`."""
        said = words(occurrence.said)
        at = next((index for index, word in enumerate(said)
                   if self.asker.verb(word) == occurrence.verb), None)
        return said[at:] if at is not None else occurrence.predicate.split()

    def same_time(self, node: str, doing: str, place: str) -> bool:
        """E2 within an episode: was the doing told of the time it was being
        carried? Unplaced facts are taken to be."""
        doings = {one.episode for one in self.timeline.occurrences
                  if one.subject == node and one.predicate == doing}
        places = {record.episode for record in self.timeline.records
                  if record.individual == node
                  and record.relation == "at_location"
                  and record.object == place}
        return not doings or not places or bool(doings & places)

    # -- finding occurrences -----------------------------------------------
    def find_clause(self, text: str) -> Occurrence | None:
        """The occurrence a clause names: `the dog chased the cat`."""
        reading = read(text, self.session.lexicon(), self.discourse.names(),
                       anchored=False)
        return self.find(reading)

    def find(self, reading) -> Occurrence | None:
        if reading.mention is None or not reading.rest:
            return None
        referent = self._resolved(reading.mention)
        if referent is None:
            return None
        _, aspect = self.clause_time(reading.aux, reading.rest)
        verb = self.verb_of(reading.aux, reading.rest, aspect)
        wanted = {f"subject {referent.id}", f"is_a {verb}"}
        if reading.obj is not None:
            other = self._resolved(reading.obj, exclude={referent.id})
            if other is not None:
                wanted.add(self._slot(reading, other))
        found = self.timeline.identify(wanted)
        return found[-1] if found else None

    def _slot(self, reading, other) -> str:
        """`object r2`, or `place r2` when a preposition comes before it."""
        at = reading.obj_at
        before = reading.rest[at - 1] if at and at > 0 else ""
        return (f"place {other.id}" if before in PREPOSITIONS
                else f"object {other.id}")

    def _wanted(self, reading, referent, other, verb: str) -> set:
        wanted = {f"is_a {verb}"}
        if referent is not None:
            wanted.add(f"subject {referent.id}")
        if other is not None:
            wanted.add(self._slot(reading, other))
        return wanted

    def situate(self, occurrence: Occurrence) -> str:
        """`yesterday — “the dog chased the cat”, after “it barked”`."""
        episode = self.timeline.episode(occurrence.episode)
        ones = [one for one in self.timeline.story()
                if one.episode == occurrence.episode]
        index = next((at for at, one in enumerate(ones)
                      if one.id == occurrence.id), 0)
        text = f"{self.phrase(episode)} — “{occurrence.said}”"
        before = [one for one in ones[:index]
                  if self.timeline.relation(one.id, occurrence.id) == "before"]
        after = [one for one in ones[index + 1:]
                 if self.timeline.relation(occurrence.id, one.id) == "before"]
        if before:
            text += f", after “{before[-1].said}”"
        if after:
            text += f"{' and' if before else ','} before “{after[0].said}”"
        return text

    # -- yes or no ---------------------------------------------------------
    def ask(self, reading, referent, other, relation: str, target: str,
            turn) -> bool:
        """A yes or no about one individual that time decides. False when it
        is not about time, or time settles nothing and the walk should."""
        when = self.when()
        tense, aspect = self.clause_time(reading.aux, reading.rest)
        if relation == "capable_of" and when.anchor:
            return self._ask_order(reading, referent, other, turn, aspect)
        if relation == "capable_of" and (reading.aux in ("did", "will")
                                         or aspect == "progressive"
                                         or when.frame is not None):
            return self._ask_doing(reading, referent, other, turn, tense,
                                   aspect)
        base, _ = _base(relation)
        if base in IN_TIME and relation != DID_NOT:
            return self._ask_state(reading, referent, other, relation,
                                   target, turn, tense)
        return False

    def _ask_order(self, reading, referent, other, turn, aspect) -> bool:
        """T2: `did the dog bark before it slept`."""
        when = self.when()
        verb = self.verb_of(reading.aux, reading.rest, aspect)
        described = self.discourse.describe(referent)
        mains = self.timeline.identify(self._wanted(reading, referent, other,
                                                    verb))
        anchor = self.find_clause(when.anchor)
        if anchor is None:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"nothing was told of “{when.anchor}”, so "
                                   f"there is nothing to place it against"}
            return True
        mains = [one for one in mains if one.id != anchor.id]
        if not mains:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"not told — nothing was said of "
                                   f"{described} that it would "
                                   f"{' '.join(reading.rest)} (T6)"}
            return True
        asked = when.relation
        found = [(one, self.timeline.relation(one.id, anchor.id))
                 for one in mains]
        yes = [one for one, how in found if how == asked]
        no = [(one, how) for one, how in found if how and how != asked]
        if yes:
            turn.answer = {"outcome": "verified", "source": "told",
                           "text": f"yes — “{yes[0].said}” came {asked} "
                                   f"“{anchor.said}”: "
                                   f"{self.explain(yes[0], anchor)} (T2)"}
        elif no:
            one, how = no[0]
            turn.answer = {"outcome": "denied", "source": "told",
                           "text": f"no — “{one.said}” came {how} "
                                   f"“{anchor.said}”: "
                                   f"{self.explain(one, anchor)} (T2)"}
        else:
            turn.answer = {"outcome": "unknown", "source": "told",
                           "text": f"not told — “{mains[0].said}” and "
                                   f"“{anchor.said}” were both told, and "
                                   f"nothing said which came first (T2: "
                                   f"absent, not false)"}
        return True

    def explain(self, one: Occurrence, other: Occurrence) -> str:
        """Why two occurrences are in the order T2 found."""
        first, second = self.timeline.episode(one.episode), \
            self.timeline.episode(other.episode)
        if first is not None and second is not None and first.id != second.id:
            return (f"{self.phrase(first)} is "
                    f"{self.timeline._episode_order(first, second)} "
                    f"{self.phrase(second)}")
        path = self.timeline.path(one.id, other.id) or \
            self.timeline.path(other.id, one.id)
        if path:
            steps = [self.timeline.occurrence(node) for node in path]
            return " → ".join(f"“{step.said}”" for step in steps if step)
        if frozenset((one.id, other.id)) in self.timeline.during:
            return self.timeline.reasons.get(
                (one.id, other.id), self.timeline.reasons.get(
                    (other.id, one.id), "told as happening together"))
        return "as told"

    def _ask_doing(self, reading, referent, other, turn, tense: str,
                   aspect: str) -> bool:
        """T5: `did the dog chase the cat`, `is the pig flying`."""
        timeline, when = self.timeline, self.when()
        verb = self.verb_of(reading.aux, reading.rest, aspect)
        described = self.discourse.describe(referent)
        if when.frame is not None:
            episode = timeline.by_key(when.frame.key)
            if episode is None:
                told = timeline.identify(self._wanted(reading, referent, other,
                                                      verb))
                turn.answer = {
                    "outcome": "unknown", "source": "conversation",
                    "text": f"not told — nothing was told of "
                            f"{when.frame.label}" + (
                                f"; you told me “{told[0].said}”, "
                                f"{self.phrase(timeline.episode(told[0].episode))}"
                                if told else "")}
                return True
            episodes = [episode]
        elif aspect == "progressive":
            episodes = [timeline.latest(tense)] if timeline.episodes else []
        elif tense == "future":
            episodes = [one for one in timeline.episodes
                        if one.tense == "future"]
        else:
            # `did`: anything that has happened, and never what is still to.
            episodes = [one for one in timeline.episodes
                        if one.tense != "future"]
        ids = {one.id for one in episodes if one is not None}
        if aspect == "progressive" and episodes and episodes[0] is not None:
            held = timeline.holding(referent.id, self._judge_doing(verb),
                                    episodes[0].id)
            if held.value is not None:
                self._held(turn, held, described, reading)
                return True
        anywhere = timeline.identify(self._wanted(reading, referent, other,
                                                  verb))
        found = [one for one in anywhere if one.episode in ids]
        if anywhere and not found:
            asked = (self.phrase(episodes[0]) if episodes and episodes[0]
                     is not None and len(episodes) == 1 else
                     "anything that has happened" if tense == "past"
                     else "that time")
            turn.answer = {"outcome": "unknown", "source": "told",
                           "text": f"not told of {asked} — you told me "
                                   f"“{anywhere[0].said}”, "
                                   f"{self.phrase(timeline.episode(anywhere[0].episode))}"
                                   f" (T3: a doing belongs to the time it was "
                                   f"told of)"}
            return True
        if found:
            one = found[0]
            text = f"yes — you told me “{one.said}”"
            if self._several():
                text += f", {self.phrase(timeline.episode(one.episode))}"
            if one.verb != verb:
                text += (f" (T5: {one.verb} is a kind of {verb}, as R1 walks "
                         f"a taxonomy)")
            else:
                text += " (T5)"
            if any(withdrawn.node == referent.id
                   and withdrawn.object == one.predicate
                   for withdrawn in self.memory.withdrawn):
                text += ("; it was carried while it did, so the doing was "
                         "what carried it (E2), and says nothing of what it "
                         "can do")
            turn.answer = {"outcome": "verified", "source": "told",
                           "text": text}
            return True
        for record in timeline.records:
            if (record.individual == referent.id and record.relation == DID_NOT
                    and record.episode in ids and record.object.split()
                    and self.asker.verb(record.object.split()[0]) == verb):
                turn.answer = {"outcome": "denied", "source": "told",
                               "text": f"no — you told me “{record.said}”"}
                return True
        same = timeline.identify({f"subject {referent.id}", f"is_a {verb}"})
        if same and other is not None:
            about = [self.discourse.by_id(one.object or one.place)
                     for one in same]
            named = " and ".join(dict.fromkeys(
                self.discourse.describe(one) for one in about if one))
            turn.answer = {"outcome": "unknown", "source": "told",
                           "text": f"not told — you told me “{same[0].said}”"
                                   + (f", and that was about {named}, not "
                                      f"{self.discourse.describe(other)}"
                                      if named else "")}
            return True
        if same:
            elsewhere = timeline.episode(same[0].episode)
            turn.answer = {"outcome": "unknown", "source": "told",
                           "text": f"not told of "
                                   f"{self.phrase(episodes[0]) if episodes and episodes[0] else 'that time'}"
                                   f" — you told me “{same[0].said}”, "
                                   f"{self.phrase(elsewhere)} (T3: a doing "
                                   f"belongs to the time it was told of)"}
            return True
        return False

    def inherited(self, reading, referent, question: str, turn) -> bool:
        """T6: a question about what one of them did, that nothing told
        answers, is not answered by what the kind does."""
        tense, aspect = self.clause_time(reading.aux, reading.rest)
        if not (reading.aux in ("did", "will") or aspect == "progressive"
                or self.when().frame is not None):
            return False
        described = self.discourse.describe(referent)
        outcome, trust = self.session._kind_answer(question, turn)
        doing = {"future": "will do"}.get(
            tense, "was doing" if aspect == "progressive" else "did")
        kind = f"{article(referent.kind)} {referent.kind}"
        text = (f"not told — nothing you said of {described} answers "
                f"“{_said(reading.said)}”. T6: what {kind} does is a "
                f"tendency of the kind, and that one of them {doing} it is "
                f"an occurrence, which was never told; for {kind} in general "
                f"v688 answers “{question}” {outcome}")
        if trust:
            text += f" ({trust})"
        turn.answer = {"outcome": "unknown", "source": "tendency",
                       "text": text}
        return True

    def _judge_doing(self, verb: str):
        def judge(kind: str, relation: str, word: str) -> int:
            if kind == "doing":
                return 1 if word == verb or verb in self.asker.verb_kinds(
                    word) else 0
            if kind == "change" and relation == "activity":
                return 1 if word == verb else 0
            if kind == "record" and relation == DID_NOT and word.split():
                return -1 if self.asker.verb(word.split()[0]) == verb else 0
            return 0
        return judge

    def _judge_state(self, relation: str, target: str, other):
        base, denied = _base(relation)
        wanted = self.session._predicate(target)
        head = wanted[0] if len(wanted) == 1 else ""
        verb = self.asker.participle(target.split()[0]) if target else None

        def judge(kind: str, told: str, word: str) -> int:
            if kind == "record":
                told_base, told_denied = _base(told)
                if told_base != base:
                    return 0
                sign = -1 if told_denied != denied else 1
                text = word[3:] if word.startswith("no ") else word
                have = self.session._predicate(text)
                if have == wanted:
                    return sign
                if (base != "at_location" and head and len(have) == 1
                        and self.session._opposite(have[0], head)):
                    return -sign
                return 0
            if kind == "change":
                if told == "location":
                    if base != "at_location":
                        return 0
                    if other is not None and word == other.id:
                        return 1
                    return 1 if self.session._predicate(word) == wanted else 0
                if told != "state" or base != "has_property" or not head:
                    return 0
                sign = -1 if denied else 1
                if word in (head, verb):
                    return sign
                if self.session._opposite(word, head) or (
                        verb and self.session._opposite(word, verb)):
                    return -sign
            return 0
        return judge

    def _ask_state(self, reading, referent, other, relation: str,
                   target: str, turn, tense: str) -> bool:
        """T3 and T4: `is the vase broken`, `was it full this morning`,
        `was the door open before i closed it`."""
        timeline, when = self.timeline, self.when()
        changed = any(change.individual == referent.id
                      for change in timeline.changes)
        if not (when.frame or when.anchor or self._several() or changed):
            return False
        at, side = None, "end"
        if when.anchor:
            anchor = self.find_clause(when.anchor)
            if anchor is None:
                turn.answer = {"outcome": "unknown", "source": "conversation",
                               "text": f"nothing was told of “{when.anchor}”"}
                return True
            episode, at, side = (timeline.episode(anchor.episode), anchor.id,
                                 when.relation)
        else:
            episode, missing = self.asked_episode(tense)
            if episode is None:
                if not missing:
                    return False
                judge = self._judge_state(relation, target, other)
                told = []
                for one in timeline.episodes:
                    held = timeline.holding(referent.id, judge, one.id,
                                            _nested=True)
                    if held.value is not None:
                        told.append(f"{self.phrase(one)}, "
                                    f"{self.quote(held.basis)}")
                text = f"not told — {missing}"
                if told:
                    text = (f"not told of {when.frame.label} — "
                            + "; ".join(told) + ". T3: a state belongs to "
                                                "the time it was told of")
                turn.answer = {"outcome": "unknown", "source": "conversation",
                               "text": text}
                return True
        described = self.discourse.describe(referent)
        held = timeline.holding(referent.id,
                                self._judge_state(relation, target, other),
                                episode.id, at, side)
        if held.value is not None:
            self._held(turn, held, described, reading, target, at, side)
            return True
        if held.elsewhere:
            told = "; ".join(f"{self.phrase(one)}, {self.quote(basis)}"
                             for one, _, basis in held.elsewhere)
            turn.answer = {
                "outcome": "unknown", "source": "told",
                "text": (f"not told of {self.phrase(episode)} — {told}. T3: "
                         f"what {described} is like and where it is belong "
                         f"to the time they were told of")}
            return True
        return False

    def quote(self, basis) -> str:
        if isinstance(basis, Record):
            return f"you told me “{basis.said}”"
        if isinstance(basis, Change):
            occurrence = self.timeline.occurrence(basis.occurrence)
            return (f"“{occurrence.said}” changed it (T4, {basis.source})"
                    if occurrence else f"T4 ({basis.source})")
        if isinstance(basis, Occurrence):
            return f"you told me “{basis.said}”"
        return ""

    def _held(self, turn, held: Holding, described: str, reading,
              target: str = "", at=None, side: str = "end") -> None:
        word = "yes" if held.value else "no"
        basis = held.basis
        if isinstance(basis, Change):
            occurrence = self.timeline.occurrence(basis.occurrence)
            said = occurrence.said if occurrence else ""
            when = "before" if at is not None and side == "before" else "after"
            text = f"{word} — {when} “{said}”"
            if occurrence is not None:
                text += ": " + self.changed(basis, occurrence,
                                            self.verb_words(occurrence))
            if at is None or at != basis.occurrence:
                text += "; nothing told since changes it back (T3)"
        elif isinstance(basis, Record):
            text = f"{word} — you told me “{basis.said}”"
            text += (" (T3: it holds within the time it was told of)"
                     if self._several() or at is not None else "")
        else:
            text = f"{word} — you told me “{basis.said}” (T5)"
        if self._several() and held.episode is not None:
            text += f"; told of {self.phrase(held.episode)}"
        turn.answer = {"outcome": "verified" if held.value else "denied",
                       "source": "told", "text": text}

    def hidden(self, reading) -> frozenset:
        """T3 for the walk: facts in time told of another episode than the
        one a question is about, which the walk may not read."""
        if not self._several():
            return frozenset()
        tense, aspect = self.clause_time(reading.aux, reading.rest)
        # `does it fly`, `can it fly`: a habit or an ability is asked of no
        # moment, and what was told of any moment bears on it.
        if aspect not in ("state", "progressive") and self.when().frame is None:
            return frozenset()
        episode, _ = self.asked_episode(tense)
        if episode is None:
            return frozenset()
        return frozenset(self.timeline.told_elsewhere(episode.id))

    # -- wh-questions ------------------------------------------------------
    def _verb_asked(self, reading) -> str:
        _, aspect = self.clause_time(reading.aux, reading.rest)
        return self.verb_of(reading.aux, reading.rest, aspect)

    def when_asked(self, reading, turn) -> None:
        """T5: `when did the dog chase the cat`."""
        referent = self.session._here(reading, turn)
        if referent is None:
            return
        rest, other = self.session._object_here(reading, turn)
        if rest is None:
            return
        verb = self._verb_asked(reading)
        found = self.timeline.identify(self._wanted(reading, referent, other,
                                                    verb))
        described = self.discourse.describe(referent)
        if not found:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"never, as far as I was told — nothing was "
                                   f"said of {described} that it would "
                                   f"{' '.join(reading.rest)} (T6)"}
            return
        turn.answer = {"outcome": "retrieved", "source": "told",
                       "text": "; ".join(self.situate(one) for one in found)}

    def how_many_times(self, reading, turn) -> None:
        referent = self.session._here(reading, turn)
        if referent is None:
            return
        rest, other = self.session._object_here(reading, turn)
        if rest is None:
            return
        verb = self._verb_asked(reading)
        wanted = self._wanted(reading, referent, other, verb)
        frame = self.when().frame
        if frame is not None:
            episode = self.timeline.by_key(frame.key)
            if episode is None:
                turn.answer = {"outcome": "unknown", "source": "conversation",
                               "text": f"nothing was told of {frame.label}"}
                return
            wanted.add(f"episode {episode.id}")
        found = self.timeline.identify(wanted)
        described = self.discourse.describe(referent)
        if not found:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"not once, as far as I was told — nothing "
                                   f"was said of {described} that it would "
                                   f"{' '.join(reading.rest)} (T6)"}
            return
        count = len(found)
        turn.answer = {"outcome": "retrieved", "source": "told",
                       "text": f"{count} time{'s' if count != 1 else ''} — "
                               + "; ".join(f"“{one.said}”" for one in found)
                               + " (T5: the occurrences under "
                                 f"is_a {verb}, counted)"}

    def doing(self, reading, turn) -> None:
        """`what was the dog doing`, `what was it doing when the cat ate`."""
        referent = self.session._here(reading, turn)
        if referent is None:
            return
        timeline, when = self.timeline, self.when()
        tense = "past" if reading.aux in ("was", "were") else "present"
        described = self.discourse.describe(referent)
        anchor = self.find_clause(when.anchor) if when.anchor else None
        if when.anchor and anchor is None:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"nothing was told of “{when.anchor}”"}
            return
        episode = (timeline.episode(anchor.episode) if anchor else
                   self.asked_episode(tense)[0])
        ones = [one for one in (timeline.in_episode(episode.id)
                                if episode else [])
                if one.subject == referent.id
                and (anchor is None or one.id != anchor.id)]
        if anchor is not None:
            place = timeline.positions(episode.id)
            ones = [one for one in ones
                    if timeline.relation(one.id, anchor.id) == "during"
                    or place.get(one.id) == place.get(anchor.id)]
        going = [one for one in ones if one.aspect == "progressive"]
        started = [change for change in timeline.changes
                   if change.individual == referent.id
                   and change.kind == "activity"]
        chosen = going or ones
        if not chosen and not started:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"nothing was told of what {described} "
                                   f"was doing {self.phrase(episode)}"}
            return
        text = "; ".join(f"“{one.said}”" for one in chosen)
        if started and not chosen:
            text = "; ".join(self.quote(change) for change in started)
        if anchor is not None:
            text += f" — while “{anchor.said}” (T2)"
        turn.answer = {"outcome": "retrieved", "source": "told",
                       "text": f"{described}: {text}"}

    def happened(self, reading, turn) -> bool:
        """What happened, in story order: in an episode, around an
        occurrence, to one individual, or the one at a place in the order.
        False where time adds nothing to what the session answers."""
        timeline, when = self.timeline, self.when()
        rest = list(reading.rest)
        if rest[:1] == ["told"] or not timeline.occurrences:
            return False
        referent = None
        if reading.mention is not None:
            referent = self.session._here(reading, turn)
            if referent is None:
                return True
        ones = timeline.story()
        if rest[:1] == ["future"]:
            ones = [one for one in ones if one.tense == "future"]
        plain = (referent is None and not when.frame and not when.anchor
                 and rest[:1] != ["future"]
                 and not any(word in SEQUENCE for word in rest))
        if plain:
            return False            # the session's list, and `order_note`
        if referent is not None:
            ones = [one for one in ones
                    if referent.id in ((one.subject, one.object, one.place)
                                       if rest[:1] == ["to"]
                                       else (one.subject,))]
        where = ""
        if when.frame is not None:
            episode = timeline.by_key(when.frame.key)
            if episode is None:
                turn.answer = {"outcome": "unknown", "source": "conversation",
                               "text": f"nothing was told of "
                                       f"{when.frame.label}"}
                return True
            ones = [one for one in ones if one.episode == episode.id]
            where = f"{self.phrase(episode)}: "
        if when.anchor:
            anchor = self.find_clause(when.anchor)
            if anchor is None:
                turn.answer = {"outcome": "unknown", "source": "conversation",
                               "text": f"nothing was told of “{when.anchor}”"}
                return True
            ones = [one for one in ones if one.id != anchor.id
                    and timeline.relation(one.id, anchor.id) == when.relation]
            where = f"{when.relation} “{anchor.said}”: "
        ordinal = next((SEQUENCE[word] for word in rest if word in SEQUENCE),
                       None)
        if ordinal is not None and ones:
            ones = ([ones[-1]] if ordinal == -1 else
                    ones[ordinal - 1:ordinal] if ordinal <= len(ones) else [])
            where = next(word for word in rest if word in SEQUENCE) + ": "
        if not ones:
            described = (self.discourse.describe(referent) if referent
                         else "anyone")
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"nothing was told of {described} "
                                   f"{where.rstrip(': ') or 'then'}"}
            return True
        turn.answer = {"outcome": "retrieved", "source": "told",
                       "text": where + self.telling(ones)}
        return True

    def order_note(self) -> str:
        """What story order adds to a list in the order things were told:
        nothing, unless it is another order or spans several episodes."""
        ones = self.timeline.story()
        if not ones or (ones == sorted(ones, key=lambda one: one.seq)
                        and not self._several()):
            return ""
        return "; in the order it happened (T1, T2): " + self.telling(ones)

    def telling(self, ones: list[Occurrence]) -> str:
        """Occurrences in story order, grouped by episode when they fall in
        several."""
        if len({one.episode for one in ones}) < 2:
            return "; then ".join(f"“{one.said}”" for one in ones)
        groups: dict[str, list] = {}
        for one in ones:
            groups.setdefault(one.episode, []).append(one)
        return "; ".join(
            f"{self.phrase(self.timeline.episode(episode))}: "
            + ", then ".join(f"“{one.said}”" for one in group)
            for episode, group in groups.items())

    def who(self, reading, turn) -> bool:
        """`who chased the cat yesterday`: the subjects of the occurrences."""
        if not self.timeline.occurrences or not reading.rest:
            return False
        rest, other = self.session._object_here(reading, turn)
        if rest is None:
            return True
        verb = self._verb_asked(reading)
        wanted = self._wanted(reading, None, other, verb)
        frame = self.when().frame
        if frame is not None:
            episode = self.timeline.by_key(frame.key)
            if episode is None:
                turn.answer = {"outcome": "unknown", "source": "conversation",
                               "text": f"nothing was told of {frame.label}"}
                return True
            wanted.add(f"episode {episode.id}")
        found = self.timeline.identify(wanted)
        doers, seen = [], set()
        for one in found:
            doer = self.discourse.by_id(one.subject or "")
            if doer is not None and doer.id not in seen:
                seen.add(doer.id)
                doers.append(doer)
        if not doers:
            if frame is None:
                return False
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"nobody here was said to "
                                   f"{' '.join([verb] + list(reading.rest[1:]))}"
                                   f" {frame.label}"}
            return True
        turn.answer = {"outcome": "retrieved", "source": "told",
                       "text": self.session._names(doers) + " — you told me "
                               + "; ".join(f"“{one.said}”" for one in found)}
        return True

    def what_did(self, reading, referent, turn) -> bool:
        """`what did the dog chase yesterday`: the objects of the
        occurrences."""
        found = [one for one in self.timeline.identify(
            {f"subject {referent.id}", f"is_a {self._verb_asked(reading)}"})]
        frame = self.when().frame
        if frame is not None:
            episode = self.timeline.by_key(frame.key)
            found = [one for one in found
                     if episode is not None and one.episode == episode.id]
        if not found:
            return False
        texts = []
        for one in found:
            thing = self.discourse.by_id(one.object or one.place or "")
            what = (self.discourse.describe(thing) if thing is not None
                    else one.object_word or one.place_word or one.predicate)
            texts.append(f"{what} — you told me “{one.said}”")
        turn.answer = {"outcome": "retrieved", "source": "told",
                       "text": "; ".join(texts)}
        return True

    def where(self, reading, referent, turn) -> bool:
        """T3 and T4 for a place: `where is the key`, `where was the pig
        yesterday`."""
        timeline = self.timeline
        places: dict[str, str] = {}
        for record in timeline.records:
            if (record.individual == referent.id
                    and record.relation == "at_location"):
                key = record.bound[0] if record.bound else record.object
                places.setdefault(key, record.object)
        for change in timeline.changes:
            if change.individual == referent.id and change.kind == "location":
                places.setdefault(change.place or change.place_word,
                                  change.place_word)
        if not places:
            return False
        said = words(reading.said)
        tense = "past" if said[1:2] in (["was"], ["were"]) else "present"
        episode, missing = self.asked_episode(tense)
        described = self.discourse.describe(referent)
        if episode is None:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"not told — {missing}" if missing else
                                   f"nothing was said about where "
                                   f"{described} is"}
            return True
        found, elsewhere = [], []
        for key, word in places.items():
            def judge(kind, relation, value, key=key, word=word):
                if kind == "record" and relation == "at_location":
                    return 1 if value == word else 0
                if kind == "change" and relation == "location":
                    return 1 if value in (key, word) else 0
                return 0
            held = timeline.holding(referent.id, judge, episode.id)
            thing = self.discourse.by_id(key)
            name = (self.discourse.describe(thing) if thing is not None
                    else f"{article(word)} {word}")
            if held.value:
                found.append(f"{name} — {self.quote(held.basis)}")
            elsewhere += [f"{name}, {self.phrase(one)} — {self.quote(basis)}"
                          for one, value, basis in held.elsewhere if value]
        if found:
            text = f"{described}: " + "; ".join(found)
            if self._several():
                text += f" ({self.phrase(episode)})"
            turn.answer = {"outcome": "retrieved", "source": "told",
                           "text": text}
            return True
        if elsewhere:
            turn.answer = {"outcome": "unknown", "source": "told",
                           "text": f"not told of {self.phrase(episode)} — "
                                   + "; ".join(elsewhere)
                                   + ". T3: where it is belongs to the time "
                                     "it was told of"}
            return True
        return False


def _said(said: str) -> str:
    return (said or "").strip().rstrip(".!?")
