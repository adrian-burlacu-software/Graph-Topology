"""The state one utterance has while it is being resolved.

This is the situation buffer, and its lifetime is deliberately short: it is
born when you type something and dies when the loop settles. That scoping is
what keeps the rest of the architecture out of the way.

    Episodic memory is this log *surviving* past the utterance. Scoped to one
    utterance there is nothing to remember, so it is not needed and not built.

    Executive control is `sorted(queue, key=rank)` in `question.py`. Thirty
    lines, not a subsystem, because v687 already routes deterministically and
    there is no production conflict to resolve.

`pins.py` in v687 is the same idea with a shorter lifetime again -- which
sense a word was taken in, held for one request. Widening that to the
utterance is a lifetime change and not a new mechanism, and it is what makes
`mouse` mean one thing for the length of a question about computers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import attention
from .gap import Gap, Doubt, read_doubts, read_gap

#: A verdict that says yes, and one that says no. `INHERITED` is a yes that
#: names the ancestor it came from; `UNKNOWN` is neither, and treating it as a
#: no is the confusion the v687 audit was written to end.
POSITIVE = frozenset({"VERIFIED", "HELD", "INHERITED"})
NEGATIVE = frozenset({"CONTRADICTED", "DENIED"})

#: Answers the loop is entitled to build further work on. A doubt question is
#: a means, not a topic, and a curiosity question is a guess: following either
#: is how a run about whales ends up asking whether a goldfish is a bony fish,
#: and how `is a shark a fish` spent 37 questions on gills and slime.
DELIBERATE = frozenset({"seed", "gap", "chain", "split", "require"})

#: How many *guesses* -- curiosity answers about a subject the utterance
#: named -- may earn a corroboration fan-out in one run. Letting eight
#: through turned `do fish run` into 47 questions about gills and slime; one
#: was too few, because whichever guess came back first took the slot and
#: `does a cat purr` spent it on `can a cat walk` instead of `is a cat
#: active`, which is the one the family disagrees about.
GUESSES_WORTH_CHECKING = 2

#: Origins whose answers are about the utterance rather than about the
#: loop's own working. Only these open gaps or start chains.
ASKED_ON_PURPOSE = frozenset({"seed", "gap", "require", "chain"})

#: Parts of speech that name something worth attending to.
ATTENDED_POS = ("NOUN", "PROPN", "VERB", "ADJ", "INTJ")

#: The tagger's labels in WordNet's alphabet.
WORDNET_POS = {"NOUN": "n", "PROPN": "n", "VERB": "v", "ADJ": "a",
               "ADV": "r"}

#: Words that are in the ontology and never worth attending to: they are in
#: every sentence and carry no situation.
IGNORED = frozenset("""
be is are was were do does did have has had can could will would may might
thing things kind kinds sort sorts type types what which who whom whose
""".split())


@dataclass
class Conflict:
    """A claim that did not survive being put to its own family."""

    claim: str
    question: str
    verdict: str
    against: list[tuple[str, str]] = field(default_factory=list)
    detail: str = ""

    def as_dict(self) -> dict:
        return {"claim": self.claim, "question": self.question,
                "verdict": self.verdict,
                "against": [{"question": q, "verdict": v}
                            for q, v in self.against],
                "detail": self.detail}


class Buffer:
    """Everything the loop knows while one utterance is in the air."""

    def __init__(self, text: str, engine, curiosity: attention.Curiosity
                 ) -> None:
        self.text = text
        self.engine = engine
        self.curiosity = curiosity
        self.cycle = 0
        self.activation = attention.Activation()
        #: word -> synset id, held for the whole utterance rather than one
        #: request. This is `pins.py` with its lifetime widened.
        self.pins: dict[str, str] = {}
        #: question text -> Answer, every question this utterance has asked.
        self.answers: dict[str, object] = {}
        #: gaps and doubts that have not yet had questions generated from them
        self._gaps: list[Gap] = []
        self._doubts: list[Doubt] = []
        self._spent_gaps: set[tuple] = set()
        self._spent_doubts: set[tuple] = set()
        #: everything seen, for the page
        self.seen_gaps: list[Gap] = []
        self.seen_doubts: list[Doubt] = []
        #: Concepts named by the utterance itself, as opposed to ones that
        #: turned up inside an answer. Curiosity runs over these only.
        #: Without the distinction attention drifts: `what is a wemble` comes
        #: back defining `greeting`, whose definition mentions `land`, and two
        #: cycles later the system is asking whether land is brown.
        self.topics: list[str] = []
        #: The subset of `topics` the utterance itself named, as opposed to
        #: ones the loop chose to look up. A guess about something you
        #: actually said is worth corroborating; a guess about something you
        #: went and fetched is not.
        self.said: set[str] = set()
        #: word -> the WordNet part-of-speech letter the tagger assigned it
        #: in this utterance, so a sense list can be offered in the part of
        #: speech the word was actually used as.
        self.tags: dict[str, str] = {}
        #: Whether the utterance asked what something *is*. Only then does a
        #: definition ladder belong: otherwise it defines the words in the
        #: loop's own follow-ups, several rungs deep, about nothing.
        self.wants_a_definition = bool(
            re.match(r"^\s*(what|who)\s+(is|are|was|were)\s", text or "",
                     re.I))
        #: Gap and doubt questions that were generated but did not fit in a
        #: cycle. They are carried rather than dropped: a family check cut
        #: short by the pool size reports `1 of 4 deny it` about a family of
        #: seven, which is worse arithmetic than not checking at all.
        self.carried: list = []
        #: The answers from the cycle just finished. A chain of thought reads
        #: their *content*, so it can only be formed once they are back.
        self.recent: list = []
        #: question -> how many answers deep it was formed. The seed is 0.
        self.depths: dict[str, int] = {}
        #: question -> the top partition the chain that produced it started
        #: in, so a definition ladder cannot wander out of its own sort.
        self.sorts: dict[str, str] = {}
        #: family checks already probed for which side the subject is on
        self._spent_splits: set[str] = set()
        #: How many curiosity answers have earned a corroboration fan-out.
        #: `do fish run` let eight of them through and spent 47 questions on
        #: gills, slime and fishy smell -- none of it about running.
        self._guesses_checked = 0
        #: sense mismatches not yet re-asked under a pin
        self._senses: list = []
        #: question text -> the subject word it was asked about
        self._subjects: dict[str, str] = {}

    # -- attending ---------------------------------------------------------
    def attend(self, text: str) -> list[str]:
        """Raise activation for every concept the text names.

        The subject is bumped hardest because it is what the sentence is
        about; everything else the ontology recognises gets a smaller share.
        Words the ontology does not know are not attended to at all -- they
        surface as a `word` gap when the question is asked, which is the
        honest place for them.
        """
        parse = self.engine.parser.parse(text)
        found: list[str] = []
        vocabulary = self.engine.parser.vocabulary
        for token in parse.tokens or []:
            lemma = (token.get("lemma") or token.get("text") or "").lower()
            if (token.get("pos") not in ATTENDED_POS or lemma in IGNORED
                    or len(lemma) < 2):
                continue
            if vocabulary and lemma not in vocabulary:
                continue
            if lemma in found:
                continue
            found.append(lemma)
            self.tags.setdefault(lemma, WORDNET_POS.get(token.get("pos"), ""))
            self.activation.bump(lemma, 0.6, self.cycle)
        # Only the *subject* becomes a topic. Everything else the sentence
        # names is activated -- it is part of the situation -- but it is not
        # something to be curious about, because curiosity asks what a thing
        # is like and the object of a question is not what the question is
        # about. `does a snake have legs` produced `is a leg furry`, and
        # `what eats meat` produced `can a meat walk`.
        if parse.subject:
            subject = parse.subject.lower()
            self.activation.bump(subject, 1.0, self.cycle)
            if subject not in found:
                found.insert(0, subject)
            if subject not in self.topics:
                self.topics.append(subject)
            self.said.add(subject)
        return found

    def take_topic(self, concept: str) -> None:
        """A concept becomes a topic by having been looked up on purpose.

        `fish` is not a topic in `a whale is a fish` -- it is the target. It
        becomes one when the loop asks `what is a fish` to close a gap,
        because then the system has chosen to be about it.
        """
        word = (concept or "").split(".")[0].replace("_", " ").lower()
        if word and word not in self.topics:
            self.topics.append(word)

    # -- recording ---------------------------------------------------------
    def record(self, answers) -> tuple[list[Gap], list[Doubt]]:
        """File a cycle's answers, and read what they left open."""
        fresh_gaps: list[Gap] = []
        fresh_doubts: list[Doubt] = []
        self.recent = list(answers)
        for answer in answers:
            parse = (answer.payload or {}).get("parse") or {}
            if parse.get("subject"):
                self._subjects[answer.question] = parse["subject"].lower()
            self.answers[answer.key] = answer
            if answer.error:
                continue
            hole = read_gap(answer.payload, answer.question)
            if hole is not None and hole.blocker:
                key = (hole.kind, hole.blocker)
                if answer.origin not in ASKED_ON_PURPOSE:
                    # A question the loop generated as a *means* is not a
                    # topic. `is an emperor penguin strong` came back
                    # UNRECORDED while checking the penguin family, and the
                    # gap it opened had the loop asking what an emperor is,
                    # what a king is, and what a jackass is. Same for
                    # curiosity: `is a greeting furry` -> UNKNOWN became
                    # `what is a furry`.
                    self.seen_gaps.append(hole)
                elif key not in self._spent_gaps:
                    self._gaps.append(hole)
                    self.seen_gaps.append(hole)
                    fresh_gaps.append(hole)
            for doubt in read_doubts(answer.payload, answer.question):
                # One question earns one corroboration pass, however many
                # reasons there are to doubt it. Three doubts on the same
                # claim would otherwise fan out three identical times.
                if doubt.reason == "sense_mismatch":
                    # Not worth corroborating -- putting the claim to other
                    # kinds of foundry mould answers nothing about pigs --
                    # but worth re-asking under a pin, which is its own
                    # queue.
                    self.seen_doubts.append(doubt)
                    if answer.origin in ASKED_ON_PURPOSE:
                        self._senses.append(doubt)
                    continue
                key = (doubt.concept, doubt.predicate, doubt.relation)
                if key in self._spent_doubts or not doubt.predicate:
                    self.seen_doubts.append(doubt)
                    continue
                guess = (answer.origin == "curiosity"
                         and answer.about in self.said
                         and self._guesses_checked < GUESSES_WORTH_CHECKING)
                worth = answer.origin in DELIBERATE or guess
                if not worth:
                    # A corroboration question is not itself corroborated, and
                    # neither is a guess about something nobody mentioned.
                    # `does a fish have gills` was curiosity about a word the
                    # loop had gone and fetched, and fanning its doubt out to
                    # six kinds of fish cost 25 questions in a run about
                    # sharks. Curiosity about the subject you actually named
                    # is different: it is how `does a cat purr` found that the
                    # cat family disagrees about being active.
                    self.seen_doubts.append(doubt)
                    continue
                if guess:
                    self._guesses_checked += 1
                self._doubts.append(doubt)
                self.seen_doubts.append(doubt)
                fresh_doubts.append(doubt)
        return fresh_gaps, fresh_doubts

    def open_senses(self) -> list:
        """Mismatches that have not yet been re-asked under a pin."""
        pending, self._senses = self._senses, []
        return pending

    def subject_of(self, question: str) -> str:
        return self._subjects.get(question, "")

    def open_gaps(self) -> list[Gap]:
        """Gaps that have not yet been turned into questions, and mark them."""
        pending, self._gaps = self._gaps, []
        for hole in pending:
            self._spent_gaps.add((hole.kind, hole.blocker))
        return pending

    def open_doubts(self) -> list[Doubt]:
        pending, self._doubts = self._doubts, []
        for doubt in pending:
            self._spent_doubts.add((doubt.concept, doubt.predicate,
                                    doubt.relation))
        return pending

    def depth_of(self, question: str) -> int:
        """How many answers had to come back before this was askable."""
        return self.depths.get(question, 0)

    def note_depth(self, question: str, depth: int) -> None:
        self.depths[question] = depth

    def sort_of_chain(self, question: str) -> str:
        return self.sorts.get(question, "")

    def note_sort(self, question: str, sort: str) -> None:
        if sort:
            self.sorts[question] = sort

    def barren(self, topic: str) -> bool:
        """Has being curious about this turned up nothing at all?

        `what eats meat` takes `meat` as its subject, and `meat` really is one
        of the corpus concepts -- so the questions are legitimate and every
        one of them comes back UNKNOWN. Attention withdraws rather than
        spending another cycle on it. This is the one place the loop learns
        anything within an utterance.
        """
        seen = [answer for answer in self.answers.values()
                if answer.origin == "curiosity" and answer.about == topic]
        if len(seen) < 3:
            return False
        return all(answer.verdict in ("UNKNOWN", "UNRECORDED", "NO_MATCH", "")
                   for answer in seen)

    def already_asked(self, question: str) -> bool:
        return question in self.answers

    # -- what the answers add up to ---------------------------------------
    def conflicts(self) -> list[Conflict]:
        """Claims whose corroboration went the other way.

        Grouped by the question that spawned them: a doubt fans a claim out to
        the concept it was inherited from and that concept's other kinds, so
        the group is exactly the family the claim was put to. A yes at the top
        and a no anywhere below it is the disagreement worth reporting.
        """
        by_parent: dict[str, list] = {}
        for answer in self.answers.values():
            if answer.origin == "doubt" and answer.parent:
                by_parent.setdefault(answer.parent, []).append(answer)

        found: list[Conflict] = []
        for parent, children in by_parent.items():
            original = self.answers.get(parent)
            if original is None or original.verdict not in POSITIVE:
                continue
            # Only the kin that answered either way. An UNKNOWN is not a
            # concept that failed to deny the claim, and counting it in the
            # denominator made this read `4 of the 7` beside an
            # over-generalisation line reading `2 of the 6` -- two tallies of
            # one family, and no way to tell which was wrong.
            decided = [child for child in children
                       if child.verdict in POSITIVE | NEGATIVE]
            against = [(child.question, child.verdict) for child in decided
                       if child.verdict in NEGATIVE]
            if not against:
                continue
            claim = children[0].predicate or ""
            found.append(Conflict(
                claim=claim, question=parent, verdict=original.verdict,
                against=against,
                detail=f"{len(against)} of the {len(decided)} concepts this "
                       f"claim was put to deny it"))
        return found

    def splits(self) -> list[dict]:
        """Family checks that came back divided, and have not been probed.

        A fan-out that all says one thing settles the claim. A fan-out that
        *disagrees with itself* -- goldfish and minnow have scales, a seahorse
        does not, the shark is unrecorded -- has not settled anything, and
        reporting the count is not an answer. The question it raises is which
        side the subject belongs on, and that question cannot be formed until
        the split comes back. It is the one place the loop reasons in a line
        rather than in a fan.
        """
        groups: dict[str, list] = {}
        for answer in self.answers.values():
            if answer.origin == "doubt" and answer.parent:
                groups.setdefault(answer.parent, []).append(answer)

        found: list[dict] = []
        for parent, kin in groups.items():
            if parent in self._spent_splits:
                continue
            agree = [one for one in kin if one.verdict in POSITIVE]
            differ = [one for one in kin if one.verdict in NEGATIVE]
            if not (agree and differ):
                continue
            asked = self.answers.get(parent)
            subject = (asked.about if asked else "") or ""
            if not subject:
                continue
            self._spent_splits.add(parent)
            found.append({
                "parent": parent, "subject": subject,
                "claim": kin[0].predicate,
                "agree": [one.about for one in agree if one.about],
                "differ": [one.about for one in differ if one.about]})
        return found

    def overreach(self) -> list[dict]:
        """Claims that belong to a few members and were filed under the class.

        `do pigs fly` rests on `mammal.n.01 capable_of fly`. That is a true
        fact about bats and false of the other forty kinds of mammal the
        store knows, and a pig inherits it because it is a mammal.

        The shape is general and does not depend on the words: a claim
        *inherited* from an ancestor, put to that ancestor's own kinds, and
        held by a minority of them. R11 hoists a fact every child states up
        to the parent; this is the same measurement run as a check, and it
        catches the hoisting that should never have happened -- whoever did
        it, and whichever rule let it through.
        """
        families: dict[str, list] = {}
        for answer in self.answers.values():
            if answer.origin == "doubt" and answer.parent:
                families.setdefault(answer.parent, []).append(answer)

        reported = {bad.question for bad in self.conflicts()}
        found: list[dict] = []
        for parent, kin in families.items():
            asked = self.answers.get(parent)
            if asked is None or asked.verdict not in POSITIVE:
                continue
            if parent in reported:
                # The conflict already says the family denies this. Saying it
                # again as a hoisting complaint is the same finding twice,
                # and `does a beagle swim` read both at once.
                continue
            decided = [one for one in kin
                       if one.verdict in POSITIVE | NEGATIVE]
            if len(decided) < 3:
                continue
            hold = [one for one in decided if one.verdict in POSITIVE]
            if len(hold) >= len(decided) / 2:
                continue
            if not hold:
                # Nothing at all bears it out. That is the family denying the
                # claim outright, which the conflict already reports; calling
                # it "a fact about almost none of them" says nothing and says
                # it twice.
                continue
            # The ancestor the claim was inherited *from*, which is where
            # it is filed -- not the subject that inherited it.
            source = next((one.concept for one in self.seen_doubts
                           if one.question == parent and one.concept), "")
            found.append({
                "claim": kin[0].predicate, "question": parent,
                "source": source,
                "holders": [one.about for one in hold],
                "held": len(hold), "asked": len(decided),
                "detail": f"{len(hold)} of the {len(decided)} kinds it was "
                          f"put to bear it out"})
        return found

    def borne_out(self, question: str) -> bool:
        """Did the family actually agree, or did it merely not object?

        `is a dog wild` rests on one Ascent++ fact and the three kinds of dog
        the norms cover returned two UNKNOWNs and one yes. Nothing
        contradicted it, which is not the same as corroboration, and calling
        both "corroborated" makes the word mean nothing.
        """
        kin = [one for one in self.answers.values()
               if one.origin == "doubt" and one.parent == question]
        decided = [one for one in kin
                   if one.verdict in POSITIVE | NEGATIVE]
        if len(decided) < 2:
            return False
        hold = [one for one in decided if one.verdict in POSITIVE]
        return len(hold) > len(decided) / 2

    def settled(self) -> bool:
        """Nothing left that the loop could act on by itself."""
        return (not self._gaps and not self._doubts and not self.carried
                and not self.recent)

    def needs_telling(self) -> list[Gap]:
        """Gaps no question of ours can close: the system has to be told.

        A `word` gap whose repair came back `UNKNOWN_WORD` again is the clear
        case, and it is the one that makes a teaching turn derivable rather
        than guessed.
        """
        blocked: list[Gap] = []
        for hole in self.seen_gaps:
            if hole.kind not in ("word", "construction"):
                continue
            repair = f"what is a {hole.blocker}"
            answer = self.answers.get(repair) or self.answers.get(
                f"what is an {hole.blocker}")
            if answer is None or answer.verdict in ("UNKNOWN_WORD", "UNPARSED"):
                blocked.append(hole)
        return blocked

    def as_dict(self) -> dict:
        return {"text": self.text, "cycle": self.cycle, "tags": dict(self.tags),
                "activation": self.activation.as_dict(),
                "pins": dict(self.pins),
                "asked": len(self.answers),
                "gaps": [hole.as_dict() for hole in self.seen_gaps],
                "doubts": [doubt.as_dict() for doubt in self.seen_doubts],
                "conflicts": [bad.as_dict() for bad in self.conflicts()],
                "needs_telling": [hole.as_dict()
                                  for hole in self.needs_telling()]}
