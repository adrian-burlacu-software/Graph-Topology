"""One conversation: read, resolve, remember, and answer by v687's rules.

    there is a beagle         placed under beagle.n.01 in episodic memory
    can it swim               v687 walks from it: nothing told, so R1 up to
                              dog, where swimming is recorded -- and v688
                              answers the kind
    it can't swim             told: not_capable_of swim
    can it swim               no -- R3 at distance 0, before anything is
                              inherited
    he was flying             told: capable_of fly -- doing shows it can
    is it black               E1: a quality does not descend to one of them

## Where each answer comes from

A question about an individual is put to `EpisodicReasoner` -- v687's own
`verify` or `classify`, walking up from the individual. Three outcomes:

- **The walk decides at the individual.** A told fact matched (R4) or a told
  negation blocked (R3). That is the answer, and it says so.
- **The walk stops there by E1.** A quality: not known of this one, and what
  the kind tends to be is shown beside it.
- **The walk passes the individual.** Nothing was told, so it is a question
  about the kind, and v688 -- corroboration, R19, the teacher -- answers it.
  `is it a dog` is the exception: the taxonomy settles it exactly, from the
  individual, with no loop needed.

## Doing, and being able to

`it was flying` is stored as `capable_of fly`: doing shows ability. `it
wasn't flying` and `it doesn't fly` are stored as `did_not`, which no rule
reads, because not doing shows nothing about ability -- stored as
`not_capable_of`, R3 would answer `can it fly` no about a pig that was simply
on the ground. `did_not` answers `does it fly` and nothing else. Only `can't`
denies ability, and what cannot fly does not fly either, which R3 gives.

What the flying was *in* is not reasoned about: `it was in an airplane` might
explain the flying away, and reading that out of context is the ambiguous
inference this layer stays out of.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.v687 import rules

from .discourse import Discourse, Referent, Resolution
from .episodic import DID_NOT, EpisodicMemory
from .reading import (Reading, article, kind_question, mode_of, progressive,
                      read)

#: v688's readings, as a word a reply can start with.
WORD = {"verified": "yes", "denied": "no"}

#: v687's verdicts, as v688's readings.
OUTCOME = {"VERIFIED": "verified", "CONTRADICTED": "denied"}


def summary_of(run: dict | None) -> tuple[str, str, str]:
    """(outcome, headline, trust) from a v688 run."""
    summary = (run or {}).get("summary") or {}
    lines = summary.get("lines") or []
    return (summary.get("outcome") or "unknown", lines[0] if lines else "",
            summary.get("trust") or "")


def be(referent: Referent) -> str:
    return "are" if referent.speaker else "is"


def reading_of(parse) -> tuple[str | None, str]:
    """(relation, target) for a statement, as v687's engine finally reads it.

    A bare word after the copula that is also a noun parses as a hedged
    `is_a`: `is a beagle black`, because `black` is a colour. v687's engine
    tries the taxonomy and falls back to the property reading when it finds
    nothing, and for a statement there is nothing to try -- nobody tells a
    beagle it is a kind of black -- so a hedged `is_a` is a property outright.
    On the page it was refused as a taxonomy claim and never stored.
    """
    relation, target = parse.relation, (parse.target or "").strip()
    if relation == "is_a" and getattr(parse, "hedged", False):
        return "has_property", target
    return relation, target


def walk_of(answer) -> dict | None:
    """v687's derivation, without the rule glossary every payload repeats."""
    if answer is None:
        return None
    return {"verdict": answer.verdict, "note": answer.note,
            "chain": list(answer.chain),
            "steps": [step.as_dict() for step in answer.steps],
            "evidence": [fact.as_dict() for fact in answer.evidence]}


@dataclass
class Turn:
    number: int
    said: str
    act: str
    reading: Reading | None = None
    resolution: Resolution | None = None
    asked: str = ""                   # the question about the kind
    answer: dict = field(default_factory=dict)
    run: dict | None = None           # v688, where the kind was asked
    walk: dict | None = None          # v687, walking from the individual
    growth: list = field(default_factory=list)
    discourse: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"number": self.number, "said": self.said, "act": self.act,
                "reading": self.reading.as_dict() if self.reading else None,
                "resolution": (self.resolution.as_dict()
                               if self.resolution else None),
                "asked": self.asked, "answer": self.answer, "run": self.run,
                "walk": self.walk, "growth": self.growth,
                "discourse": self.discourse, "memory": self.memory}


class Session:
    """One conversation: attention in `discourse`, what it knows in `memory`."""

    def __init__(self, asker) -> None:
        self.asker = asker
        self.memory = EpisodicMemory(asker.reasoner)
        self.discourse = Discourse(self.memory, asker.sense)
        self.turns: list[Turn] = []

    # -- one utterance -----------------------------------------------------
    def say(self, text: str) -> Turn:
        self.discourse.next_turn()
        grown = len(self.memory.growth)
        reading = read(text, self.asker, self.discourse.names())
        turn = Turn(self.discourse.turn, reading.said, reading.act, reading)
        {"introduce": self._introduce, "tell": self._tell,
         "ask": self._ask, "what": self._what, "name": self._name,
         "ask_name": self._ask_name}.get(
            reading.act, self._generic)(reading, turn)
        turn.growth = [one.as_dict() for one in self.memory.growth[grown:]]
        turn.discourse = self.discourse.as_dict()
        turn.memory = self.memory.as_dict()
        self.turns.append(turn)
        return turn

    def _resolve(self, reading: Reading, turn: Turn) -> Referent | None:
        turn.resolution = self.discourse.resolve(reading.mention)
        if turn.resolution.referent is None:
            turn.answer = {"outcome": "which", "source": "conversation",
                           "text": turn.resolution.how}
        return turn.resolution.referent

    def _kind(self, referent: Referent) -> str:
        return f"{article(referent.kind)} {referent.kind}"

    # -- telling -----------------------------------------------------------
    def _claim(self, referent: Referent, aux, rest, holds: bool):
        """(relation, object, kind question) for a statement, as v687 reads
        it; relation is None where v687 reads no relation it stores."""
        aux, rest = progressive(aux, rest, self.asker)
        question = kind_question(aux, rest, referent.kind, self.asker.lemma)
        parse = self.asker.parse(question)
        relation, target = reading_of(parse)
        if not relation or not target or relation == "is_a":
            return None, None, question
        if holds:
            return relation, target, question
        if relation in rules.POSITIVES:
            if relation == "capable_of" and mode_of(aux) != "can":
                return DID_NOT, target, question
            return rules.POSITIVES[relation], target, question
        return relation, f"no {target}", question

    def _remember(self, referent: Referent, aux, rest, holds: bool,
                  said: str, turn: Turn) -> str:
        relation, obj, question = self._claim(referent, aux, rest, holds)
        turn.asked = question
        if relation is None:
            turn.run = self.asker.run(question)
            return (f"not something I can store about "
                    f"{self.discourse.describe(referent)}: v687 reads "
                    f"“{question}” as no relation it keeps")
        self.memory.tell(referent.id, relation, obj, said)
        turn.run = self.asker.run(question)
        outcome, _, _ = summary_of(turn.run)
        self.memory.against[(referent.id, relation, obj)] = outcome
        kind = self._kind(referent)
        stored = f"{relation} “{obj}”"
        if outcome in WORD and (WORD[outcome] == "yes") != holds:
            return (f"stored {stored} — an exception: for {kind} in general "
                    f"v688 answers “{question}” {outcome}")
        if outcome in WORD:
            return f"stored {stored}, as expected of {kind}"
        return (f"stored {stored} — new: nothing settles “{question}” for "
                f"{kind} in general")

    def _introduce(self, reading: Reading, turn: Turn) -> None:
        referent = self._resolve(reading, turn)
        if reading.owned:
            self.discourse.own(referent)
        if reading.name:
            self.discourse.rename(referent, reading.name)
        for modifier in reading.mention.modifiers:
            self.memory.tell(referent.id, "has_property", modifier,
                             reading.said)
        text = (f"noted: {self.discourse.describe(referent)}, placed under "
                f"{self.memory.parent.get(referent.id) or referent.kind}")
        if reading.owned:
            text += ", yours"
        if reading.relative is not None:
            text += " — " + self._remember(
                referent, reading.relative.aux, reading.relative.rest,
                reading.relative.holds, reading.said, turn)
        turn.answer = {"outcome": "noted", "source": "conversation",
                       "text": text}

    def _tell(self, reading: Reading, turn: Turn) -> None:
        referent = self._resolve(reading, turn)
        if referent is None:
            return
        narrowed = self._narrow(referent, reading)
        if narrowed:
            turn.answer = {"outcome": "noted", "source": "conversation",
                           "text": narrowed}
            return
        turn.answer = {"outcome": "noted", "source": "told",
                       "text": "noted — " + self._remember(
                           referent, reading.aux, reading.rest,
                           reading.holds, reading.said, turn)}

    def _narrow(self, referent: Referent, reading: Reading) -> str:
        """`it is a beagle`, said of a dog: a narrower kind, by R1 both ways."""
        rest = reading.rest
        if (reading.aux not in ("am", "is", "was") or not reading.holds
                or len(rest) < 2 or rest[0] not in ("a", "an")):
            return ""
        named = " ".join(rest[1:])
        if not self.asker.known(named):
            return ""
        described = self.discourse.describe(referent)
        verb = be(referent)
        if self.memory.reasoner.classify(referent.id, named).verdict == \
                "VERIFIED":
            return (f"yes — {described} {verb} {self._kind(referent)}, and "
                    f"that is {article(named)} {named}")
        sense = self.asker.sense(named)
        if sense and self.memory.base.classify(sense, referent.kind).verdict \
                == "VERIFIED":
            before = referent.kind
            self.discourse.narrow(referent, named)
            return (f"noted — {described} "
                    f"{'were' if referent.speaker else 'was'} known only as "
                    f"{article(before)} {before}, and {verb} now placed under "
                    f"{sense}")
        return ""

    def _name(self, reading: Reading, turn: Turn) -> None:
        referent = self._resolve(reading, turn)
        if referent is None or not reading.name:
            return
        before = referent.name
        self.discourse.rename(referent, reading.name)
        who = ("you are" if referent.speaker else
               f"{self.discourse.describe(referent, named=False)} is called")
        text = f"noted: {who} {reading.name}"
        if before and before != reading.name:
            text += f" (it was {before})"
        turn.answer = {"outcome": "noted", "source": "told", "text": text}

    # -- asking ------------------------------------------------------------
    def _ask(self, reading: Reading, turn: Turn) -> None:
        referent = self._resolve(reading, turn)
        if referent is None:
            return
        aux, rest = progressive(reading.aux, reading.rest, self.asker)
        turn.asked = kind_question(aux, rest, referent.kind, self.asker.lemma)
        parse = self.asker.parse(turn.asked)
        relation, target = parse.relation, (parse.target or "").strip()
        reasoner = self.memory.reasoner
        described = self.discourse.describe(referent)
        kind = self._kind(referent)

        walk = None
        if relation == "is_a" and target:
            walk = reasoner.classify(referent.id, target)
            # v687's engine's order for a hedged `is_a`: the taxonomy first,
            # the property reading when it does not verify. `is it black`
            # came back CONTRADICTED by R27 -- black is a colour, and a colour
            # is not in the animal branch -- which answers a question nobody
            # asked about one beagle.
            if getattr(parse, "hedged", False) and walk.verdict != "VERIFIED":
                relation = "has_property"
                walk = reasoner.verify(referent.id, relation, target,
                                       self.asker.matcher)
        elif relation and target:
            walk = reasoner.verify(referent.id, relation, target,
                                   self.asker.matcher)
        turn.walk = walk_of(walk)

        if walk is not None and relation == "is_a":
            outcome = OUTCOME.get(walk.verdict, "unknown")
            turn.answer = {
                "outcome": outcome, "source": "kind",
                "text": (f"{WORD.get(outcome, 'not settled')} — walking up "
                         f"from {described}, the taxonomy answers "
                         f"{walk.verdict} (R1)")}
            return

        here = [fact for fact in (walk.evidence if walk else [])
                if fact.concept == referent.id]
        if walk is not None and walk.verdict in OUTCOME and here:
            fact = here[0]
            outcome = OUTCOME[walk.verdict]
            said = self.memory.said.get(
                (referent.id, fact.relation, fact.object), "")
            rule = "R3" if walk.verdict == "CONTRADICTED" else "R4"
            text = (f"{WORD[outcome]} — you told me so: “{said}”. {rule} at "
                    f"distance 0, before anything is inherited")
            kind_said = self.memory.against.get(
                (referent.id, fact.relation, fact.object))
            if kind_said in WORD and kind_said != outcome:
                text += (f"; for {kind} in general v688 answers {kind_said}, "
                         f"so {described} {be(referent)} an exception")
            turn.answer = {"outcome": outcome, "source": "told",
                           "text": text}
            return

        if relation == "capable_of" and mode_of(aux) == "does":
            for fact in self.memory.facts.get(referent.id, []):
                if (fact.relation == DID_NOT
                        and self.asker.matcher(fact.object, target)):
                    said = self.memory.said.get(
                        (referent.id, fact.relation, fact.object), "")
                    turn.answer = {
                        "outcome": "denied", "source": "told",
                        "text": f"no — you told me so: “{said}”"}
                    return

        turn.run = self.asker.run(turn.asked)
        outcome, _, trust = summary_of(turn.run)
        v688 = f"v688 answers “{turn.asked}” {outcome}" + (
            f" ({trust})" if trust else "")
        if walk is not None and any(step.rule == "E1" for step in walk.steps):
            turn.answer = {
                "outcome": "unknown", "source": "tendency",
                "text": (f"not known of {described} — E1: a quality does not "
                         f"descend from {referent.kind} to one of them. For "
                         f"{kind} in general, {v688}")}
            return
        turn.answer = {
            "outcome": outcome, "source": "kind",
            "text": (f"{WORD.get(outcome, 'not settled')} — nothing was told "
                     f"of {described}, so v687's walk passes up to "
                     f"{referent.kind}, and {v688}")}

    def _ask_name(self, reading: Reading, turn: Turn) -> None:
        referent = self._resolve(reading, turn)
        if referent is None:
            return
        whose = ("your" if referent.speaker else
                 f"{self.discourse.describe(referent, named=False)}'s")
        if referent.name:
            turn.answer = {"outcome": "retrieved", "source": "told",
                           "text": f"{whose} name is {referent.name}"}
        else:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"nobody has told me {whose} name"}

    def _what(self, reading: Reading, turn: Turn) -> None:
        referent = self._resolve(reading, turn)
        if referent is None:
            return
        text = (f"{self.discourse.describe(referent, named=False)}: "
                f"{self._kind(referent)}, under "
                f"{self.memory.parent.get(referent.id) or 'no known sense'}")
        if referent.name:
            text += f", called {referent.name}"
        told = [self.memory.said.get((referent.id, fact.relation,
                                      fact.object), "")
                for fact in self.memory.facts.get(referent.id, [])]
        told = list(dict.fromkeys(one for one in told if one))
        if told:
            text += "; you told me " + "; ".join(f"“{one}”" for one in told)
        turn.answer = {"outcome": "retrieved", "source": "conversation",
                       "text": text}

    def _generic(self, reading: Reading, turn: Turn) -> None:
        """About a kind, not an individual: v688's question, asked as said."""
        turn.asked = reading.said
        turn.run = self.asker.run(reading.said)
        outcome, headline, trust = summary_of(turn.run)
        turn.answer = {"outcome": outcome, "source": "kind",
                       "text": headline + (f" ({trust})" if trust else "")}

    def as_dict(self) -> dict:
        return {"turns": [turn.as_dict() for turn in self.turns],
                "discourse": self.discourse.as_dict(),
                "memory": self.memory.as_dict()}
