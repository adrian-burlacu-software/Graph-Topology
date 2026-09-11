"""One conversation: read, resolve, remember, and answer by v687's rules.

    a wemble is a kind of animal   taught: wemble -> animal.n.01
    wembles can fly                taught: capable_of fly, on wemble
    beagles can't swim             taught: not_capable_of swim, on beagle.n.01
    there is a beagle              an individual, under beagle.n.01
    can it swim                    no -- R3 at distance 1: what you taught
                                   about beagles blocks what dogs do
    there was a pig                an individual, under hog.n.03
    he was flying                  told: capable_of fly -- doing shows it can
    it was in an airplane          told: at_location airplane -- and E2: an
                                   airplane flies, so the flying was the
                                   airplane's, and the pig's is withdrawn

Everything goes into episodic memory and nothing into the store.

## Where each answer comes from

A question -- about an individual or about a kind -- is put to
`EpisodicReasoner`, v687's own `verify` or `classify` walking up through both
memories:

- **What you told me of this one decides it**: R3 or R4 at distance 0.
- **What you taught me decides it**: a norm on its kind, a taught edge, or a
  kind the store never had, met further up the same walk.
- **A taught kind the walk cannot settle** is asked of the nearest kind above
  it that the store has: v688 has never heard of a wemble, but it has heard
  of animals, and `can a wemble breathe` is `can an animal breathe`.
- **The walk stops at the individual by E1**: a quality, not known of it.
- **Nothing episodic bears on it**: the walk passed into the store, and v688
  -- corroboration, R19, the teacher -- answers the kind.

## Doing, being able to, and being carried

`it was flying` is stored as `capable_of fly`: doing shows ability. `it
wasn't flying` is `did_not`, which no rule reads, because not doing shows
nothing about ability. Only `can't` denies ability, and what cannot fly does
not fly either, which R3 gives.

**E2.** `it was in an airplane` is `at_location airplane`, and something seen
doing what the thing carrying it does was not the one doing it: the flying
was the airplane's. The pig's `capable_of fly` is withdrawn and kept as
`carried fly`, whichever order the two were said in. Whether the carrier does
it is asked of v687's walk first and of v688 only if the store has nothing. A
claim said outright -- `it can fly` -- is never withdrawn, and neither is a
doing whose carrier cannot do it: a pig on a cat was still flying.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.v687 import rules

from .discourse import Discourse, Referent, Resolution
from .episodic import CARRIED, DID_NOT, TOLD, EpisodicMemory, name_of
from .reading import (COPULA, RELATIVE, Reading, article, kind_question,
                      mode_of, progressive, read)

#: v688's readings, as a word a reply can start with.
WORD = {"verified": "yes", "denied": "no"}

#: v687's verdicts, as v688's readings.
OUTCOME = {"VERIFIED": "verified", "CONTRADICTED": "denied"}

#: Being inside or on something: what E2 reads as being carried.
CARRYING = ("in", "on", "inside", "aboard")

#: `a kind of animal`, `a type of dog`.
HEDGES = ("kind", "type", "sort")


def summary_of(run: dict | None) -> tuple[str, str, str]:
    """(outcome, headline, trust) from a v688 run."""
    summary = (run or {}).get("summary") or {}
    lines = summary.get("lines") or []
    return (summary.get("outcome") or "unknown", lines[0] if lines else "",
            summary.get("trust") or "")


def be(referent: Referent) -> str:
    return "are" if referent.speaker else "is"


def reading_of(parse) -> tuple[str | None, str]:
    """(relation, target) as v687's engine finally reads them.

    A bare word after the copula that is also a noun parses as a hedged
    `is_a`: `is a beagle black`, because `black` is a colour. v687's engine
    tries the taxonomy and falls back to the property reading when it finds
    nothing, and nobody tells a beagle it is a kind of black, so a hedged
    `is_a` is a property here. On the page it was refused as a taxonomy
    claim and never stored.
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


def carrier_in(rest: list[str]) -> str:
    """`in an airplane` -> `airplane`; "" if it is not being in or on
    something. v687's parser reads the phrase as a quality."""
    if len(rest) < 2 or rest[0] not in CARRYING:
        return ""
    body = list(rest[1:])
    if body and body[0] in ("a", "an", "the"):
        body = body[1:]
    return " ".join(body)


def taxonomy_parent(aux: str | None, rest: list[str]) -> str:
    """`a kind of animal`, `an animal` -> `animal`; "" if not a kind."""
    if aux not in COPULA or len(rest) < 2 or rest[0] not in ("a", "an"):
        return ""
    body = list(rest[1:])
    if len(body) >= 3 and body[0] in HEDGES and body[1] == "of":
        body = body[2:]
        if body and body[0] in ("a", "an"):
            body = body[1:]
    cut = next((index for index, word in enumerate(body)
                if word in RELATIVE), len(body))
    return " ".join(body[:cut])


def rule_of(walk, fact) -> str:
    """`R3 at distance 1`: the step that used this fact."""
    for step in walk.steps:
        matched = step.matched or {}
        if (matched.get("concept") == fact.concept
                and matched.get("object") == fact.object):
            return f"{step.rule} at distance {step.distance}"
    return ""


class Taught:
    """v687's lexicon, plus the kinds this conversation taught it.

    v687's parser reads `can a wemble fly` as being about `fly`, because
    `wemble` is not a word it has. A taught kind is looked for first.
    """

    def __init__(self, asker, kinds: dict) -> None:
        self.asker = asker
        self.kinds = kinds

    def _taught(self, word: str) -> str:
        if word in self.kinds:
            return word
        lemma = self.asker.lemma(word)
        return lemma if lemma in self.kinds else ""

    def subject(self, question: str):
        if self.kinds:
            for word in question.lower().split():
                found = self._taught(word)
                if found:
                    return found
        return self.asker.subject(question)

    def lemma(self, word: str) -> str:
        return self.asker.lemma(word)

    def known(self, phrase: str) -> bool:
        return phrase in self.kinds or self.asker.known(phrase)

    def progressive(self, word: str):
        return self.asker.progressive(word)


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
    walk: dict | None = None          # v687, walking through both memories
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
        lexicon = Taught(self.asker, self.memory.kinds)
        reading = read(text, lexicon, self.discourse.names())
        turn = Turn(self.discourse.turn, reading.said, reading.act, reading)
        {"introduce": self._introduce, "tell": self._tell,
         "ask": self._ask, "what": self._what, "name": self._name,
         "ask_name": self._ask_name, "teach": self._teach}.get(
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

    def _kind_node(self, referent: Referent) -> str | None:
        return (self.memory.edges.get(referent.id) or [None])[0]

    def _taught_through(self, walk) -> bool:
        """Did the walk climb a taught edge above any individual?"""
        return any(self.memory.edges.get(node) for node in walk.chain
                   if node not in self.memory.individuals)

    # -- reading a claim ---------------------------------------------------
    def _relation(self, aux, rest, word: str, holds: bool):
        """(relation, object, kind question, mode) for a claim or question
        about `word`; relation is None where nothing can be read."""
        aux, rest = progressive(aux, rest, self.asker)
        question = kind_question(aux, rest, word, self.asker.lemma)
        mode = mode_of(aux)
        parent = taxonomy_parent(aux, rest) or self._plural_kind(aux, rest)
        if parent:
            return "is_a", parent, question, mode
        carrier = carrier_in(rest) if aux in COPULA else ""
        if carrier:
            relation, target = "at_location", carrier
        else:
            relation, target = reading_of(self.asker.parse(question))
            if not relation or not target or relation == "is_a":
                relation, target = self._cue(aux, rest)
        if not relation or not target:
            return None, None, question, mode
        if holds:
            return relation, target, question, mode
        if relation in rules.POSITIVES:
            if relation == "capable_of" and mode != "can":
                return DID_NOT, target, question, mode
            return rules.POSITIVES[relation], target, question, mode
        return relation, f"no {target}", question, mode

    def _plural_kind(self, aux, rest) -> str:
        """`dogs are animals`: a plural kind after `are`, with no article."""
        if aux not in ("are", "were") or not rest or len(rest) > 3:
            return ""
        last = rest[-1]
        lemma = self.asker.lemma(last)
        phrase = " ".join(list(rest[:-1]) + [lemma])
        if lemma != last and (self.asker.known(phrase)
                              or phrase in self.memory.kinds):
            return phrase
        return ""

    def _cue(self, aux, rest) -> tuple[str | None, str]:
        """v687's relation cues, for a subject its parser cannot place."""
        body = [word for word in rest if word not in ("a", "an", "the")]
        if not body:
            return None, ""
        if aux in COPULA:
            return "has_property", " ".join(body)
        if aux in ("has", "have"):
            return "has_a", " ".join(body)
        if body[0] in ("have", "has"):
            return "has_a", " ".join(body[1:])
        if aux is None:
            return "capable_of", " ".join([self.asker.lemma(body[0])]
                                          + body[1:])
        return "capable_of", " ".join(body)

    def _walk(self, node: str, relation: str, target: str):
        reasoner = self.memory.reasoner
        if relation == "is_a":
            return reasoner.classify(node, target)
        return reasoner.verify(node, relation, target, self.asker.matcher)

    # -- teaching kinds ----------------------------------------------------
    def _teach(self, reading: Reading, turn: Turn) -> None:
        """A claim about a kind: taxonomy or a norm, into episodic memory."""
        word = reading.mention.kind
        node = self.memory.kind_node(word, self.asker.sense(word))
        relation, obj, question, mode = self._relation(
            reading.aux, reading.rest, word, reading.holds)
        turn.asked = question
        kind = f"{article(word)} {word}"
        new_kind = self.memory.episodic_only(node)

        if relation == "is_a":
            if not reading.holds:
                turn.answer = {
                    "outcome": "unknown", "source": "conversation",
                    "text": (f"not stored: v687 keeps no negated taxonomy, "
                             f"so “{reading.said}” has nowhere to go")}
                return
            walk = self.memory.reasoner.classify(node, obj)
            turn.walk = walk_of(walk)
            if walk.verdict == "VERIFIED":
                turn.answer = {
                    "outcome": "verified", "source": "kind",
                    "text": (f"yes — {kind} is already {article(obj)} {obj}; "
                             f"R1 walks there without being taught")}
                return
            parent = self.memory.kind_node(obj, self.asker.sense(obj))
            self.memory.relate(node, parent, reading.said)
            text = (f"taught: {kind} is a kind of {obj} — {name_of(node)} "
                    f"now sits under {parent} in episodic memory")
            if walk.verdict == "CONTRADICTED":
                text += (", though the store puts them in branches that "
                         "share nothing (R27); here, what you taught holds")
            turn.answer = {"outcome": "noted", "source": "taught",
                           "text": text}
            return

        if relation is None:
            turn.answer = {
                "outcome": "unknown", "source": "conversation",
                "text": (f"not something I can store about {kind}: v687 "
                         f"reads “{question}” as no relation it keeps")}
            return
        self.memory.tell(node, relation, obj, reading.said, mode)
        stored = f"taught: {relation} “{obj}” on {name_of(node)}"
        if new_kind:
            text = (f"{stored} — {word} is a kind taught here, so nothing "
                    f"in the store bears on it")
        else:
            turn.run = self.asker.run(question)
            outcome, _, _ = summary_of(turn.run)
            self.memory.against[(node, relation, obj)] = outcome
            if outcome in WORD and (WORD[outcome] == "yes") != reading.holds:
                text = (f"{stored} — v688 answers “{question}” {outcome} "
                        f"from the store, and in this conversation what you "
                        f"taught answers first for every {word}")
            elif outcome in WORD:
                text = f"{stored}, as the store already has it"
            else:
                text = f"{stored} — new: the store settles nothing about it"
        turn.answer = {"outcome": "noted", "source": "taught", "text": text}

    # -- telling individuals -----------------------------------------------
    def _remember(self, referent: Referent, aux, rest, holds: bool,
                  said: str, turn: Turn) -> str:
        relation, obj, question, mode = self._relation(
            aux, rest, referent.kind, holds)
        turn.asked = question
        described = self.discourse.describe(referent)
        kind_node = self._kind_node(referent)
        store_kind = bool(kind_node) and not self.memory.episodic_only(
            kind_node)
        if relation in (None, "is_a"):
            if store_kind:
                turn.run = self.asker.run(question)
            return (f"not something I can store about {described}: v687 "
                    f"reads “{question}” as no relation it keeps")
        self.memory.tell(referent.id, relation, obj, said, mode)
        stored = f"stored {relation} “{obj}”"
        kind = self._kind(referent)
        if store_kind:
            turn.run = self.asker.run(question)
            outcome, _, _ = summary_of(turn.run)
            self.memory.against[(referent.id, relation, obj)] = outcome
            if outcome in WORD and (WORD[outcome] == "yes") != holds:
                text = (f"{stored} — an exception: for {kind} in general "
                        f"v688 answers “{question}” {outcome}")
            elif outcome in WORD:
                text = f"{stored}, as expected of {kind}"
            else:
                text = (f"{stored} — new: nothing settles “{question}” for "
                        f"{kind} in general")
        else:
            text = f"{stored} — {referent.kind} is a kind taught here"
        carried = self._carry(referent)
        return text + (f"; {carried}" if carried else "")

    def _carry(self, referent: Referent) -> str:
        """E2: what it was seen doing while carried by something that does
        it was the carrier's doing."""
        facts = self.memory.facts.get(referent.id, [])
        carriers = [fact.object for fact in facts
                    if fact.relation == "at_location"]
        done = [fact for fact in facts if fact.relation == "capable_of"
                and self.memory.mode.get((referent.id, fact.relation,
                                          fact.object)) == "does"]
        notes = []
        for fact in done:
            carrier = next((one for one in carriers
                            if self._does(one, fact.object)), "")
            if not carrier:
                continue
            withdrawal = self.memory.withdraw(referent.id, fact.relation,
                                              fact.object, carrier)
            notes.append(
                f"E2: {article(carrier)} {carrier} does “{fact.object}”, so "
                f"“{withdrawal.said}” was the {carrier}'s doing — "
                f"capable_of “{fact.object}” is withdrawn from "
                f"{self.discourse.describe(referent)} and kept as "
                f"{CARRIED}")
        return "; ".join(notes)

    def _does(self, carrier: str, action: str) -> bool:
        """Does the carrier do this? v687's walk, then v688 on the kind."""
        node = self.asker.sense(carrier) or self.memory.kinds.get(carrier)
        if not node:
            return False
        walk = self.memory.reasoner.verify(node, "capable_of", action,
                                           self.asker.matcher)
        if walk.verdict != "UNKNOWN" or self.memory.episodic_only(node):
            return walk.verdict == "VERIFIED"
        run = self.asker.run(f"can {article(carrier)} {carrier} {action}")
        return summary_of(run)[0] == "verified"

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
                f"{self._kind_node(referent) or referent.kind}")
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
        named = taxonomy_parent(reading.aux, reading.rest)
        if not reading.holds or not named:
            return ""
        if not (self.asker.known(named) or named in self.memory.kinds):
            return ""
        described = self.discourse.describe(referent)
        verb = be(referent)
        if self.memory.reasoner.classify(referent.id, named).verdict == \
                "VERIFIED":
            return (f"yes — {described} {verb} {self._kind(referent)}, and "
                    f"that is {article(named)} {named}")
        node = self.memory.kind_node(named, self.asker.sense(named))
        if self.memory.reasoner.classify(node, referent.kind).verdict == \
                "VERIFIED":
            before = referent.kind
            self.discourse.narrow(referent, named)
            return (f"noted — {described} "
                    f"{'were' if referent.speaker else 'was'} known only as "
                    f"{article(before)} {before}, and {verb} now placed under "
                    f"{node}")
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
    def _from_walk(self, walk, subject: str, turn: Turn) -> bool:
        """Answer from a told or taught fact the walk used, if it used one."""
        told = [fact for fact in walk.evidence if fact.source == TOLD]
        if walk.verdict not in OUTCOME or not told:
            return False
        fact = told[0]
        outcome = OUTCOME[walk.verdict]
        said = self.memory.said.get((fact.concept, fact.relation,
                                     fact.object), "")
        where = rule_of(walk, fact)
        if fact.concept in self.memory.individuals:
            text = f"{WORD[outcome]} — you told me so: “{said}”"
            if where:
                text += f". {where}, before anything is inherited"
            kind_said = self.memory.against.get(
                (fact.concept, fact.relation, fact.object))
            if kind_said in WORD and kind_said != outcome:
                text += (f"; the kind in general is {kind_said}, so "
                         f"{subject} is an exception")
            turn.answer = {"outcome": outcome, "source": "told",
                           "text": text}
        else:
            text = (f"{WORD[outcome]} — you taught me: “{said}”, recorded of "
                    f"{name_of(fact.concept)}")
            if where:
                text += f" ({where})"
            turn.answer = {"outcome": outcome, "source": "taught",
                           "text": text}
        return True

    def _ask(self, reading: Reading, turn: Turn) -> None:
        referent = self._resolve(reading, turn)
        if referent is None:
            return
        relation, target, question, mode = self._relation(
            reading.aux, reading.rest, referent.kind, True)
        turn.asked = question
        described = self.discourse.describe(referent)
        kind = self._kind(referent)
        taught_kind = self.memory.episodic_only(self._kind_node(referent))
        if relation is None:
            turn.answer = {
                "outcome": "unknown", "source": "conversation",
                "text": f"v687 reads “{question}” as no relation it keeps"}
            return
        walk = self._walk(referent.id, relation, target)
        turn.walk = walk_of(walk)

        if relation == "is_a":
            outcome = OUTCOME.get(walk.verdict, "unknown")
            turn.answer = {
                "outcome": outcome, "source": "kind",
                "text": (f"{WORD.get(outcome, 'not settled')} — walking up "
                         f"from {described}, the taxonomy answers "
                         f"{walk.verdict} (R1)")}
            return

        if self._from_walk(walk, described, turn):
            return

        if relation == "capable_of" and mode == "does":
            for fact in self.memory.facts.get(referent.id, []):
                if (fact.relation == DID_NOT
                        and self.asker.matcher(fact.object, target)):
                    said = self.memory.said.get(
                        (referent.id, fact.relation, fact.object), "")
                    turn.answer = {"outcome": "denied", "source": "told",
                                   "text": f"no — you told me so: “{said}”"}
                    return

        e1 = any(step.rule == "E1" for step in walk.steps)
        if (taught_kind and walk.verdict not in OUTCOME and not e1
                and self._ask_above(reading, walk, described, turn)):
            return
        if taught_kind or (walk.verdict in OUTCOME
                           and self._taught_through(walk)):
            outcome = "unknown" if e1 else OUTCOME.get(walk.verdict,
                                                       "unknown")
            evidence = walk.evidence[0] if walk.evidence else None
            text = (f"{WORD.get(outcome, 'not settled')} — {described} "
                    f"{be(referent)} {kind}, and what was taught about it is "
                    f"the whole answer: v687's walk says {walk.verdict}")
            if evidence is not None:
                text += (f", from {name_of(evidence.concept)} "
                         f"{evidence.relation} “{evidence.object}”")
            if e1:
                text += " — E1: a quality does not descend to one of them"
            turn.answer = {"outcome": outcome, "source": "taught",
                           "text": text}
            return

        turn.run = self.asker.run(question)
        outcome, _, trust = summary_of(turn.run)
        v688 = f"v688 answers “{question}” {outcome}" + (
            f" ({trust})" if trust else "")
        if e1:
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

    def _ask_above(self, reading: Reading, walk, subject: str,
                   turn: Turn) -> bool:
        """Ask what a taught kind's walk could not settle of the nearest
        kind above it that the store has.

        On the real store `can a wemble breathe` walked from a wemble into
        animal and found only qualified rows (R28) and word-level ones too
        broad to inherit (R12) -- UNKNOWN, while v688, corroborating and
        asking its teacher, answers `can an animal breathe`. A taught kind
        inherits that answer as it inherits the store's facts, from the
        nearest place the store has anything to say.
        """
        above = next((node for node in walk.chain
                      if not self.memory.episodic_only(node)), None)
        if above is None:
            return False
        word = name_of(above)
        _, _, question, _ = self._relation(reading.aux, reading.rest, word,
                                           True)
        turn.run = self.asker.run(question)
        outcome, _, trust = summary_of(turn.run)
        text = (f"{WORD.get(outcome, 'not settled')} — nothing taught settles "
                f"it for {subject}, so it is asked of {word}, the nearest kind "
                f"above it that the store has: v688 answers “{question}” "
                f"{outcome}")
        if trust:
            text += f" ({trust})"
        turn.answer = {"outcome": outcome, "source": "kind", "text": text}
        return True

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
                f"{self._kind_node(referent) or 'no known sense'}")
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
        """About a kind: from episodic memory if anything taught bears on
        it, otherwise v688's question, asked as said."""
        if (reading.mention is not None and reading.mention.kind
                and reading.rest and self._about_kind(reading, turn)):
            return
        turn.asked = reading.said
        turn.run = self.asker.run(reading.said)
        outcome, headline, trust = summary_of(turn.run)
        turn.answer = {"outcome": outcome, "source": "kind",
                       "text": headline + (f" ({trust})" if trust else "")}

    def _about_kind(self, reading: Reading, turn: Turn) -> bool:
        word = reading.mention.kind
        node = self.asker.sense(word) or self.memory.kinds.get(word)
        if node is None:
            turn.answer = {
                "outcome": "unknown", "source": "conversation",
                "text": (f"“{word}” is not a kind the store has, and nothing "
                         f"here has taught me one")}
            return True
        relation, target, question, _ = self._relation(
            reading.aux, reading.rest, word, True)
        if relation is None:
            return False
        walk = self._walk(node, relation, target)
        new_kind = self.memory.episodic_only(node)
        if not (new_kind or self._taught_through(walk)
                or any(fact.source == TOLD for fact in walk.evidence)):
            return False            # nothing episodic bears on it
        turn.asked = question
        turn.walk = walk_of(walk)
        if self._from_walk(walk, f"{article(word)} {word}", turn):
            return True
        if (new_kind and walk.verdict not in OUTCOME
                and self._ask_above(reading, walk, f"{article(word)} {word}",
                                    turn)):
            return True
        outcome = OUTCOME.get(walk.verdict, "unknown")
        evidence = walk.evidence[0] if walk.evidence else None
        text = (f"{WORD.get(outcome, 'not settled')} — by what was taught "
                f"about {word}, v687's walk says {walk.verdict}")
        if evidence is not None:
            text += (f", from {name_of(evidence.concept)} "
                     f"{evidence.relation} “{evidence.object}”")
        turn.answer = {"outcome": outcome, "source": "taught", "text": text}
        return True

    def as_dict(self) -> dict:
        return {"turns": [turn.as_dict() for turn in self.turns],
                "discourse": self.discourse.as_dict(),
                "memory": self.memory.as_dict()}
