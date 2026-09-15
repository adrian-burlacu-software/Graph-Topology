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
- **A store row reached through anything taught is judged by v688**, never
  trusted raw. `animal capable_of fly` is a crawled row about bats, and a
  wemble walked into it and answered yes. The row is put to v688 on the kind
  it was found on -- `can an animal fly` -- and with no row at all, to the
  nearest kind above that the store has. Only what was told or taught is
  answered from the walk alone.
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
doing whose carrier cannot do it: a pig on a cat was still flying. E2 is
recomputed whenever either one is told something, so `the airplane couldn't
fly` gives the pig its flying back.

**Objects.** `it chased the cat` and `it was in the plane` name a second
individual. It is resolved as a subject is, never to the subject, and what is
stored is the fact about its kind -- `capable_of "chase a cat"` -- with which
cat kept beside it. Asked `did it chase the second cat`, a fact bound to the
first is not an answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from research.v687 import rules
from research.v687.executive import (ANSWERED, CONTINUE, DECLINED, Executive,
                                     Operator)
from research.v688 import retrieval

from .goals import Answering
from research.v688.teacher import SETTLING_FLOOR

from .definitions import DEFINED, GlossReader, question_for, says

from .discourse import OBJECT_WEIGHT, Discourse, Referent, Resolution
from .episodic import (CARRIED, DID_NOT, TOLD, EpisodicMemory, Knowledge,
                       name_of)
from .events import KNOWLEDGE
from .reading import (ARTICLES, AUX, CARRYING, COPULA, HAVING,
                      QUESTION_WORDS, RELATIVE, Mention, Reading, article,
                      kind_question, mode_of, perfect, progressive, read,
                      words)
from .relations import DIMENSIONS, Relations, phrase as relation_phrase
from .story import Story
from .tense import When
from .timeline import Timeline

#: An auxiliary agreeing with `they`, as it agrees with one of a kind.
SINGULAR = {"do": "does", "are": "is", "were": "was", "have": "has"}

#: `they`, where nothing here is it, and what stands in its place.
THEY = frozenset({"they", "them"})

#: v688's readings, as a word a reply can start with.
WORD = {"verified": "yes", "denied": "no"}

#: v687's verdicts, as v688's readings.
OUTCOME = {"VERIFIED": "verified", "CONTRADICTED": "denied"}

#: A question's auxiliary, denied: what `why can't it fly` asks of its kind.
DENIAL = {"can": "can't", "could": "couldn't", "does": "doesn't",
          "do": "don't", "did": "didn't", "is": "isn't", "are": "aren't",
          "was": "wasn't", "were": "weren't", "has": "hasn't",
          "have": "haven't", "had": "hadn't", "will": "won't",
          "would": "wouldn't"}

#: `a kind of animal`, `a type of dog`.
HEDGES = ("kind", "type", "sort")


def summary_of(run: dict | None) -> tuple[str, str, str]:
    """(outcome, headline, trust) from a v688 run.

    An answer that is not a yes or no -- a listing, an identification, a
    comparison, a script -- is headlined by what it held: v688's first line
    for it is the verdict word and the question back. What sits beside a yes
    or no, like the kinds that do not fly, is added to it."""
    summary = (run or {}).get("summary") or {}
    lines = summary.get("lines") or []
    headline = lines[0] if lines else ""
    trust = summary.get("trust") or ""
    content = summary.get("content") or {}
    if content.get("text") and content.get("answers"):
        headline, trust = content["text"], ""
    elif content.get("text"):
        headline = (f"{headline} — {content['text']}" if headline
                    else content["text"])
    return summary.get("outcome") or "unknown", headline, trust


def be(referent: Referent) -> str:
    if referent.speaker:
        return "are"
    return "am" if referent.addressee else "is"


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
    something. v687's parser reads the phrase as a quality.

    `either in the school or the park` -> `either school or park`: one place
    of two, and which one not told (T3 answers `maybe`)."""
    if rest[:1] == ["either"] and "or" in rest[2:]:
        at = rest.index("or")
        first = carrier_in(rest[1:at])
        second = carrier_in(rest[1:2] + rest[at + 1:])
        return f"either {first} or {second}" if first and second else ""
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

    def tags(self, words):
        return self.asker.tags(words)

    def analyse(self, words):
        return self.asker.analyse(words)

    def known(self, phrase: str) -> bool:
        return phrase in self.kinds or self.asker.known(phrase)

    def progressive(self, word: str):
        return self.asker.progressive(word)

    def participle(self, word: str):
        return self.asker.participle(word)


@dataclass
class Turn:
    number: int
    said: str
    act: str
    reading: Reading | None = None
    resolution: Resolution | None = None
    #: the object: `the dog` in `it chased the dog`
    binding: Resolution | None = None
    asked: str = ""                   # the question about the kind
    answer: dict = field(default_factory=dict)
    run: dict | None = None           # v688, where the kind was asked
    walk: dict | None = None          # v687, walking through both memories
    growth: list = field(default_factory=list)
    discourse: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    #: definitions read into definitions memory during this turn
    learned: list = field(default_factory=list)
    #: the operators that fired on each claim, in order (`executive.py`)
    trace: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"number": self.number, "said": self.said, "act": self.act,
                "heard": self.reading.heard if self.reading else {},
                "trace": self.trace,
                "reading": self.reading.as_dict() if self.reading else None,
                "resolution": (self.resolution.as_dict()
                               if self.resolution else None),
                "object": self.binding.as_dict() if self.binding else None,
                "asked": self.asked, "answer": self.answer, "run": self.run,
                "walk": self.walk, "growth": self.growth,
                "learned": self.learned,
                "discourse": self.discourse, "memory": self.memory}


class Session:
    """One conversation: attention in `discourse`, what it knows in `memory`."""

    def __init__(self, asker, knowledge: Knowledge | None = None,
                 conversation: str = "", example: bool = False,
                 definitions=None) -> None:
        self.asker = asker
        self.conversation = conversation
        #: an example keeps what it teaches to itself (`longterm.py`)
        self.example = example
        #: definitions memory, shared: what a WordNet gloss says is not
        #: something an example made up, so examples read into it too
        self.definitions = definitions
        self.memory = EpisodicMemory(asker.reasoner, knowledge, conversation,
                                     definitions)
        self.discourse = Discourse(self.memory, asker.sense,
                                   getattr(asker, "gender", None))
        #: story time: a projection of the same stream (`timeline.py`), and
        #: what the conversation does with it (`story.py`)
        self.timeline = Timeline(self.memory.log)
        self.story = Story(self, self.timeline)
        #: where things are against each other, and how big: a projection
        #: of the same stream (`relations.py`)
        self.relations = Relations(self.memory.log)
        #: what the part being acted on says about when
        self._when = When()
        self.turns: list[Turn] = []
        self._turn: Turn | None = None
        self._reader: GlossReader | None = None

    def lexicon(self) -> "Taught":
        return Taught(self.asker, self.memory.kinds)

    def memory_view(self) -> dict:
        """Memory as the page shows it: episodic memory and story time."""
        return {**self.memory.as_dict(), "timeline": self.timeline.as_dict()}

    def snapshot(self) -> dict:
        """Everything needed to carry on after a restart, as plain values."""
        return {"conversation": self.conversation, "example": self.example,
                "memory": self.memory.snapshot(),
                "discourse": self.discourse.snapshot(),
                "knowledge": (self.memory.knowledge.as_state()
                              if self.example else None)}

    @classmethod
    def rebuild(cls, asker, conversation: str, events,
                knowledge: Knowledge | None = None, definitions=None,
                example: bool = False) -> "Session":
        """A conversation from its stream: every event applied again, in
        order, and nothing asked of v687 or v688 on the way."""
        session = cls(asker, knowledge, conversation, example, definitions)
        stream = session.memory.log.conversation
        stream.events[:] = list(events)
        stream.mark_saved()
        session.memory.log.replay(stream.events)
        session.memory.store("replayed")
        if session.timeline.occurrences:
            session.timeline.replan("replayed")
        return session

    @classmethod
    def resume(cls, asker, state: dict, knowledge: Knowledge | None = None,
               definitions=None) -> "Session":
        """A conversation kept as a snapshot before conversations were kept
        as events. The snapshot becomes the stream's first event, so what
        is said from here on is appended after it. An example brings its own
        knowledge back with it; any other reads the one it is given."""
        example = bool(state.get("example"))
        conversation = state.get("conversation") or ""
        if example:
            knowledge = Knowledge.from_state(state.get("knowledge") or {},
                                             f"{KNOWLEDGE}:{conversation}")
        session = cls(asker, knowledge, conversation, example, definitions)
        session.memory.log.record("imported", {
            "memory": state.get("memory") or {},
            "discourse": state.get("discourse") or {}})
        session.memory.store("resumed")
        return session

    # -- one utterance -----------------------------------------------------
    def say(self, text: str) -> Turn:
        self.discourse.next_turn()
        grown = len(self.memory.growth)
        lexicon = Taught(self.asker, self.memory.kinds)
        reading = read(text, lexicon, self.discourse.names())
        turn = Turn(self.discourse.turn, reading.said, reading.act, reading)
        self._turn = turn
        acts = {"introduce": self._introduce, "tell": self._tell,
                "ask": self._ask, "what": self._what, "name": self._name,
                "ask_name": self._ask_name, "teach": self._teach,
                "compound": self._compound, "define": self._define,
                "why": self._why}
        # `hello`, `thanks`, `what can you do` (`social.py`).
        from .social import ACTS as SOCIAL
        acts.update({name: self._social for name in SOCIAL})

        def acted(handler):
            def apply(memory: dict) -> str:
                handler(memory["reading"], memory["turn"])
                return ANSWERED
            return apply

        # What the utterance does, as operators (`executive.py`): the goals a
        # question states, each by its relation's operator (`goals.py`), then
        # each remaining act on its own reading, and anything else is v688's.
        acting = Executive(
            Answering(self).operators()
            + [Operator(name, acted(handler),
                        proposes=lambda memory, name=name:
                        memory["reading"].act == name)
               for name, handler in acts.items()]
            + [Operator("generic", acted(self._generic),
                        proposes=lambda memory:
                        memory["reading"].act not in acts
                        and memory["reading"].act != "question")])
        # Several claims in one statement (`clauses.py`) are acted on in
        # order, and answered together. What the first one resolved to is
        # what the page shows. An anchor nothing was told of is told first,
        # as a claim of its own: `after the dog chased the cat, it slept`.
        parts = []
        for one in [reading] + list(reading.more):
            anchor = self._untold_anchor(one)
            if anchor is not None:
                parts.append(anchor)
            parts.append(one)
        replies, first = [], None
        for index, one in enumerate(parts):
            turn.answer = {}
            self._when = one.when or When()
            self.memory.hidden = frozenset()
            fired = acting.run({"reading": one, "turn": turn,
                                "goals": one.goals})
            # Of several claims, each as it was claimed: its subject, its
            # auxiliary and the rest -- `said` is the whole utterance.
            said = [one.mention.text] if (one.mention is not None
                                           and one.mention.text) else []
            said += ([one.aux] if one.aux else []) + list(one.rest)
            turn.trace.append({"claim": one.said if len(parts) == 1
                               else " ".join(said) or one.said,
                               "act": one.act, **fired.as_dict()})
            if index == 0:
                first = (turn.resolution, turn.binding)
            replies.append(dict(turn.answer))
            # What a bare `why` asks about: the last yes or no put.
            if one.act in ("ask", "generic") and turn.asked:
                self._last_question = (one, turn.asked)
            # What `they` means when nothing here is: the last kind named.
            if (one.mention is not None and one.mention.kind
                    and (one.act in ("generic", "teach", "define")
                         or ("again", "question") in one.cells)):
                self._last_kind = one.mention.kind
        if len(parts) > 1:
            turn.resolution, turn.binding = first
            outcomes = [one.get("outcome") for one in replies]
            turn.answer = {
                "outcome": "noted" if "noted" in outcomes else outcomes[0],
                "source": replies[0].get("source") or "conversation",
                "text": "; ".join(one.get("text") for one in replies
                                  if one.get("text"))}
        self._when = When()
        self.memory.hidden = frozenset()
        turn.growth = [one.as_dict() for one in self.memory.growth[grown:]]
        turn.discourse = self.discourse.as_dict()
        turn.memory = self.memory_view()
        self.turns.append(turn)
        return turn

    def _social(self, reading: Reading, turn: Turn) -> None:
        """`hello`, `thanks`, `what can you do` (`social.py`): nothing told,
        nothing asked of memory; answered by what the conversation is, and
        what it can do by the relations its operators answer."""
        from .social import answer

        turn.answer = {"outcome": "social", "source": "conversation",
                       "act": reading.act,
                       "text": answer(reading.act, list(
                           Answering(self).cells()))}

    def _untold_anchor(self, reading: Reading) -> Reading | None:
        """The clause a statement is placed against, when nothing told is
        what it names: it is a claim too, and is told before the statement."""
        when = reading.when
        if (when is None or not when.anchor
                or reading.act not in ("tell", "introduce")):
            return None
        anchored = read(when.anchor, self.lexicon(), self.discourse.names(),
                        anchored=False)
        if anchored.act != "tell" or self.story.find(anchored) is not None:
            return None
        anchored.said = when.anchor
        anchored.when = When(frame=when.frame)
        return anchored

    def _kind_answer(self, question: str, turn: Turn) -> tuple[str, str]:
        """(outcome, trust) of v688 on a question about a kind, kept on the
        turn."""
        turn.asked = question
        turn.run = self._run(question)
        outcome, _, trust = summary_of(turn.run)
        return outcome, trust

    def _resolve(self, reading: Reading, turn: Turn,
                 described: bool = False) -> Referent | None:
        turn.resolution = self.discourse.resolve(reading.mention,
                                                 described=described)
        if turn.resolution.referent is None:
            turn.answer = {"outcome": "which", "source": "conversation",
                           "text": turn.resolution.how}
        return turn.resolution.referent

    # -- relations between individuals (`relations.py`) --------------------
    def _compared(self, reading: Reading, referent: Referent, other,
                  turn: Turn) -> bool:
        """`is the box bigger than the chest`, `does the box fit in the
        chest`, `is the rectangle to the right of the square`: S2 and S3 over
        what was told. False when the question is not a relation to another
        individual here."""
        found = relation_phrase(list(reading.rest))
        if found is None or other is None:
            return False
        answer = self.relations.compare(referent.id, other.id,
                                        found.dimension, found.side)
        described = self.discourse.describe(referent)
        them = self.discourse.describe(other)
        if answer.value is None:
            turn.answer = {
                "outcome": "unknown", "source": "told",
                "text": (f"not told — nothing said puts {described} and "
                         f"{them} in order, {found.dimension} (S2: absent, "
                         f"not false)")}
            return True
        if answer.level:
            why = (f"{described} and {them} are in line, so neither is "
                   f"{' '.join(found.words)} the other (S3)")
        else:
            why = (" → ".join(f"“{one.said}”" for one in answer.path)
                   + " (S2)")
        turn.answer = {"outcome": "verified" if answer.value else "denied",
                       "source": "told",
                       "text": f"{'yes' if answer.value else 'no'} — {why}"}
        return True

    def _related_to(self, reading: Reading, turn: Turn) -> None:
        """`what is north of the office`, `what is the kitchen north of`: the
        other side of what was told, read both ways (S1)."""
        referent = self._here(reading, turn)
        if referent is None:
            return
        asks_first = reading.rest[:1] == ["?"]
        words = [word for word in reading.rest if word != "?"]
        found = relation_phrase(words)
        side = found.side if asks_first else -found.side
        beside = self.relations.beside(referent.id, found.dimension, side)
        described = self.discourse.describe(referent)
        said = " ".join(words)
        if not beside:
            turn.answer = {
                "outcome": "unknown", "source": "conversation",
                "text": (f"not told — nothing was said to be {said} "
                         f"{described}" if asks_first else
                         f"not told — {described} was not said to be {said} "
                         f"anything")}
            return
        people = list({one: self.discourse.by_id(one)
                       for one, _ in beside}.values())
        quotes = "; ".join(dict.fromkeys(f"“{relation.said}”"
                                         for _, relation in beside))
        turn.answer = {"outcome": "retrieved", "source": "told",
                       "text": (f"{self._names([one for one in people if one])}"
                                f" — {quotes} (S1)")}

    # -- qualities toward something, attributes, motives --------------------
    def _toward(self, reading: Reading, turn: Turn) -> None:
        """`what is Gertrude afraid of`: what a quality toward something is
        toward -- told of this one, or of the nearest kind above it that has
        one, told before stored (R3; E1 is about what a thing is like)."""
        referent = self._here(reading, turn)
        if referent is None:
            return
        head = " ".join(reading.rest) + " "
        described = self.discourse.describe(referent)
        reasoner = self.memory.reasoner
        for node, distance, _ in reasoner.ascend(referent.id):
            found = [fact for fact in reasoner.facts_of(node, "has_property")
                     if fact.object.lower().startswith(head)]
            if not found:
                continue
            fact = found[0]
            said = self.memory.said.get((fact.concept, fact.relation,
                                         fact.object), "")
            if node == referent.id:
                why, source = f"you told me “{said}”", "told"
            elif fact.source == TOLD:
                why = (f"{described} {be(referent)} {self._kind(referent)}, "
                       f"and you taught me “{said}” (R3 at distance "
                       f"{distance})")
                source = "taught"
            else:
                why = (f"{described} {be(referent)} {self._kind(referent)}, "
                       f"and the store records {name_of(node)} as "
                       f"“{fact.object}” ({fact.source}, R3 at distance "
                       f"{distance})")
                source = "kind"
            turn.answer = {"outcome": "retrieved", "source": source,
                           "text": f"{fact.object[len(head):]} — {why}"}
            return
        self._generic(replace(reading, act="generic", mention=None), turn)

    def _values(self, individual: str, attribute: str) -> list[tuple]:
        """(value, said, seq) for each value of an attribute told of one
        individual, in the order told. `green` is a colour because a sense
        of `green` is a kind of a sense of `color`."""
        wanted = {sense["id"] for sense in (self.asker.reasoner.senses_of(
            self.asker.lemma(attribute), "n") or [])}
        cache = self.__dict__.setdefault("_valued", {})
        out = []
        for event in self.memory.log.conversation.events:
            data = event.data
            if (event.type != "told" or data.get("node") != individual
                    or data.get("relation") != "has_property"):
                continue
            value = (data.get("object") or "").lower()
            key = (value, attribute)
            if key not in cache:
                senses = self.asker.reasoner.senses_of(value, "n") or []
                cache[key] = any(
                    node in wanted for sense in senses
                    for node, _, _ in self.asker.reasoner.ascend(sense["id"]))
            if cache[key]:
                out.append((value, data.get("said", ""), event.seq))
        return out

    def _attribute(self, reading: Reading, turn: Turn) -> None:
        """`what color is Greg`: the value told of this one; failing that,
        I1 -- induction from the others of its kind told of here, all of them
        where they agree and the last told where they do not; failing that,
        its kind's, as v688 answers it."""
        referent = self._here(reading, turn)
        if referent is None:
            return
        attribute = reading.rest[0]
        described = self.discourse.describe(referent)
        own = self._values(referent.id, attribute)
        if own:
            value, said, _ = own[-1]
            turn.answer = {"outcome": "retrieved", "source": "told",
                           "text": f"{value} — you told me “{said}”"}
            return
        kin = [one for one in self._individuals(referent.kind)
               if one.id != referent.id]
        told = sorted((seq, value, said, one) for one in kin
                      for value, said, seq in self._values(one.id, attribute))
        if told:
            # The most recently told is the most active evidence: I1 is
            # retrieval by recency over the kin (`retrieval.latest`).
            _, value, said, one = retrieval.latest(told, lambda each: each[0])
            if len({each[1] for each in told}) == 1:
                why = f"every {referent.kind} here that was told of is {value}"
            else:
                why = (f"the {referent.kind}s told of here differ, and the "
                       f"last told, {self.discourse.describe(one)}, is "
                       f"{value}")
            turn.answer = {
                "outcome": "retrieved", "source": "induced",
                "text": (f"probably {value} — nothing was told of "
                         f"{described}'s {attribute}, and {why}: “{said}” "
                         f"(I1: induction from others of its kind)")}
            return
        question = (f"what {attribute} is {article(referent.kind)} "
                    f"{referent.kind}")
        turn.asked = question
        turn.run = self._run(question)
        outcome, headline, trust = summary_of(turn.run)
        turn.answer = {"outcome": outcome, "source": "kind",
                       "text": (f"nothing was told of {described}'s "
                                f"{attribute} — as {self._kind(referent)}: "
                                f"{headline}" + (f" ({trust})" if trust
                                                 else ""))}

    def _states(self, individual: str) -> list[tuple]:
        """(state, said, seq) for every one-word quality told of someone, in
        the order told: `Sumit is tired`."""
        return [((event.data.get("object") or "").lower(),
                 event.data.get("said", ""), event.seq)
                for event in self.memory.log.conversation.events
                if event.type == "told"
                and event.data.get("node") == individual
                and event.data.get("relation") == "has_property"
                and len((event.data.get("object") or "").split()) == 1]

    def _motives(self):
        if getattr(self, "_motives_cache", None) is None:
            from .motives import Motives

            self._motives_cache = Motives(self.asker)
        return self._motives_cache

    def _where_going(self, reading: Reading, turn: Turn) -> None:
        """`where will Sumit go`: nothing told says. What was told of Sumit
        may -- a state the store says moves one to something that a place
        this conversation has been to is for (`motives.py`)."""
        referent = self._here(reading, turn)
        if referent is None:
            return
        described = self.discourse.describe(referent)
        places = {change.place for change in self.timeline.changes
                  if change.kind == "location" and change.place}
        people = {one.id for one in self._individuals("person")}
        here = [one for one in self.discourse.referents
                if one.id in places and one.id not in people]
        motives = self._motives()
        states = self._states(referent.id)
        for state, said, _ in reversed(states):
            ranked = sorted(((len(motives.support(state, one.kind)), one)
                             for one in here), key=lambda each: -each[0])
            if ranked and ranked[0][0]:
                one = ranked[0][1]
                goal, row, purpose = motives.meeting(state, one.kind)
                turn.answer = {
                    "outcome": "retrieved", "source": "kind",
                    "text": (f"probably {self.discourse.describe(one)} — you "
                             f"told me “{said}”; the store has “{row}”, and "
                             f"{article(one.kind)} {one.kind} is for "
                             f"“{purpose}” (ConceptNet)")}
                return
        if states:
            state, said, _ = states[-1]
            goals = motives.goals(state)
            turn.answer = {
                "outcome": "unknown", "source": "conversation",
                "text": (f"not told — you told me “{said}”"
                         + (f", which the store says moves one to "
                            f"“{goals[0][0]}”, and nowhere here was said to "
                            f"be for that" if goals else ""))}
            return
        turn.answer = {"outcome": "unknown", "source": "conversation",
                       "text": f"not told — nothing was said of where "
                               f"{described} will go"}

    def _by_change(self, reading: Reading):
        """The told occurrence a question names by what it changed, when the
        verb told is not a kind of the verb asked: `why did Yann go to the
        kitchen`, told `Yann journeyed to the kitchen` (T4 put Yann there),
        or `why did Yann get the apple`, told `Yann grabbed the apple` (it put
        the apple with Yann). The last such, in story order."""
        if reading.mention is None or reading.obj is None:
            return None
        who = self.story._resolved(reading.mention)
        if who is None:
            return None
        other = self.story._resolved(reading.obj, exclude={who.id})
        if other is None:
            return None
        order = {one.id: index
                 for index, one in enumerate(self.timeline.story())}
        found = []
        for change in self.timeline.changes:
            occurrence = self.timeline.occurrence(change.occurrence)
            if (change.kind != "location" or not change.after
                    or occurrence is None or occurrence.subject != who.id):
                continue
            if (change.individual, change.place) in ((who.id, other.id),
                                                     (other.id, who.id)):
                found.append((order.get(change.occurrence, -1), change.seq,
                              occurrence))
        return max(found, key=lambda one: one[:2])[2] if found else None

    def _because(self, reading: Reading, turn: Turn) -> bool:
        """`why did Sumit go to the bedroom`: a told occurrence, and a state
        told of the one who did it before it that the store says moves one to
        what it did, where, or with what (`motives.py`); failing that, the
        last state told of them, said as only that."""
        occurrence = self.story.find(reading) or self._by_change(reading)
        if occurrence is None or not occurrence.subject:
            return False
        referent = self.discourse.by_id(occurrence.subject)
        states = [one for one in self._states(occurrence.subject)
                  if one[2] < occurrence.seq]
        if referent is None or not states:
            return False
        described = self.discourse.describe(referent)
        motives = self._motives()
        kinds = [thing.kind for thing in map(self.discourse.by_id,
                                             (occurrence.place,
                                              occurrence.object)) if thing]
        for state, said, _ in reversed(states):
            for kind in kinds:
                found = motives.meeting(state, kind)
                if found is not None:
                    goal, row, purpose = found
                    turn.answer = {
                        "outcome": "retrieved", "source": "kind",
                        "text": (f"because {described} {be(referent)} "
                                 f"{state} — you told me “{said}”; the store "
                                 f"has “{row}”, and {article(kind)} {kind} is "
                                 f"for “{purpose}” (ConceptNet)")}
                    return True
        state, said, _ = states[-1]
        turn.answer = {
            "outcome": "unknown", "source": "told",
            "text": (f"perhaps because {described} {be(referent)} {state} — "
                     f"the last thing told of {described} before "
                     f"“{occurrence.said}”, though the store does not say how "
                     f"it bears on it")}
        return True

    def _route(self, reading: Reading, turn: Turn) -> None:
        """`how do you go from the kitchen to the garden`: S4, the shortest
        way over the compass as told, a direction a step."""
        start = self._here(reading, turn)
        if start is None:
            return
        goal = self._here(replace(reading, mention=reading.obj), turn)
        if goal is None:
            return
        steps = self.relations.route(start.id, goal.id)
        there = self.discourse.describe(start)
        where = self.discourse.describe(goal)
        if not steps:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"not told — nothing said leads from "
                                   f"{there} to {where} (S4)"}
            return
        directions = ", then ".join(direction for direction, _, _ in steps)
        through = "; ".join(
            f"{direction} to {self.discourse.describe(self.discourse.by_id(node))}"
            f": “{relation.said}”" for direction, node, relation in steps)
        turn.answer = {"outcome": "retrieved", "source": "told",
                       "text": f"{directions} — {through} (S4)"}

    def _kind(self, referent: Referent) -> str:
        return f"{article(referent.kind)} {referent.kind}"

    def _kind_node(self, referent: Referent) -> str | None:
        return (self.memory.edges.get(referent.id) or [None])[0]

    def _taught_through(self, walk) -> bool:
        """Did the walk climb a taught edge above any individual?"""
        return any(self.memory.edges.get(node) for node in walk.chain
                   if node not in self.memory.individuals)

    def _bind(self, reading: Reading, subject: Referent, turn: Turn):
        """(rest, object): the verb phrase with its object resolved.

        The object is resolved by the same identification and salience as a
        subject, never to the subject itself, and refreshed below it, so `it`
        in the next utterance still means the subject. What goes on to v687
        and v688 is the object's kind in place of its phrase -- `in an
        airplane`, `chase a cat` -- because the store knows kinds; which one
        it was travels beside the fact. (None, None) when it fits no one, and
        the turn says why.
        """
        if reading.obj is None:
            return list(reading.rest), None
        turn.binding = self.discourse.resolve(
            reading.obj, exclude={subject.id}, weight=OBJECT_WEIGHT,
            described=True)
        found = turn.binding.referent
        if found is None:
            turn.answer = {"outcome": "which", "source": "conversation",
                           "text": turn.binding.how}
            return None, None
        rest = list(reading.rest[:reading.obj_at]) + [article(found.kind),
                                                      found.kind]
        return rest, found

    # -- reading a claim ---------------------------------------------------
    def _relation(self, aux, rest, word: str, holds: bool):
        """(relation, object, kind question, mode) for a claim or question
        about `word`; relation is None where nothing can be read."""
        aux, rest = progressive(aux, rest, self.asker)
        aux, rest = perfect(aux, rest, self.asker)
        question = kind_question(aux, rest, word, self.asker.lemma)
        mode = mode_of(aux)
        # `north of the office`, `fits inside the box`: a relation to another
        # individual (`relations.py`), kept as said, beside which one it is.
        if relation_phrase(rest) is not None:
            return ("has_property" if holds else "not_has_property",
                    " ".join(rest), question, mode)
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
        if aux in HAVING:
            return "has_a", " ".join(body)
        if body[0] in HAVING:
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
    # -- definitions -------------------------------------------------------
    def _run(self, question: str) -> dict:
        """v688 on a question, and every definition it retrieved on the way
        read into definitions memory."""
        run = self.asker.run(question)
        self._harvest(run)
        return run

    def _harvest(self, run) -> None:
        if self.definitions is None or not isinstance(run, dict):
            return
        seen: list = []
        for cycle in run.get("cycles") or []:
            for answer in cycle.get("answers") or []:
                concept = answer.get("concept")
                if (answer.get("verdict") == "DEFINED" and concept
                        and concept not in seen):
                    seen.append(concept)
                    self._learn_definition(concept)

    def _learn_definition(self, concept: str) -> dict | None:
        """Read one gloss into definitions memory, asking the teacher about
        each fact where there is one. Once per concept, ever."""
        if self.definitions is None or self.definitions.has(concept):
            return None
        gloss = self.asker.reasoner.gloss(concept)
        if not gloss:
            return None
        if self._reader is None:
            self._reader = GlossReader(self.asker)
        reading = self._reader.read(concept, gloss)
        checks = {}
        for fact in reading.facts:
            found = self.asker.judge(question_for(
                name_of(concept), fact.relation, fact.object, fact.rule))
            if found is None:
                continue
            supports, confidence = found
            expected = not fact.relation.startswith("not_")
            checks[(fact.relation, fact.object)] = (
                "below" if confidence < SETTLING_FLOOR else
                "agreed" if supports == expected else "disputed", confidence)
        self.definitions.keep(reading, "retrieved", checks)
        entry = self.definitions.entry(concept)
        if self._turn is not None and entry is not None:
            self._turn.learned.append(entry)
        return entry

    def _definition_text(self, entry: dict, word: str) -> str:
        facts = [says(fact["relation"], fact["object"], fact["rule"] or "")
                 for fact in entry["facts"] if fact["checked"] != "disputed"]
        text = f"{article(word)} {word}: “{entry['gloss']}”"
        if entry.get("genus"):
            text += f" — a kind of {entry['genus']}"
        if facts:
            text += "; it " + "; ".join(facts)
        disputed = [f"{fact['relation']} {fact['object']}"
                    for fact in entry["facts"]
                    if fact["checked"] == "disputed"]
        if disputed:
            text += (" (the teacher disputed, and nothing reads: "
                     + "; ".join(disputed) + ")")
        return text

    def _define(self, reading: Reading, turn: Turn) -> None:
        """`what is a testicle`: from definitions memory, retrieving the
        definition through v688 the first time it is asked."""
        word = reading.mention.kind
        node = self.asker.sense(word)
        taught = self.memory.kinds.get(word)
        if node is None and taught is not None:
            # `what is a wemble`: the store has no word for it, and v688 said
            # so. What it is, is what it was taught to be.
            parents = [name_of(one) for one in self.memory.edges.get(taught,
                                                                      [])]
            told = [self.memory.said.get((taught, fact.relation, fact.object),
                                         "")
                    for fact in self.memory.facts.get(taught, [])]
            told = [one for one in dict.fromkeys(told) if one]
            text = (f"{article(word)} {word}: a kind of "
                    f"{', '.join(parents)}, as you taught me" if parents else
                    f"{article(word)} {word}: a kind taught here")
            if told:
                text += "; and you taught me " + "; ".join(
                    f"“{one}”" for one in told)
            turn.answer = {"outcome": "retrieved", "source": "taught",
                           "text": text}
            return
        if node is None or self.definitions is None:
            self._generic(reading, turn)
            return
        known = self.definitions.has(node)
        if not known:
            turn.asked = reading.said
            turn.run = self._run(reading.said)
            self._learn_definition(node)
        entry = self.definitions.entry(node)
        if entry is None:
            outcome, headline, trust = summary_of(turn.run)
            turn.answer = {"outcome": outcome, "source": "kind",
                           "text": headline + (f" ({trust})" if trust
                                               else "")}
            return
        when = ("from what I read in its definition before" if known else
                "read from its definition just now, and kept")
        turn.answer = {"outcome": "retrieved", "source": "definition",
                       "text": f"{self._definition_text(entry, word)} — "
                               f"{when}"}

    def _against_definition(self, referent: Referent, relation: str,
                            obj: str) -> str:
        """What was just told of one of them that its kind's definition
        rules out: `it is old`, said of a kitten, a young domestic cat.

        Told still wins -- R3 answers from what you said -- but a definition
        is not a tendency, so it is said out loud rather than stored as one
        more exception."""
        if self.definitions is None:
            return ""
        opposite = rules.NEGATIONS.get(relation) or rules.POSITIVES.get(
            relation)
        wanted = self._predicate(obj[3:] if obj.startswith("no ") else obj)
        for node, distance, _ in self.memory.reasoner.ascend(referent.id):
            if not distance or self.memory.episodic_only(node):
                continue
            for fact in self.definitions.facts(node):
                have = self._predicate(fact.object)
                differ = [index for index, (one, other)
                          in enumerate(zip(have, wanted)) if one != other]
                clash = ((opposite and fact.relation == opposite
                          and have == wanted)
                         or (fact.relation == relation
                             and len(have) == len(wanted) and len(differ) == 1
                             and self._opposite(have[differ[0]],
                                                wanted[differ[0]])))
                if clash:
                    entry = self.definitions.entry(node) or {}
                    return (f"; that goes against the definition of "
                            f"{name_of(node)}, “{entry.get('gloss', '')}”, "
                            f"which says it {says(fact.relation, fact.object)}"
                            f" — kept as you said it, and an exception to "
                            f"what makes it one")
        return ""

    def _compound(self, reading: Reading, turn: Turn) -> None:
        turn.answer = {
            "outcome": "unknown", "source": "conversation",
            "text": ("that reads as more than one claim, and the parse could "
                     "not find where one ends and the next begins, so nothing "
                     "was stored — say them one at a time")}

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
            if walk.verdict == "VERIFIED" and self._taught_through(walk):
                turn.answer = {
                    "outcome": "verified", "source": "taught",
                    "text": (f"yes — you already taught me that {kind} is "
                             f"{article(obj)} {obj}")}
                return
            if walk.verdict == "VERIFIED":
                turn.answer = {
                    "outcome": "verified", "source": "kind",
                    "text": (f"yes — {kind} is already {article(obj)} {obj}; "
                             f"R1 walks there without being taught")}
                return
            parent = self.memory.kind_node(obj, self.asker.sense(obj))
            self.memory.relate(node, parent, reading.said)
            text = (f"taught: {kind} is a kind of {obj} — {name_of(node)} "
                    f"now sits under {parent}, {self._kept()}")
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
        stored = (f"taught: {relation} “{obj}” on {name_of(node)}, "
                  f"{self._kept()}")
        if new_kind:
            text = (f"{stored} — {word} is a kind taught here, so nothing "
                    f"in the store bears on it")
        else:
            turn.run = self._run(question)
            outcome, _, _ = summary_of(turn.run)
            self.memory.judge(node, relation, obj, outcome)
            if outcome in WORD and (WORD[outcome] == "yes") != reading.holds:
                text = (f"{stored} — v688 answers “{question}” {outcome} "
                        f"from the store, and in this conversation what you "
                        f"taught answers first for every {word}")
            elif outcome in WORD:
                text = f"{stored}, as the store already has it"
            else:
                text = f"{stored} — new: the store settles nothing about it"
        turn.answer = {"outcome": "noted", "source": "taught", "text": text}

    def _kept(self) -> str:
        """Where taught knowledge goes, as the reply says it."""
        return ("kept in this example only" if self.example
                else "kept in long-term memory")

    # -- what was said that bears on a question without answering it ------
    def _predicate(self, text: str) -> list[str]:
        """`shrink in cold temperatures` -> shrink, in, cold, temperature."""
        return [self.asker.lemma(word) for word in (text or "").lower().split()
                if word not in ARTICLES]

    def _related(self, walk, relation, target):
        """(the opposite of the question, [what comes close]), among what was
        told or taught on this walk.

        Opposite: the same predicate with its head word replaced by a WordNet
        antonym -- `shrink in cold temperatures` against `expand in cold
        temperatures`. Doing one under the same condition is not doing the
        other, so it answers no. An antonym anywhere else -- `expand in hot
        temperatures` against `expand in cold` -- is a different condition
        and answers nothing. Close: sharing a content word, quoted beside
        whatever answers, because `do testicles shrink` should say what was
        taught about them even when the qualification keeps it from being a
        yes (R28).
        """
        if not relation or relation == "is_a" or not target:
            return None, []
        wanted = self._predicate(target)
        family = set(rules.family(relation))
        contrary, near = None, []
        for node in walk.chain:
            for fact in self.memory.facts.get(node, []):
                if (fact.source != TOLD or fact.relation not in family
                        or (node, fact.relation, fact.object)
                        in self.memory.hidden):
                    continue
                have = self._predicate(fact.object)
                if have == wanted:
                    continue
                if len(have) == len(wanted):
                    differ = [index for index, (one, other)
                              in enumerate(zip(have, wanted)) if one != other]
                    if (differ == [0] and relation in rules.POSITIVES
                            and self._opposite(have[0], wanted[0])):
                        contrary = contrary or (fact, have[0], wanted[0])
                        continue
                if set(have) & set(wanted):
                    near.append(fact)
        return contrary, near

    def _opposite(self, one: str, other: str) -> bool:
        return (other in self.asker.antonyms(one)
                or one in self.asker.antonyms(other))

    def _contrary(self, contrary, turn: Turn) -> None:
        fact, had, asked = contrary
        key = (fact.concept, fact.relation, fact.object)
        individual = fact.concept in self.memory.individuals
        earlier = self.memory.learned_earlier(key)
        verb = "told" if individual else "taught"
        when = " in an earlier conversation" if earlier else ""
        who = self.discourse.by_id(fact.concept) if individual else None
        turn.answer = {
            "outcome": "denied",
            "source": ("told" if individual else
                       "learned" if earlier else "taught"),
            "text": (f"no — you {verb} me{when}: "
                     f"“{self.memory.said.get(key, '')}”, recorded of "
                     f"{self.discourse.describe(who) if who else name_of(fact.concept)}"
                     f" as {fact.relation} "
                     f"“{fact.object}”. “{had}” is the opposite of “{asked}” "
                     f"in WordNet, and the rest of it is the same")}

    def _near_note(self, near) -> str:
        said = [self.memory.said.get((fact.concept, fact.relation,
                                      fact.object), "") for fact in near]
        said = [one for one in dict.fromkeys(said) if one]
        if not said:
            return ""
        return ("; what you said that comes close: "
                + "; ".join(f"“{one}”" for one in said))

    # -- telling individuals -----------------------------------------------
    def _remember(self, referent: Referent, aux, rest, holds: bool,
                  said: str, turn: Turn,
                  other: Referent | None = None,
                  part: Reading | None = None) -> str:
        relation, obj, question, mode = self._relation(
            aux, rest, referent.kind, holds)
        turn.asked = question
        described = self.discourse.describe(referent)
        kind_node = self._kind_node(referent)
        store_kind = bool(kind_node) and not self.memory.episodic_only(
            kind_node)
        if relation in (None, "is_a"):
            if store_kind:
                turn.run = self._run(question)
            return (f"not something I can store about {described}: v687 "
                    f"reads “{question}” as no relation it keeps")
        when = self.story.before_telling(
            relation, mode, aux, part.rest if part is not None else rest,
            holds)
        self.memory.tell(referent.id, relation, obj, said, mode,
                         bound=other.id if other else None, when=when)
        placed = self.story.narrate(part, referent, other, relation, obj,
                                    holds, said)
        stored = f"stored {relation} “{obj}”"
        if other is not None:
            stored += f", about {self.discourse.describe(other)}"
        stored += self._against_definition(referent, relation, obj)
        kind = self._kind(referent)
        if store_kind:
            turn.run = self._run(question)
            outcome, _, _ = summary_of(turn.run)
            self.memory.judge(referent.id, relation, obj, outcome)
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
        if placed:
            text += f"; {placed}"
        text += self.story.told_note(when)
        notes = [self._carry(referent)] + [
            self._carry(one) for one in self._carried_by(referent)]
        carried = "; ".join(note for note in notes if note)
        return text + (f"; {carried}" if carried else "")

    def _carry(self, referent: Referent) -> str:
        """E2, recomputed from what is told now.

        What it was seen doing while carried by something that does it was
        the carrier's doing, and is withdrawn. What was withdrawn comes back
        once nothing carrying it does the thing -- `the airplane couldn't
        fly` -- because then the doing was its own after all.
        """
        node = referent.id
        facts = list(self.memory.facts.get(node, []))
        carriers = [(fact.object, one)
                    for fact in facts if fact.relation == "at_location"
                    for one in (sorted(self.memory.bound.get(
                        (node, fact.relation, fact.object), ())) or [None])]
        described = self.discourse.describe(referent)
        notes = []
        for fact in facts:
            if not (fact.relation == "capable_of"
                    and self.memory.mode.get((node, fact.relation,
                                              fact.object)) == "does"):
                continue
            # E2 within one time: being in an airplane yesterday explains
            # nothing about flying today.
            carrier = next(((word, one) for word, one in carriers
                            if self.story.same_time(node, fact.object, word)
                            and self._does(word, fact.object, one)), None)
            if carrier is None:
                continue
            withdrawal = self.memory.withdraw(node, fact.relation,
                                              fact.object, *carrier)
            who = self._carrier(*carrier)
            notes.append(
                f"E2: {who} does “{fact.object}”, so “{withdrawal.said}” was "
                f"{who}'s doing — capable_of “{fact.object}” is withdrawn "
                f"from {described} and kept as {CARRIED}")
        for fact in facts:
            if fact.relation != CARRIED or any(
                    self.story.same_time(node, fact.object, word)
                    and self._does(word, fact.object, one)
                    for word, one in carriers):
                continue
            withdrawal = self.memory.restore(node, fact.object)
            if withdrawal is None:
                continue
            who = self._carrier(withdrawal.carrier, withdrawal.carrier_id)
            notes.append(
                f"E2 undone: {who} does not “{fact.object}” after all, so "
                f"“{withdrawal.said}” was {described}'s own doing — "
                f"capable_of “{fact.object}” is restored")
        return "; ".join(notes)

    def _carrier(self, word: str, individual: str | None) -> str:
        found = self.discourse.by_id(individual) if individual else None
        return (self.discourse.describe(found) if found is not None
                else f"{article(word)} {word}")

    def _carried_by(self, carrier: Referent) -> list[Referent]:
        """Everyone told to be in or on this one."""
        return [one for one in self.discourse.everyone()
                if one.id != carrier.id and any(
                    fact.relation == "at_location"
                    and carrier.id in self.memory.bound.get(
                        (one.id, fact.relation, fact.object), ())
                    for fact in self.memory.facts.get(one.id, []))]

    def _does(self, carrier: str, action: str,
              individual: str | None = None) -> bool:
        """Does the carrier do this? v687's walk, then v688 on the kind.

        A carrier that is one of the conversation's individuals is walked
        from itself, so what was told of it -- `the airplane couldn't fly` --
        comes before what airplanes do.
        """
        if individual in self.memory.individuals:
            node = individual
            kind = (self.memory.edges.get(individual) or [None])[0]
            carrier = self.memory.words.get(individual) or carrier
        else:
            node = kind = (self.asker.sense(carrier)
                           or self.memory.kinds.get(carrier))
        if not node:
            return False
        walk = self.memory.reasoner.verify(node, "capable_of", action,
                                           self.asker.matcher)
        if (walk.verdict != "UNKNOWN" or not kind
                or self.memory.episodic_only(kind)):
            return walk.verdict == "VERIFIED"
        run = self._run(f"can {article(carrier)} {carrier} {action}")
        return summary_of(run)[0] == "verified"

    def _introduce(self, reading: Reading, turn: Turn) -> None:
        self.story.introduced(reading)
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
            rest, other = self._bind(reading.relative, referent, turn)
            if rest is None:
                text += f" — and nothing more, because {turn.binding.how}"
            else:
                text += " — " + self._remember(
                    referent, reading.relative.aux, rest,
                    reading.relative.holds, reading.said, turn, other,
                    part=reading.relative)
        turn.answer = {"outcome": "noted", "source": "conversation",
                       "text": text}

    def _tell_each(self, reading: Reading, turn: Turn) -> None:
        """`Mary and Daniel went to the kitchen`, `then they went to the
        hallway`: told of each of them, as happening together.

        `they` with no one talked about together is not about individuals:
        it is the kind named last, and v688's (`_they`).
        """
        found = self.discourse.resolve_all(reading.mention)
        if not found:
            self._generic(replace(reading, act="generic", mention=None), turn)
            return
        missing = next((one for one in found if one.referent is None), None)
        if missing is not None:
            turn.resolution = missing
            turn.answer = {"outcome": "which", "source": "conversation",
                           "text": missing.how}
            return
        when, replies = self._when, []
        for index, resolution in enumerate(found):
            referent = resolution.referent
            one = replace(reading, mention=Mention(
                "individual", text=reading.mention.text, name=referent.id))
            # Together: each after the first happened during it (T1), not
            # after it.
            if index:
                self._when = replace(when, link="during", anchor="",
                                     relation="")
            turn.answer = {}
            self._tell(one, turn)
            replies.append(f"{self.discourse.describe(referent)}: "
                           f"{turn.answer.get('text', '')}")
        self._when = when
        turn.resolution = found[0]
        turn.answer = {"outcome": "noted", "source": "told",
                       "text": "; ".join(replies)}

    def _tell(self, reading: Reading, turn: Turn) -> None:
        if reading.mention is not None and reading.mention.form in (
                "group", "plural"):
            self._tell_each(reading, turn)
            return
        # A statement: a description nothing here fits introduces what it
        # describes (`the blue square is to the left of the triangle`).
        referent = self._resolve(reading, turn, described=True)
        if referent is None:
            return
        narrowed = self._narrow(referent, reading)
        if narrowed:
            turn.answer = {"outcome": "noted", "source": "conversation",
                           "text": narrowed}
            return
        rest, other = self._bind(reading, referent, turn)
        if rest is None:
            return
        turn.answer = {"outcome": "noted", "source": "told",
                       "text": "noted — " + self._remember(
                           referent, reading.aux, rest,
                           reading.holds, reading.said, turn, other,
                           part=reading)}

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
               "I am called" if referent.addressee else
               f"{self.discourse.describe(referent, named=False)} is called")
        text = f"noted: {who} {reading.name}"
        if before and before != reading.name:
            text += f" (it was {before})"
        turn.answer = {"outcome": "noted", "source": "told", "text": text}

    # -- asking ------------------------------------------------------------
    def _from_walk(self, walk, subject: str, turn: Turn,
                   other: Referent | None = None) -> bool:
        """Answer from a told or taught fact the walk used, if it used one."""
        told = [fact for fact in walk.evidence if fact.source == TOLD]
        if walk.verdict not in OUTCOME:
            return False
        if not told:
            return self._from_definition(walk, turn)
        fact = told[0]
        outcome = OUTCOME[walk.verdict]
        said = self.memory.said.get((fact.concept, fact.relation,
                                     fact.object), "")
        where = rule_of(walk, fact)
        bound = self.memory.bound.get((fact.concept, fact.relation,
                                       fact.object), set())
        if other is not None and bound and other.id not in bound:
            named = " and ".join(
                self.discourse.describe(one) for one in
                map(self.discourse.by_id, sorted(bound)) if one is not None)
            turn.answer = {
                "outcome": "unknown", "source": "told",
                "text": (f"not told — you told me “{said}”, and that was "
                         f"about {named}, not "
                         f"{self.discourse.describe(other)}")}
            return True
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
            earlier = self.memory.learned_earlier(
                (fact.concept, fact.relation, fact.object))
            when = " in an earlier conversation" if earlier else ""
            text = (f"{WORD[outcome]} — you taught me{when}: “{said}”, "
                    f"recorded of {name_of(fact.concept)}")
            if where:
                text += f" ({where})"
            turn.answer = {"outcome": outcome,
                           "source": "learned" if earlier else "taught",
                           "text": text}
        return True

    def _from_definition(self, walk, turn: Turn) -> bool:
        """Answer from a fact the walk read out of a definition."""
        defined = [fact for fact in walk.evidence if fact.source == DEFINED]
        if not defined:
            return False
        fact = defined[0]
        entry = (self.definitions.entry(fact.concept)
                 if self.definitions is not None else None) or {}
        outcome = OUTCOME[walk.verdict]
        where = rule_of(walk, fact)
        text = (f"{WORD[outcome]} — by the definition of "
                f"{name_of(fact.concept)}, “{entry.get('gloss', '')}”: it "
                f"{says(fact.relation, fact.object)}")
        if where:
            text += f" ({where})"
        turn.answer = {"outcome": outcome, "source": "definition",
                       "text": text}
        return True

    # -- wh-questions about this conversation's individuals ----------------
    def _individuals(self, kind: str = "") -> list[Referent]:
        """The individuals here of a kind, found as a description is: by
        walking the episodic trie for `is_a kind`, which every kind above an
        individual is stored as."""
        people = [one for one in self.discourse.referents if not one.apart]
        if not kind:
            return people
        found = set(self.memory.identify({f"is_a {kind}"}).candidates)
        return [one for one in people if one.id in found]

    def _names(self, people: list[Referent]) -> str:
        names = [self.discourse.describe(one) for one in people]
        return (", ".join(names[:-1]) + " and " + names[-1]
                if len(names) > 1 else "".join(names))

    def _here(self, reading: Reading, turn: Turn) -> Referent | None:
        """The individual a question is about. A question never puts one
        down: asked `where is the dog` with no dog, there is no dog."""
        mention = reading.mention
        if (mention.form in ("definite", "demonstrative", "possessive")
                and mention.kind and not self._individuals(mention.kind)):
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"no {mention.kind} has come up"}
            return None
        return self._resolve(reading, turn)

    def _object_here(self, reading: Reading, turn: Turn, exclude=frozenset()):
        """(rest, object) for a question: the object resolved if it is one
        of the individuals here, and left a kind otherwise. A question never
        puts one down: `is the pig in an airplane` is about airplanes."""
        found = reading.obj
        if found is None or found.form in ("indefinite", "another", "kind") \
                or (found.kind and not self._individuals(found.kind)):
            return list(reading.rest), None
        turn.binding = self.discourse.resolve(found, exclude=set(exclude),
                                              weight=OBJECT_WEIGHT)
        other = turn.binding.referent
        if other is None:
            turn.answer = {"outcome": "which", "source": "conversation",
                           "text": turn.binding.how}
            return None, None
        return (list(reading.rest[:reading.obj_at])
                + [article(other.kind), other.kind]), other

    def _told_of(self, relation: str, obj: str, other=None,
                 among=None) -> list[Referent]:
        """Who was told this: the trie walked for `relation object`, and the
        object's individual checked where the question named one."""
        found = set(self.memory.identify({f"{relation} {obj.lower()}"})
                    .candidates)
        people = [one for one in self.discourse.everyone()
                  if one.id in found and (among is None or one in among)]
        if other is not None:
            people = [one for one in people if other.id in self.memory.bound
                      .get((one.id, relation, obj), ())]
        return people

    def _quotes(self, people, relation: str, obj: str) -> str:
        said = dict.fromkeys(self.memory.said.get((one.id, relation, obj), "")
                             for one in people)
        return "; ".join(f"“{one}”" for one in said if one)

    def _how_many(self, reading: Reading, turn: Turn) -> None:
        kind = reading.mention.kind if reading.mention else ""
        ones = self._individuals(kind)
        if not ones:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"none — no {kind or 'one'} has come up"}
            return
        turn.answer = {"outcome": "retrieved", "source": "conversation",
                       "text": f"{len(ones)} — {self._names(ones)}"}

    def _which(self, reading: Reading, turn: Turn) -> None:
        kind = reading.mention.kind if reading.mention else ""
        ones = self._individuals(kind)
        if not ones:
            self._generic(reading, turn)        # `which birds cannot fly`
            return
        rest, other = self._object_here(reading, turn)
        if rest is None:
            return
        relation, obj, question, _ = self._relation(
            reading.aux, rest, kind or ones[0].kind, reading.holds)
        if relation is None:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"v687 reads “{question}” as no relation "
                                   f"it keeps"}
            return
        told = self._told_of(relation, obj, other, ones)
        if told:
            quotes = self._quotes(told, relation, obj)
            turn.answer = {"outcome": "retrieved", "source": "told",
                           "text": self._names(told) + (
                               f" — you told me {quotes}" if quotes else "")}
            return
        # Nothing told picks one out; what each inherits may. A quality does
        # not descend to one of them (E1), so that walk says nothing.
        walked = [one for one in ones
                  if self._walk(one.id, relation, obj).verdict == "VERIFIED"]
        if walked:
            turn.answer = {"outcome": "retrieved", "source": "kind",
                           "text": f"nothing was told of any of them, but by "
                                   f"what their kinds do: "
                                   f"{self._names(walked)}"}
            return
        said = " ".join(reading.rest)
        turn.answer = {"outcome": "unknown", "source": "conversation",
                       "text": f"no {kind or 'one'} here was said to "
                               f"{'be ' if reading.aux in COPULA else ''}"
                               f"{said}"}

    def _who(self, reading: Reading, turn: Turn) -> None:
        if not any(self.memory.facts.get(one.id) or self.memory.labels.get(
                one.id) for one in self.discourse.referents):
            self._generic(reading, turn)    # `who invented the telephone`
            return
        if (reading.rest and self.asker.lemma(reading.rest[0]) == "own"
                and reading.obj is not None):
            _, owned = self._object_here(reading, turn)
            if owned is None:
                if not turn.answer:
                    self._generic(reading, turn)
                return
            owner = next((label.split(" ", 1)[1] for label in
                          self.memory.labels.get(owned.id, ())
                          if label.startswith("owner ")), None)
            found = self.discourse.by_id(owner) if owner else None
            described = self.discourse.describe(owned)
            turn.answer = (
                {"outcome": "retrieved", "source": "told",
                 "text": f"{self.discourse.describe(found)} — {described} is "
                         f"{'yours' if found.speaker else 'theirs'}"}
                if found is not None else
                {"outcome": "unknown", "source": "conversation",
                 "text": f"nobody was said to own {described}"})
            return
        if self.story.who(reading, turn):
            return
        rest, other = self._object_here(reading, turn)
        if rest is None:
            return
        relation, obj, question, _ = self._relation(
            reading.aux, rest, self.discourse.referents[0].kind, reading.holds)
        if relation is None:
            self._generic(reading, turn)
            return
        told = self._told_of(relation, obj, other)
        if told:
            quotes = self._quotes(told, relation, obj)
            turn.answer = {"outcome": "retrieved", "source": "told",
                           "text": self._names(told) + (
                               f" — you told me {quotes}" if quotes else "")}
            return
        # `who went to the kitchen`: said to go there, not to went.
        said = [self.asker.lemma(rest[0])] + list(rest[1:]) if rest else rest
        turn.answer = {"outcome": "unknown", "source": "conversation",
                       "text": f"nobody here was said to {' '.join(said)}"}

    def _where(self, reading: Reading, turn: Turn) -> None:
        referent = self._here(reading, turn)
        if referent is None:
            return
        if self.story.where(reading, referent, turn):
            return
        described = self.discourse.describe(referent)
        places = []
        for fact in self.memory.facts.get(referent.id, []):
            if fact.relation != "at_location":
                continue
            key = (referent.id, fact.relation, fact.object)
            there = [self.discourse.by_id(one)
                     for one in sorted(self.memory.bound.get(key, ()))]
            where = self._names([one for one in there if one]) or fact.object
            said = self.memory.said.get(key, "")
            places.append(where + (f" — you told me “{said}”" if said else ""))
        turn.answer = (
            {"outcome": "retrieved", "source": "told",
             "text": f"{described}: " + "; ".join(places)} if places else
            {"outcome": "unknown", "source": "conversation",
             "text": f"nothing was said about where {described} is"})

    def _what_did(self, reading: Reading, turn: Turn) -> None:
        referent = self._here(reading, turn)
        if referent is None:
            return
        if self.story.what_did(reading, referent, turn):
            return
        described = self.discourse.describe(referent)
        verb = self.asker.lemma(reading.rest[0])
        found = []
        for fact in self.memory.facts.get(referent.id, []):
            words = fact.object.split()
            if not words or self.asker.lemma(words[0]) != verb:
                continue
            key = (referent.id, fact.relation, fact.object)
            bound = [self.discourse.by_id(one)
                     for one in sorted(self.memory.bound.get(key, ()))]
            what = (self._names([one for one in bound if one])
                    or " ".join(words[1:]) or fact.object)
            said = self.memory.said.get(key, "")
            found.append(what + (f" — you told me “{said}”" if said else ""))
        turn.answer = (
            {"outcome": "retrieved", "source": "told",
             "text": "; ".join(found)} if found else
            {"outcome": "unknown", "source": "conversation",
             "text": f"nothing was told of {described} that it would "
                     f"{' '.join(reading.rest)}"})

    def _about(self, reading: Reading, turn: Turn) -> None:
        """Everything told of one individual, and for what it can do or has,
        what its kind can do or has as well."""
        referent = self._here(reading, turn)
        if referent is None:
            return
        described = self.discourse.describe(referent)
        wanted = {("do", True): ("capable_of",),
                  ("do", False): ("not_capable_of",),
                  ("have", True): ("has_a", "has_part")}.get(
            (reading.rest[0] if reading.rest else "", reading.holds))
        told = [self.memory.said.get((referent.id, fact.relation,
                                      fact.object), "")
                for fact in self.memory.facts.get(referent.id, [])
                if wanted is None or fact.relation in wanted]
        told = [one for one in dict.fromkeys(told) if one]
        parts = [self._kind(referent)]
        if referent.name:
            parts.append(f"called {referent.name}")
        if any(label == "owner you"
               for label in self.memory.labels.get(referent.id, ())):
            parts.append("yours")
        text = f"{described}: " + ", ".join(parts)
        if told:
            text += "; you told me " + "; ".join(f"“{one}”" for one in told)
        elif wanted is not None and not reading.holds:
            # What one cannot do is only ever told: its kind being able to do
            # a thing says nothing of what this one cannot.
            text += (f"; nothing it cannot "
                     f"{reading.rest[0] if reading.rest else 'do'} was told "
                     f"of {described}")
        node = self._kind_node(referent)
        if (wanted is not None and reading.holds and node
                and not self.memory.episodic_only(node)):
            who = f"{article(referent.kind)} {referent.kind}"
            turn.asked = (f"what can {who} do" if reading.rest == ["do"]
                          else f"what does {who} have")
            turn.run = self._run(turn.asked)
            _, headline, _ = summary_of(turn.run)
            if headline:
                text += f"; and as {who}: {headline}"
        turn.answer = {"outcome": "retrieved",
                       "source": "told" if told else "kind", "text": text}

    def _happened(self, reading: Reading, turn: Turn) -> None:
        """What was told, in the order it was told: of one individual's doings,
        or of everyone. What happened in the story, and in its order, is the
        story's (`story.py`); this is telling time."""
        if self.story.happened(reading, turn):
            return
        referent = None
        if reading.mention is not None:
            referent = self._here(reading, turn)
            if referent is None:
                return
        events = []
        for (node, relation, _), said in self.memory.said.own.items():
            if node not in self.memory.individuals:
                continue
            if referent is not None and (node != referent.id or relation not in
                                         ("capable_of", "not_capable_of",
                                          DID_NOT)):
                continue
            events.append(said)
        events = list(dict.fromkeys(one for one in events if one))
        note = (self.story.order_note()
                if referent is None and reading.rest[:1] != ["told"] else "")
        if not events:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": "nothing has been told of anyone here yet"
                                   if referent is None else
                                   f"nothing was told of what "
                                   f"{self.discourse.describe(referent)} did"}
            return
        which = [word for word in reading.rest if word != "told"][:1]
        if which == ["first"]:
            events, lead = events[:1], "first: "
        elif which in (["last"], ["next"]):
            events, lead = events[-1:], "last: "
        else:
            lead = "in the order you told me: "
        turn.answer = {"outcome": "retrieved", "source": "told",
                       "text": lead + "; ".join(f"“{one}”" for one in events)
                               + (note if lead.startswith("in the order")
                                  else "")}

    #: Where an answer came from, said as the grounds for it.
    GROUNDS = {"told": "from what you told me",
               "taught": "from what you taught me",
               "learned": "from what you taught me in an earlier conversation",
               "kind": "from what its kind is recorded as, by v688",
               "tendency": "from its kind, as a tendency",
               "definition": "from its definition",
               "conversation": "from this conversation"}

    def _meta(self, reading: Reading, turn: Turn) -> None:
        """`how do you know that`, `how sure are you`: the last answer's
        grounds -- where it came from, and what v688 made of it."""
        last = next((one for one in reversed(self.turns)
                     if one.answer and not (
                         one.reading is not None
                         and ("grounds", "answer") in one.reading.cells)),
                    None)
        if last is None:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": "nothing has been answered yet"}
            return
        grounds = self.GROUNDS.get(last.answer.get("source"),
                                   last.answer.get("source") or "somewhere")
        text = f"“{last.said}” was answered {grounds}"
        summary = (last.run or {}).get("summary") or {}
        rated = [summary.get("trust")] if summary.get("trust") else []
        if summary.get("confidence") is not None:
            rated.append(f"{round(summary['confidence'] * 100)}% confident")
        if rated:
            text += f"; v688 rates it {', '.join(rated)}"
        reasons = (summary.get("lines") or [])[1:3]
        if reasons:
            text += "; " + "; ".join(reasons)
        walk = last.walk or {}
        steps = walk.get("steps") or []
        if steps and not reasons:
            text += f"; v687's walk: {steps[-1].get('rule')} — " \
                    f"{steps[-1].get('detail')}"
        turn.answer = {"outcome": "retrieved", "source": "conversation",
                       "text": text}

    def _ellipsis(self, reading: Reading, turn: Turn) -> None:
        """`what about a cat`: the last question, asked of another kind."""
        last = getattr(self, "_last_question", None)
        kind = reading.mention.kind if reading.mention else ""
        before = last[0] if last else None
        old = before.mention.kind if before is not None and before.mention \
            else ""
        if last is None or not kind or not old:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": f"what about {article(kind)} {kind}? no "
                                   f"question about a kind has been asked "
                                   f"yet" if kind else "what about what?"}
            return
        asked = last[1]
        phrase = f"{article(old)} {old}"
        question = (asked.replace(phrase, f"{article(kind)} {kind}", 1)
                    if phrase in asked else asked.replace(old, kind, 1))
        again = read(question, Taught(self.asker, self.memory.kinds),
                     self.discourse.names())
        again.said = question
        getattr(self, "_" + again.act, self._generic)(again, turn)
        if turn.answer.get("text"):
            turn.answer["text"] = (f"asked as “{question}”: "
                                   f"{turn.answer['text']}")

    def _why(self, reading: Reading, turn: Turn) -> None:
        """What a yes or no about one individual rests on.

        v687's walk already says it -- what you told me of this one, what you
        taught me of its kind, or the kind it inherits from -- so a why about
        an individual is its yes or no, answered the same way; where the walk
        passes to the kind, v688 is asked the kind's why, which says what the
        kind's answer rests on. A bare `why` asks it of the last question.
        """
        from research.v688.rephrase import AUX as ASKS

        if reading.mention is None:
            last = getattr(self, "_last_question", None)
            if last is None or last[1].split()[:1] == [] or (
                    last[1].split()[0] not in ASKS):
                turn.answer = {"outcome": "unknown", "source": "conversation",
                               "text": "why what? no yes-or-no question has "
                                       "been asked yet"}
                return
            reading, asked = last
            if reading.act != "ask":
                turn.asked = asked
                turn.run = self._run(f"why {asked}")
                outcome, headline, trust = summary_of(turn.run)
                turn.answer = {"outcome": outcome, "source": "kind",
                               "text": headline + (f" ({trust})" if trust
                                                   else "")}
                return
        # `why did Sumit go to the bedroom`: it was told that he did, so what
        # is asked is what moved him to it, not whether he did.
        if reading.aux == "did" and self._because(reading, turn):
            return
        self._why_asked = True
        try:
            self._ask(reading, turn)
        finally:
            self._why_asked = False

    def _ask(self, reading: Reading, turn: Turn) -> None:
        """A yes or no about one individual, run by the executive
        (`research/v687/executive.py`). Each step is an operator: its
        condition is what it needs to have been found already, the order
        written is its utility, and working memory `m` holds what the
        steps before it found. An operator that settles the question -- or
        cannot go on, having said why -- answers; the others write slots."""
        m: dict = {"why": getattr(self, "_why_asked", False)}

        def they(_) -> str:
            # `can they swim`, with no one talked about together: the kind.
            self._generic(replace(reading, act="generic", mention=None), turn)
            return ANSWERED

        def resolve(_) -> str:
            m["referent"] = self._resolve(reading, turn)
            return ANSWERED if m["referent"] is None else CONTINUE

        def bind(_) -> str:
            rest, other = self._object_here(reading, turn,
                                            exclude={m["referent"].id})
            if rest is None:
                return ANSWERED
            m["rest"], m["other"] = rest, other
            return CONTINUE

        def relate(_) -> str:
            referent = m["referent"]
            relation, target, question, mode = self._relation(
                reading.aux, m["rest"], referent.kind, True)
            turn.asked = question
            m.update(question=question, target=target, mode=mode,
                     described=self.discourse.describe(referent),
                     kind=self._kind(referent),
                     taught_kind=self.memory.episodic_only(
                         self._kind_node(referent)))
            if relation is None:
                turn.answer = {
                    "outcome": "unknown", "source": "conversation",
                    "text": f"v687 reads “{question}” as no relation it keeps"}
                return ANSWERED
            m["relation"] = relation
            return CONTINUE

        def compared(_) -> str:
            # A relation to another individual here: S2 and S3, not the walk.
            return (ANSWERED if self._compared(
                replace(reading, rest=m["rest"]), m["referent"], m["other"],
                turn) else DECLINED)

        def in_story(_) -> str:
            return (ANSWERED if self.story.ask(
                reading, m["referent"], m["other"], m["relation"],
                m["target"], turn) else DECLINED)

        def walk(_) -> str:
            self.memory.hidden = self.story.hidden(reading)
            m["walk"] = self._walk(m["referent"].id, m["relation"],
                                   m["target"])
            turn.walk = walk_of(m["walk"])
            return CONTINUE

        def taxonomy(_) -> str:
            outcome = OUTCOME.get(m["walk"].verdict, "unknown")
            turn.answer = {
                "outcome": outcome, "source": "kind",
                "text": (f"{WORD.get(outcome, 'not settled')} — walking up "
                         f"from {m['described']}, the taxonomy answers "
                         f"{m['walk'].verdict} (R1)")}
            return ANSWERED

        def from_walk(_) -> str:
            return (ANSWERED if self._from_walk(m["walk"], m["described"],
                                                turn, m["other"])
                    else DECLINED)

        def did_not(_) -> str:
            referent = m["referent"]
            for fact in self.memory.facts.get(referent.id, []):
                if (fact.relation == DID_NOT
                        and (referent.id, fact.relation, fact.object)
                        not in self.memory.hidden
                        and self.asker.matcher(fact.object, m["target"])):
                    said = self.memory.said.get(
                        (referent.id, fact.relation, fact.object), "")
                    turn.answer = {"outcome": "denied", "source": "told",
                                   "text": f"no — you told me so: “{said}”"}
                    return ANSWERED
            return DECLINED

        def contrary(_) -> str:
            found, near = self._related(m["walk"], m["relation"], m["target"])
            if found is not None:
                self._contrary(found, turn)
                return ANSWERED
            m["note"] = self._near_note(near)
            m["e1"] = any(step.rule == "E1" for step in m["walk"].steps)
            return CONTINUE

        def taught_through() -> bool:
            return bool(m["taught_kind"] or (
                m["walk"].verdict in OUTCOME
                and self._taught_through(m["walk"])))

        def above(_) -> str:
            return (ANSWERED if self._ask_above(reading, m["walk"],
                                                m["described"], turn,
                                                m["rest"]) else DECLINED)

        def taught(_) -> str:
            found, e1, referent = m["walk"], m["e1"], m["referent"]
            outcome = "unknown" if e1 else OUTCOME.get(found.verdict,
                                                       "unknown")
            evidence = found.evidence[0] if found.evidence else None
            text = (f"{WORD.get(outcome, 'not settled')} — {m['described']} "
                    f"{be(referent)} {m['kind']}, and what was taught about "
                    f"it is the whole answer: v687's walk says "
                    f"{found.verdict}")
            if evidence is not None:
                text += (f", from {name_of(evidence.concept)} "
                         f"{evidence.relation} “{evidence.object}”")
            if e1:
                text += " — E1: a quality does not descend to one of them"
            turn.answer = {"outcome": outcome, "source": "taught",
                           "text": text + m["note"]}
            return ANSWERED

        def inherited(_) -> str:
            # T6: nothing told says this one did it, and what the kind does
            # is not what one of them did.
            return (ANSWERED if self.story.inherited(
                reading, m["referent"], m["question"], turn) else DECLINED)

        def of_the_kind(_) -> str:
            why, e1, question = m["why"], m["e1"], m["question"]
            referent, described, kind = (m["referent"], m["described"],
                                         m["kind"])
            note = m["note"]
            if why and not reading.holds:
                # `why can't it fly`: the premise is a denial, and the kind's
                # why has to be asked as one, or the answer says it does not
                # hold.
                head, _, tail = question.partition(" ")
                question_asked = f"why {DENIAL.get(head, head + ' not')} {tail}"
            else:
                question_asked = f"why {question}" if why else question
            turn.run = self._run(question_asked)
            outcome, headline, trust = summary_of(turn.run)
            v688 = f"v688 answers “{question}” {outcome}" + (
                f" ({trust})" if trust else "")
            if why and not e1:
                # Nothing was told of this one, so what its kind's answer
                # rests on is what this one's does.
                turn.answer = {
                    "outcome": outcome, "source": "kind",
                    "text": (f"nothing was told of {described}, so it is as "
                             f"{kind}: {headline}") + note}
            elif e1:
                turn.answer = {
                    "outcome": "unknown", "source": "tendency",
                    "text": (f"not known of {described} — E1: a quality does "
                             f"not descend from {referent.kind} to one of "
                             f"them. For {kind} in general, {v688}") + note}
            else:
                turn.answer = {
                    "outcome": outcome, "source": "kind",
                    "text": (f"{WORD.get(outcome, 'not settled')} — nothing "
                             f"was told of {described}, so v687's walk passes "
                             f"up to {referent.kind}, and {v688}") + note}
            return ANSWERED

        Executive([
            Operator("they, of a kind", they,
                     proposes=lambda _: (reading.mention is not None
                                         and reading.mention.form == "plural"
                                         and not self.discourse.group)),
            Operator("resolve", resolve),
            Operator("bind the object", bind,
                     proposes=lambda _: "referent" in m),
            Operator("read the relation", relate,
                     proposes=lambda _: "rest" in m),
            Operator("compared", compared, rule="S2",
                     proposes=lambda _: "relation" in m),
            Operator("in the story", in_story, rule="T3",
                     proposes=lambda _: m.get("relation") not in (None,
                                                                   "is_a")),
            Operator("walk", walk, rule="R1",
                     proposes=lambda _: "relation" in m),
            Operator("taxonomy", taxonomy, rule="R1",
                     proposes=lambda _: ("walk" in m
                                         and m["relation"] == "is_a")),
            Operator("from the walk", from_walk, rule="R3",
                     proposes=lambda _: "walk" in m),
            Operator("did not", did_not,
                     proposes=lambda _: ("walk" in m
                                         and m["relation"] == "capable_of"
                                         and m["mode"] == "does")),
            Operator("contrary", contrary, proposes=lambda _: "walk" in m),
            Operator("taught, above", above,
                     proposes=lambda _: ("note" in m and not m["e1"]
                                         and taught_through())),
            Operator("taught", taught,
                     proposes=lambda _: "note" in m and taught_through()),
            Operator("nothing by inheritance", inherited, rule="T6",
                     proposes=lambda _: ("note" in m and not m["why"]
                                         and m["relation"] == "capable_of")),
            Operator("the kind", of_the_kind, proposes=lambda _: "note" in m),
        ]).run(m)

    def _ask_above(self, reading: Reading, walk, subject: str,
                   turn: Turn, rest=None) -> bool:
        """Ask what a taught kind's walk could not settle of the nearest
        kind above it that the store has.

        On the real store `can a wemble breathe` walked from a wemble into
        animal and found only qualified rows (R28) and word-level ones too
        broad to inherit (R12) -- UNKNOWN, while v688, corroborating and
        asking its teacher, answers `can an animal breathe`. A taught kind
        inherits that answer as it inherits the store's facts, from the
        nearest place the store has anything to say.
        """
        row = next((fact for fact in walk.evidence
                    if fact.source != TOLD
                    and not self.memory.episodic_only(fact.concept)), None)
        above = row.concept if row is not None else next(
            (node for node in walk.chain
             if not self.memory.episodic_only(node)), None)
        if above is None:
            return False
        word = name_of(above)
        _, _, question, _ = self._relation(
            reading.aux, reading.rest if rest is None else rest, word, True)
        turn.run = self._run(question)
        outcome, _, trust = summary_of(turn.run)
        where = (f"the store's row “{row.relation} {row.object}” is on "
                 f"{word}, so v688 judges it there" if row is not None else
                 f"nothing taught settles it, so it is asked of {word}, the "
                 f"nearest kind above that the store has")
        text = (f"{WORD.get(outcome, 'not settled')} — for {subject}, {where}: "
                f"v688 answers “{question}” {outcome}")
        if trust:
            text += f" ({trust})"
        turn.answer = {"outcome": outcome, "source": "kind", "text": text}
        return True

    def _ask_name(self, reading: Reading, turn: Turn) -> None:
        referent = self._resolve(reading, turn)
        if referent is None:
            return
        whose = ("your" if referent.speaker else
                 "my" if referent.addressee else
                 f"{self.discourse.describe(referent, named=False)}'s")
        if referent.name:
            turn.answer = {"outcome": "retrieved", "source": "told",
                           "text": f"{whose} name is {referent.name}"}
        elif referent.addressee:
            turn.answer = {"outcome": "unknown", "source": "conversation",
                           "text": "nobody has given me a name"}
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

    def _taught_can(self, word: str, node: str, wanted: str,
                    turn: Turn) -> None:
        """`what can a wemble do`, for a kind the store has no word for:
        what was taught of it, and what its taught kind above can do."""
        relations = ("capable_of",) if wanted == "do" else ("has_a",
                                                            "has_part")
        told = [self.memory.said.get((node, fact.relation, fact.object), "")
                for fact in self.memory.facts.get(node, [])
                if fact.relation in relations]
        told = [one for one in dict.fromkeys(told) if one]
        who = f"{article(word)} {word}"
        text = (f"{who}: you taught me " + "; ".join(f"“{one}”" for one in told)
                if told else
                f"nothing was taught of what {who} "
                f"{'can do' if wanted == 'do' else 'has'}")
        above = next((one for one in self.memory.edges.get(node, [])
                      if not self.memory.episodic_only(one)), None)
        if above is not None:
            kind = f"{article(name_of(above))} {name_of(above)}"
            turn.asked = (f"what can {kind} do" if wanted == "do"
                          else f"what does {kind} have")
            turn.run = self._run(turn.asked)
            _, headline, _ = summary_of(turn.run)
            if headline:
                text += f"; and as {kind}: {headline}"
        turn.answer = {"outcome": "retrieved" if told else "unknown",
                       "source": "taught", "text": text}

    def _they(self, reading: Reading) -> str | None:
        """`can they bark`, when nothing here is `they` and a kind was named
        before it: the question, asked of one of that kind."""
        kind = getattr(self, "_last_kind", None)
        tokens = words(reading.said)
        if (not kind or reading.mention is not None or not tokens
                or not THEY & set(tokens)
                or tokens[0] not in AUX | QUESTION_WORDS):
            return None
        said: list[str] = []
        for token in tokens:
            if token not in THEY:
                said.append(token)
                continue
            # `can't they swim` asks what `can they swim` does.
            if len(said) >= 2 and said[-1] == "not" and said[-2] in AUX:
                said.pop()
            if said and said[-1] in SINGULAR:
                said[-1] = SINGULAR[said[-1]]
            said += [article(kind), kind]
        return " ".join(said)

    def _generic(self, reading: Reading, turn: Turn) -> None:
        """About a kind: from episodic memory if anything taught bears on
        it, otherwise v688's question, asked as said."""
        they = self._they(reading)
        if they is not None:
            again = read(they, Taught(self.asker, self.memory.kinds),
                         self.discourse.names())
            if again.act == "generic":
                self._generic(again, turn)
                turn.answer["text"] = (f"asked as “{they}”: "
                                       + (turn.answer.get("text") or ""))
                return
        words = reading.said.lower().rstrip("?").split()
        if (len(words) == 5 and words[0] == "what"
                and words[1] in ("can", "does", "do")
                and words[2] in ("a", "an") and words[4] in ("do", "have")):
            node = self.memory.kinds.get(words[3])
            if node is not None and self.asker.sense(words[3]) is None:
                self._taught_can(words[3], node, words[4], turn)
                return
        if (reading.mention is not None and reading.mention.kind
                and reading.rest and self._about_kind(reading, turn)):
            return
        turn.asked = reading.said
        turn.run = self._run(reading.said)
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
        told = any(fact.source in (TOLD, DEFINED) for fact in walk.evidence)
        contrary, near = self._related(walk, relation, target)
        note = self._near_note(near)
        if contrary is not None and not told:
            turn.asked = question
            turn.walk = walk_of(walk)
            self._contrary(contrary, turn)
            return True
        if not (new_kind or self._taught_through(walk) or told):
            if not note:
                return False        # nothing episodic bears on it
            turn.asked = question
            turn.run = self._run(question)
            outcome, headline, trust = summary_of(turn.run)
            turn.answer = {"outcome": outcome, "source": "kind",
                           "text": (headline + (f" ({trust})" if trust
                                                else "") + note)}
            return True
        turn.asked = question
        turn.walk = walk_of(walk)
        if self._from_walk(walk, f"{article(word)} {word}", turn):
            return True
        if relation != "is_a" and self._ask_above(
                reading, walk, f"{article(word)} {word}", turn):
            return True
        outcome = OUTCOME.get(walk.verdict, "unknown")
        evidence = walk.evidence[0] if walk.evidence else None
        text = (f"{WORD.get(outcome, 'not settled')} — by what was taught "
                f"about {word}, v687's walk says {walk.verdict}")
        if evidence is not None:
            text += (f", from {name_of(evidence.concept)} "
                     f"{evidence.relation} “{evidence.object}”")
        turn.answer = {"outcome": outcome, "source": "taught",
                       "text": text + note}
        return True

    def as_dict(self) -> dict:
        return {"turns": [turn.as_dict() for turn in self.turns],
                "discourse": self.discourse.as_dict(),
                "memory": self.memory_view()}
