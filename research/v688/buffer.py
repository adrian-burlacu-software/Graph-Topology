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

from dataclasses import dataclass, field

from . import attention
from .gap import Gap, Doubt, read_doubts, read_gap

#: A verdict that says yes, and one that says no. `INHERITED` is a yes that
#: names the ancestor it came from; `UNKNOWN` is neither, and treating it as a
#: no is the confusion the v687 audit was written to end.
POSITIVE = frozenset({"VERIFIED", "HELD", "INHERITED"})
NEGATIVE = frozenset({"CONTRADICTED", "DENIED"})

#: Parts of speech that name something worth attending to.
ATTENDED_POS = ("NOUN", "PROPN", "VERB", "ADJ", "INTJ")

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
        #: Gap and doubt questions that were generated but did not fit in a
        #: cycle. They are carried rather than dropped: a family check cut
        #: short by the pool size reports `1 of 4 deny it` about a family of
        #: seven, which is worse arithmetic than not checking at all.
        self.carried: list = []

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
            self.activation.bump(lemma, 0.6, self.cycle)
        if parse.subject:
            self.activation.bump(parse.subject.lower(), 1.0, self.cycle)
            if parse.subject.lower() not in found:
                found.insert(0, parse.subject.lower())
        for word in found:
            if word not in self.topics:
                self.topics.append(word)
        return found

    # -- recording ---------------------------------------------------------
    def record(self, answers) -> tuple[list[Gap], list[Doubt]]:
        """File a cycle's answers, and read what they left open."""
        fresh_gaps: list[Gap] = []
        fresh_doubts: list[Doubt] = []
        for answer in answers:
            self.answers[answer.question] = answer
            if answer.error:
                continue
            hole = read_gap(answer.payload, answer.question)
            if hole is not None and hole.blocker:
                key = (hole.kind, hole.blocker)
                if answer.origin == "curiosity":
                    # A curiosity question that finds nothing has found
                    # nothing. Chasing it turns `is a greeting furry` ->
                    # UNKNOWN into `what is a furry`, and the loop spends its
                    # workers defining the adjectives it invented.
                    self.seen_gaps.append(hole)
                elif key not in self._spent_gaps:
                    self._gaps.append(hole)
                    self.seen_gaps.append(hole)
                    fresh_gaps.append(hole)
            for doubt in read_doubts(answer.payload, answer.question):
                # One question earns one corroboration pass, however many
                # reasons there are to doubt it. Three doubts on the same
                # claim would otherwise fan out three identical times.
                key = (doubt.concept, doubt.predicate, doubt.relation)
                if key in self._spent_doubts or not doubt.predicate:
                    self.seen_doubts.append(doubt)
                    continue
                if answer.origin == "doubt":
                    # A corroboration question is not itself put out for
                    # corroboration: that is how a loop recurses forever
                    # over its own weak evidence.
                    self.seen_doubts.append(doubt)
                    continue
                self._doubts.append(doubt)
                self.seen_doubts.append(doubt)
                fresh_doubts.append(doubt)
        return fresh_gaps, fresh_doubts

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
            against = [(child.question, child.verdict) for child in children
                       if child.verdict in NEGATIVE]
            if not against:
                continue
            claim = children[0].predicate or ""
            found.append(Conflict(
                claim=claim, question=parent, verdict=original.verdict,
                against=against,
                detail=f"{len(against)} of the {len(children)} concepts this "
                       f"claim was put to deny it"))
        return found

    def settled(self) -> bool:
        """Nothing left that the loop could act on by itself."""
        return not self._gaps and not self._doubts and not self.carried

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
        return {"text": self.text, "cycle": self.cycle,
                "activation": self.activation.as_dict(),
                "pins": dict(self.pins),
                "asked": len(self.answers),
                "gaps": [hole.as_dict() for hole in self.seen_gaps],
                "doubts": [doubt.as_dict() for doubt in self.seen_doubts],
                "conflicts": [bad.as_dict() for bad in self.conflicts()],
                "needs_telling": [hole.as_dict()
                                  for hole in self.needs_telling()]}
