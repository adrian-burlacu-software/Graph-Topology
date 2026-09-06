"""Questions about a thing's relation to something else.

    what does a dog's owner need
    what can a violin's player do
    where does a dog's owner live

V684 answers questions with one subject. This one has two: the question is
about owners, but only the owners a dog has, and an answer that skips the
first half is an answer about owners in general with the dog as decoration.

So it runs in two parts, and both are shown:

    bridge   why an owner is relevant to a dog at all. `bridge.py` searches
             the fact graph for a route and returns the facts that license
             it -- `dog has_a "owner"`. Without a route the question is not
             refused, it is answered and labelled as unbridged, because "I
             know about owners but cannot connect them to dogs" is a
             different and more useful thing to say than nothing.

    answer   v684 takes over unchanged: the role concept, the relation the
             question asked for, and inheritance up the taxonomy. Every rule
             R1-R13 still applies, and the trace it produces is the same one
             the existing page already replays.

The split matters more than it looks. The bridge is a search over 1.7M facts
with a branching factor of 870; the answer is a walk over 15 concepts. Putting
them in one search would drag the reliable half into the expensive one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..v684 import build
from ..v684.language import Parser
from ..v684.reason import Answer, Reasoner
from .bridge import Bridge, Route, SearchReport
from .graph import FactGraph

DEFAULT_STORE = build.DEFAULT_STORE.with_name("v684_reasoning_compressed.sqlite")


@dataclass
class Bridged:
    """A two-part answer: why the role is relevant, then what it says."""
    question: str
    anchor_word: str | None = None
    role_word: str | None = None
    anchor: str | None = None
    role: str | None = None
    relation: str | None = None
    route: Route | None = None
    alternatives: list[Route] = field(default_factory=list)
    report: SearchReport | None = None
    answer: Answer | None = None
    verdict: str = "UNKNOWN"
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": self.question, "verdict": self.verdict,
            "anchor_word": self.anchor_word, "role_word": self.role_word,
            "anchor": self.anchor, "role": self.role,
            "relation": self.relation, "note": self.note,
            "route": self.route.as_dict() if self.route else None,
            "alternatives": [r.as_dict() for r in self.alternatives],
            "search": self.report.as_dict() if self.report else None,
        }
        payload["answer"] = self.answer.as_dict() if self.answer else None
        return payload


class BridgedReasoner:
    """v685: bridge two concepts, then let v684 answer about the second."""

    def __init__(self, store: Path = DEFAULT_STORE, depth: int = 3,
                 breadth: int = 60):
        self.reasoner = Reasoner(store)
        self.parser = Parser(vocabulary=self.reasoner.vocabulary())
        self.graph = FactGraph(store)
        self.bridge = Bridge(self.graph, depth=depth, breadth=breadth)

    def close(self) -> None:
        self.reasoner.close()
        self.graph.close()

    # -- reading the question ---------------------------------------------
    def possessive(self, question: str) -> tuple[str | None, str | None]:
        """`a dog's owner` -> ("dog", "owner").

        spaCy marks this with the `poss` dependency and does it reliably
        across the shapes tried -- what/where/can questions all produce
        `dog --poss--> owner`. No pattern matching on apostrophes.
        """
        if self.parser.nlp is None:
            return None, None
        for token in self.parser.nlp(question):
            if token.dep_ == "poss" and token.pos_ in ("NOUN", "PROPN"):
                head = token.head
                if head.pos_ in ("NOUN", "PROPN"):
                    return token.lemma_.lower(), head.lemma_.lower()
        return None, None

    def sense_of(self, word: str) -> str | None:
        senses = self.reasoner.senses_of(word)
        return senses[0]["id"] if senses else None

    #: How many senses of the role word to try bridging before giving up.
    #: They are already ordered best-first by v684, so the tail is the part
    #: least likely to be meant.
    ROLE_SENSES_TRIED = 6

    def role_sense(self, word: str, anchor: str | None
                   ) -> tuple[str | None, SearchReport | None]:
        """The sense of the role word that the anchor can actually reach.

        `player` defaults to the sports sense, so "what can a violin's player
        do" answered with concussions and anthems. The bridge settles it
        without any new machinery: of the senses `player` might mean, take one
        the violin connects to. The graph disambiguates, which is the whole
        point of having built it -- a word's meaning here is which other
        concepts it is joined to.

        Falls back to v684's own ordering when nothing is reachable, so an
        unbridged answer is still given rather than withheld.
        """
        senses = self.reasoner.senses_of(word)
        if not senses:
            return None, None
        if anchor is None:
            return senses[0]["id"], None
        best: tuple[float, str, SearchReport] | None = None
        for candidate in senses[:self.ROLE_SENSES_TRIED]:
            report = self.bridge.search(anchor, candidate["id"], want=2)
            if not report.routes:
                continue
            route = report.routes[0]
            key = (len(route.hops), route.cost)
            if best is None or key < (len(best[2].routes[0].hops),
                                      best[2].routes[0].cost):
                best = (route.cost, candidate["id"], report)
        if best:
            return best[1], best[2]
        return senses[0]["id"], None

    # -- answering ---------------------------------------------------------
    def ask(self, question: str) -> Bridged:
        result = Bridged(question=question)
        anchor_word, role_word = self.possessive(question)
        result.anchor_word, result.role_word = anchor_word, role_word
        if not role_word:
            result.verdict = "NOT_BRIDGED"
            result.note = ("No possessive in the question -- nothing to bridge. "
                           "Ask v684 directly.")
            return result

        parse = self.parser.parse(question)
        result.relation = parse.relation or "capable_of"
        result.anchor = self.sense_of(anchor_word) if anchor_word else None
        result.role, report = self.role_sense(role_word, result.anchor)
        if not result.role:
            result.verdict = "UNKNOWN_WORD"
            result.note = f"“{role_word}” is not in the ontology."
            return result

        if result.anchor:
            if report is None:
                report = self.bridge.search(result.anchor, result.role)
            result.report = report
            if report.routes:
                result.route = report.routes[0]
                result.alternatives = report.routes[1:]
            else:
                result.note = (
                    f"Nothing in the store connects {result.anchor} to "
                    f"{result.role}, so this is what is known about "
                    f"{role_word} generally, not about a {anchor_word}'s.")
        elif anchor_word:
            result.note = f"“{anchor_word}” is not in the ontology."

        # v684 answers the second half, entirely unchanged
        result.answer = self.reasoner.describe(result.role, result.relation)
        result.verdict = "BRIDGED" if result.route else "UNBRIDGED"
        return result
