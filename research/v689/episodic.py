"""Episodic memory: the individuals of a conversation, held the way the store
holds kinds and reasoned over by the same rules.

## The same rules

v687's store is semantic memory -- what is true of kinds. An individual is one
more node at the bottom of its taxonomy: the pig you mentioned sits under
`hog.n.03`. What you told me about it is one more row of the same shape, with
`told` as its source. `EpisodicReasoner` is v687's `Reasoner` with its two
lookups extended, `parents_of` and `facts_of`, so every rule in `rules.py`
runs on an individual unchanged:

    it can't swim        not_capable_of swim   R3: an explicit negation at this
                                               level blocks what beagles do
    it was flying        capable_of fly        R4: found at distance 0, before
                                               anything is inherited
    it has no tail       has_a "no tail"       R3: a denial written into the
                                               object, as the crawl writes them
    can the beagle bark  nothing told          R1 up to beagle, R2, R4, R5

One rule is added, and it is R2's question asked one level lower.

**E1. A quality does not descend from a kind to an individual.** `is a beagle
black` is recorded, and beagles are tricoloured; `is the second beagle black`
is about one dog. R2 decides per relation what descends between kinds. E1
says `has_property` does not descend onto an individual at all, so a quality
is answered from what was said about that individual or not at all.

One relation is added, and no rule reads it. **`did_not`**: `it wasn't flying`
says nothing about whether it can, so it is not `not_capable_of`. Stored as
that, R3 would answer `can it fly` no about a pig that was merely on the
ground.

## The trie, live

Appendix 3 stores individuals as goal nodes under ordered predicate paths.
The episodic trie is that structure over the conversation. An individual's
predicates are its kind and every kind above it -- so `the dog` reaches a
beagle by R1's closure rather than by a special case -- what it was told, what
it is called, and whose it is. It is re-planned with `adaptive_coverage` on
every change, because "the topographical growth of tries is driven by
allocation", and every change reports what it allocated. A second beagle costs
nothing: it shares every predicate with the first until one of them is told
something.

Resolving a description is identification, the trie read downwards, as
`identify.py` reads it: walk down until the description is exhausted, and
everything stored beneath that point fits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.v687 import rules
from research.v687.ordering import adaptive_coverage
from research.v687.reason import Fact, Reasoner
from research.v687.rules import Step
from research.v687.trie import ROOT, PredicateTrie

#: The source column of everything a conversation stores.
TOLD = "told"

#: E1: what a quality is.
QUALITIES = frozenset({"has_property", "has_attribute", "not_has_property"})

#: Not doing a thing. Read by the session for `does it`, never by a rule.
DID_NOT = "did_not"

DENIERS = ("no ", "not ")


def name_of(node: str | None) -> str:
    """`hunting dog.n.01` -> `hunting dog`; an individual's id is its own."""
    if not node:
        return ""
    return node.rsplit(".", 2)[0] if node.count(".") >= 2 else node


def _stem(obj: str) -> str:
    """An object with its denial taken off: `no tail` and `tail` are the
    same claim, asserted and denied."""
    text = (obj or "").lower().strip()
    for denier in DENIERS:
        if text.startswith(denier):
            return text[len(denier):].strip()
    return text


@dataclass
class Growth:
    """What one change to episodic memory did to the trie."""

    reason: str
    individuals: int
    cells: int            # individual-predicate pairs, stored flat
    nodes: int
    allocated: int        # nodes this change added; negative if it freed some

    def as_dict(self) -> dict:
        return {"reason": self.reason, "individuals": self.individuals,
                "cells": self.cells, "nodes": self.nodes,
                "allocated": self.allocated,
                "shared": (round(1 - self.nodes / self.cells, 3)
                           if self.cells else 0.0)}


@dataclass
class Identification:
    """A description walked down the trie, and who is stored beneath it."""

    wanted: list
    candidates: list
    visited: int
    #: every path at which the description was exhausted
    reached: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"wanted": list(self.wanted),
                "candidates": list(self.candidates),
                "visited": self.visited,
                "reached": [list(path) for path in self.reached]}


class EpisodicMemory:
    """One conversation's individuals, what was told of them, and the trie."""

    def __init__(self, reasoner: Reasoner) -> None:
        self.base = reasoner
        #: individual -> the synset it sits under; None if the ontology has
        #: no sense for the word it was introduced by
        self.parent: dict[str, str | None] = {}
        #: individual -> every kind it is, nearest first, as words
        self.lineage: dict[str, list[str]] = {}
        self.facts: dict[str, list[Fact]] = {}
        #: (individual, relation, object) -> the utterance that told it
        self.said: dict[tuple, str] = {}
        #: (individual, relation, object) -> v688's answer for the kind,
        #: when it was told
        self.against: dict[tuple, str] = {}
        #: individual -> `name rex`, `owner you`
        self.labels: dict[str, set[str]] = {}
        self.trie = PredicateTrie()
        self.plan: list = []
        self.members: dict[tuple, list[str]] = {}
        self.growth: list[Growth] = []
        self.reasoner = EpisodicReasoner(reasoner, self)

    # -- writing -----------------------------------------------------------
    def place(self, individual: str, word: str, sense: str | None) -> Growth:
        """Put an individual under a kind, or move it under a narrower one."""
        self.parent[individual] = sense
        kinds = [word]
        if sense:
            kinds += [name_of(node) for node, _, _ in self.base.ascend(sense)]
        self.lineage[individual] = list(dict.fromkeys(kinds))
        self.facts.setdefault(individual, [])
        self.labels.setdefault(individual, set())
        return self.store(f"{individual} is {word}")

    def tell(self, individual: str, relation: str, obj: str,
             said: str) -> Growth:
        """Record one fact about one individual.

        A fact replaces what it contradicts -- the same relation or its
        negation, about the same object. `it can swim` after `it can't swim`
        is a correction, not two facts for R3 to adjudicate.
        """
        opposed = {relation}
        opposed |= {rules.NEGATIONS[relation]} if relation in rules.NEGATIONS \
            else set()
        opposed |= {rules.POSITIVES[relation]} if relation in rules.POSITIVES \
            else set()
        stem = _stem(obj)
        kept = [fact for fact in self.facts[individual]
                if not (fact.relation in opposed and _stem(fact.object) == stem)]
        kept.append(Fact(individual, relation, obj, TOLD, 1.0, False))
        self.facts[individual] = kept
        # Kept without its closing punctuation, because it is only ever shown
        # inside quotation marks inside a sentence.
        self.said[(individual, relation, obj)] = said.strip().rstrip(".!?")
        return self.store(f"{individual} {relation} {obj}")

    def label(self, individual: str, predicate: str) -> Growth:
        """`name rex`, `owner you`: one of each kind, the latest kept."""
        head = predicate.split(" ", 1)[0] + " "
        self.labels[individual] = {one for one in self.labels[individual]
                                   if not one.startswith(head)} | {predicate}
        return self.store(f"{individual} {predicate}")

    def predicates_of(self, individual: str) -> frozenset:
        return frozenset(
            [f"is_a {kind}" for kind in self.lineage.get(individual, [])]
            + [f"{fact.relation} {fact.object.lower()}"
               for fact in self.facts.get(individual, [])]
            + list(self.labels.get(individual, ())))

    def store(self, reason: str) -> Growth:
        """Re-plan and rebuild: Appendix 3, run on every change."""
        before = self.trie.node_count
        corpus = tuple(sorted((individual, self.predicates_of(individual))
                              for individual in self.parent))
        self.plan = adaptive_coverage(corpus)
        self.trie, self.members = PredicateTrie(), {}
        for individual, path in self.plan:
            self.trie.insert(individual, path)
            for cut in range(len(path) + 1):
                self.members.setdefault(tuple(path[:cut]), []).append(
                    individual)
        growth = Growth(reason, len(corpus),
                        sum(len(predicates) for _, predicates in corpus),
                        self.trie.node_count, self.trie.node_count - before)
        self.growth.append(growth)
        return growth

    # -- reading -----------------------------------------------------------
    def identify(self, wanted) -> Identification:
        """Walk down the trie until the description is exhausted.

        A predicate on the path that the description does not name is walked
        through, not refused -- a black beagle is still a beagle -- and once
        every wanted predicate has been met, everyone stored beneath fits.
        """
        wanted = frozenset(wanted)
        order = list(self.parent)
        found: set[str] = set()
        reached: list[tuple] = []
        visited = 0
        stack = [(ROOT, frozenset())]
        while stack:
            node, have = stack.pop()
            visited += 1
            if have >= wanted:
                path = self.trie.path(node)
                reached.append(path)
                found.update(self.members.get(path, ()))
                continue
            for symbol, child in self.trie.children(node).items():
                stack.append((child, have | (frozenset({symbol}) & wanted)))
        return Identification(sorted(wanted),
                              sorted(found, key=order.index), visited,
                              reached)

    def as_dict(self) -> dict:
        paths = dict(self.plan)
        last = self.growth[-1] if self.growth else None
        return {"nodes": self.trie.node_count,
                "cells": last.cells if last else 0,
                "individuals": [
                    {"id": individual, "parent": self.parent[individual],
                     "path": list(paths.get(individual, ())),
                     "facts": [
                         {"relation": fact.relation, "object": fact.object,
                          "said": self.said.get(
                              (individual, fact.relation, fact.object), "")}
                         for fact in self.facts.get(individual, [])]}
                    for individual in self.parent]}


class EpisodicReasoner(Reasoner):
    """v687's `Reasoner`, with a conversation's individuals at the bottom of
    its taxonomy. Shares the base reasoner's connection and caches."""

    def __init__(self, base: Reasoner, memory: EpisodicMemory) -> None:
        # Deliberately not `super().__init__`: that opens a second
        # connection to the store, and this is the same reasoner with two
        # lookups extended.
        self.__dict__.update(base.__dict__)
        self.memory = memory
        self._individual_only = False

    def parents_of(self, concept: str) -> list[str]:
        if concept in self.memory.parent:
            sense = self.memory.parent[concept]
            return [sense] if sense else []
        return super().parents_of(concept)

    def facts_of(self, concept: str, relation: str | None = None) -> list:
        if concept not in self.memory.parent:
            return super().facts_of(concept, relation)
        group = set(rules.family(relation)) if relation else None
        # Copies: `verify` writes distance and decayed confidence onto the
        # facts it is handed, and these are the memory's own rows.
        return [Fact(fact.concept, fact.relation, fact.object, fact.source,
                     fact.confidence, False)
                for fact in self.memory.facts.get(concept, [])
                if group is None or fact.relation in group]

    def gloss(self, concept: str) -> str | None:
        if concept in self.memory.parent:
            return (f"an individual under "
                    f"{self.memory.parent[concept] or 'no known kind'}")
        return super().gloss(concept)

    def too_broad(self, concept: str) -> bool:
        return False if concept in self.memory.parent else \
            super().too_broad(concept)

    def ascend(self, concept: str):
        for node, distance, parents in super().ascend(concept):
            yield node, distance, parents
            if self._individual_only:
                return

    def verify(self, concept: str, relation: str, target: str, matcher):
        quality = concept in self.memory.parent and relation in QUALITIES
        self._individual_only = quality
        try:
            answer = super().verify(concept, relation, target, matcher)
        finally:
            self._individual_only = False
        if quality and answer.verdict == "UNKNOWN":
            kind = name_of(self.memory.parent.get(concept)) or "its kind"
            answer.steps.append(Step(
                len(answer.steps), "stop", concept, 0, "E1",
                f"`{relation}` is a quality, and a quality does not descend "
                f"from {kind} to one of them."))
            answer.note = (
                f"Nothing was said of this one being “{target}”. What holds "
                f"of {kind} in general is a tendency of the kind, and E1 does "
                f"not let it descend onto an individual.")
        return answer
