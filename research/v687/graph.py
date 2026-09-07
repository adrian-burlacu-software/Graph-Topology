"""The fact graph: v684's store read as something you can walk through.

V684 answers by going *up*. Every question it takes is about one concept, and
the only edge it follows is `is_a`. That is why its search is 15 concepts and
one millisecond -- and why it cannot answer "what does a dog's owner need",
which is not a question about dogs at all.

The store already contains the edges needed for that, hidden inside the text:

    dog.n.01  capable_of  "belong to owner"

`owner` is a concept. 80.1% of the store's fact objects name one, so resolving
the object text turns 1.7M free-text facts into a directed multi-relation
graph over the same 117,659 concepts. Nothing is rebuilt; this is a different
reading of the same rows.

The cost of that reading is the whole reason the next module exists:

    dog.n.01   up the is_a DAG      15 concepts   (v684, exhaustive, ~1 ms)
    dog.n.01   one fact hop        680 concepts
    dog.n.01   two fact hops     8,064 concepts   (and that is a sample)

An exhaustive walk stops being possible somewhere in the first hop.
"""
from __future__ import annotations

import collections
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import senses


@dataclass(frozen=True)
class Hop:
    """One edge, carrying the fact that licenses it.

    `text` is kept because it is the evidence a person reads: the hop from
    `dog` to `owner` is only believable because something says "belong to
    owner", and an answer that cannot show that sentence is not an answer.
    """
    source: str
    relation: str
    text: str
    target: str
    confidence: float

    def as_dict(self) -> dict:
        return {"source": self.source, "relation": self.relation,
                "text": self.text, "target": self.target,
                "confidence": round(self.confidence, 4)}


class FactGraph:
    """Concepts joined by their facts, resolved through the lemma index."""

    #: A concept naming more objects than this is a hub. `entity` and `person`
    #: connect to everything and mean nothing as a route.
    HUB_DEGREE = 4000

    def __init__(self, store: Path):
        self.connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True,
                                          check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        parents: dict[str, set[str]] = collections.defaultdict(set)
        self.children: dict[str, list[str]] = collections.defaultdict(list)
        for child, parent in self.connection.execute(
                "SELECT child, parent FROM taxonomy"):
            parents[child].add(parent)
            self.children[parent].append(child)
        self.parents = parents
        lemma_index: dict[str, list[str]] = collections.defaultdict(list)
        for lemma, concept in self.connection.execute(
                "SELECT lemma, concept FROM lemmas"):
            lemma_index[lemma].append(concept)
        self.lemmas = lemma_index
        self.resolver = senses.Ranges(parents, lemma_index, {})
        self._hops: dict[str, tuple[Hop, ...]] = {}

    def close(self) -> None:
        self.connection.close()

    def denotes(self, phrase: str) -> str | None:
        """The concept a fact's object names, if it names one."""
        options = self.resolver.denotes(phrase)
        nouns = [c for c in options if c.rsplit(".", 2)[1] == "n"]
        return (nouns or options or [None])[0]

    def hops(self, concept: str) -> tuple[Hop, ...]:
        """Every concept one fact away, with the fact that gets you there.

        Memoised: the search revisits the same concepts constantly, and this
        is the expensive half of a hop.
        """
        cached = self._hops.get(concept)
        if cached is not None:
            return cached
        seen: set[tuple[str, str]] = set()
        found: list[Hop] = []
        for row in self.connection.execute(
                "SELECT relation, object, confidence FROM facts WHERE concept = ? "
                "ORDER BY confidence DESC", (concept,)):
            target = self.denotes(row["object"])
            if target is None or target == concept:
                continue
            key = (row["relation"], target)
            if key in seen:            # many phrasings, one edge
                continue
            seen.add(key)
            found.append(Hop(concept, row["relation"], row["object"], target,
                             row["confidence"]))
        # `is_a` belongs in the same graph: reaching `owner` sometimes means
        # generalising first. It is marked so a route can be read back.
        for parent in self.parents.get(concept, ()):
            found.append(Hop(concept, "is_a", parent.rsplit(".", 2)[0],
                             parent, 0.95))
        self._hops[concept] = tuple(found)
        return self._hops[concept]

    def degree(self, concept: str) -> int:
        return len(self.hops(concept))

    def is_hub(self, concept: str) -> bool:
        """Too well connected to carry meaning as an intermediate step."""
        return self.degree(concept) >= self.HUB_DEGREE
