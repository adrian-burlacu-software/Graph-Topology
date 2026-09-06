"""Metrics for a stored ontology.

The paper's two claims are separate measurements and must not be conflated:

Growth  -- "the topographical growth of tries is driven by allocation"
           (Section 15). Measured as nodes allocated versus the flat cost of
           storing every individual-predicate pair independently.

Access  -- "performance is centred around the length of the word and not the
           size of the vocabulary" (Section 2.b). Measured as the depth at
           which an individual is reached, tracked while the vocabulary grows.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass
from typing import Callable

from .ordering import Plan
from .substrate import Corpus
from .trie import PredicateTrie


@dataclass(frozen=True)
class Measurement:
    """One ordering's result on one corpus.

    Depth is reported three ways because the corpus has extreme outliers:
    `en:person` carries 6,231 predicates, so the mean is pulled by a handful of
    ConceptNet hub concepts that no ordering can help. The median is what a
    typical individual actually costs to reach.
    """

    ordering: str
    corpus: str
    individuals: int
    flat_cells: int
    distinct_predicates: int
    nodes: int
    reuse_rate: float
    mean_depth: float
    median_depth: float
    p99_depth: int
    max_depth: int
    terminal_nodes: int

    def as_dict(self) -> dict:
        return asdict(self)


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    return sorted(values)[min(len(values) - 1, int(len(values) * fraction))]


def store(corpus: Corpus, plan: Plan) -> PredicateTrie:
    """Grow a trie by inserting every individual along its planned path."""
    trie = PredicateTrie()
    for individual, path in plan:
        trie.insert(individual, path)
    return trie


def measure(corpus: Corpus, ordering_name: str, plan: Plan) -> Measurement:
    trie = store(corpus, plan)
    depths = list(trie.depths())
    cells = corpus.cells
    return Measurement(
        ordering=ordering_name,
        corpus=corpus.name,
        individuals=len(corpus),
        flat_cells=cells,
        distinct_predicates=corpus.predicates,
        nodes=trie.node_count,
        reuse_rate=round(1 - trie.node_count / cells, 6) if cells else 0.0,
        mean_depth=round(statistics.fmean(depths), 4) if depths else 0.0,
        median_depth=statistics.median(depths) if depths else 0.0,
        p99_depth=_percentile(depths, 0.99),
        max_depth=max(depths, default=0),
        terminal_nodes=trie.terminal_nodes,
    )


def compare(corpus: Corpus, orderings: dict[str, Callable[[Corpus], Plan]]) -> list[Measurement]:
    """Score every ordering on the same corpus, cheapest storage first."""
    results = [measure(corpus, name, fn(corpus)) for name, fn in orderings.items()]
    return sorted(results, key=lambda item: item.nodes)


@dataclass
class ScalingPoint:
    individuals: int
    flat_cells: int
    nodes: int
    mean_depth: float
    median_depth: float
    p99_depth: int
    max_depth: int


def scaling_curve(
    corpus: Corpus,
    ordering: Callable[[Corpus], Plan],
    steps: int = 10,
    seed: int = 3,
) -> list[ScalingPoint]:
    """Grow the vocabulary and watch node count against access depth.

    The paper predicts the two diverge: nodes keep climbing while depth stays
    flat. A rising depth curve would falsify it.

    The corpus is shuffled once, deterministically, before prefixes are taken.
    Taking them in the corpus's own order would grow the vocabulary
    alphabetically -- every `en:a...` concept before any `en:z...` -- which is
    not a sample of the ontology but a sample of the dictionary, and it makes
    the depth curve wander for reasons that have nothing to do with size.
    """
    order = list(corpus.items)
    random.Random(seed).shuffle(order)
    points: list[ScalingPoint] = []
    for step in range(1, steps + 1):
        size = max(1, len(order) * step // steps)
        slice_ = Corpus(corpus.name, tuple(order[:size]))
        trie = store(slice_, ordering(slice_))
        depths = list(trie.depths())
        points.append(
            ScalingPoint(
                individuals=len(slice_),
                flat_cells=slice_.cells,
                nodes=trie.node_count,
                mean_depth=round(statistics.fmean(depths), 4) if depths else 0.0,
                median_depth=statistics.median(depths) if depths else 0.0,
                p99_depth=_percentile(depths, 0.99),
                max_depth=max(depths, default=0),
            )
        )
    return points
