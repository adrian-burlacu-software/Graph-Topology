"""Predicate orderings.

Appendix 3 asks for one thing: "order the predicates such that they capture the
maximum number of other predicates". Every ordering here turns a corpus of
(individual, predicate set) pairs into a storage plan -- an ordered predicate
path per individual -- so a single trie can score all of them identically.

Two of them claim to be the paper's:

`global_coverage` is the Appendix 3 *text*: "the predicates with the highest
number of individual ontological types are aggregated and fired in decreasing
order". One order, computed once, applied to everyone.

`adaptive_coverage` is the Appendix 3 *figure*. Figure 20 cannot be produced by
any single global order: inside the `Big` branch it places `Long Hair` before
`Spots`, though both cover two individuals overall, because within that branch
`Long Hair` covers two and `Spots` covers one. The order is recomputed per
branch. Whether the extra work pays is the experiment's question.
"""
from __future__ import annotations

import heapq
import random
from collections import Counter
from itertools import permutations
from typing import Callable, Hashable, Sequence

Predicate = Hashable
Corpus = Sequence[tuple[Hashable, frozenset]]
Plan = list[tuple[Hashable, tuple[Predicate, ...]]]


def _key(predicate: Predicate) -> str:
    """Deterministic tiebreak so no result depends on set iteration order."""
    return repr(predicate)


def coverage(corpus: Corpus) -> Counter:
    """How many individuals each predicate captures."""
    return Counter(predicate for _, predicates in corpus for predicate in predicates)


def _by_rank(corpus: Corpus, rank: dict[Predicate, int]) -> Plan:
    return [
        (individual, tuple(sorted(predicates, key=lambda p: (rank[p], _key(p)))))
        for individual, predicates in corpus
    ]


def lexical(corpus: Corpus) -> Plan:
    """No optimization at all: sort predicates by their own identity."""
    return [
        (individual, tuple(sorted(predicates, key=_key)))
        for individual, predicates in corpus
    ]


def shuffled(corpus: Corpus, seed: int = 7) -> Plan:
    """An arbitrary but fixed order -- the honest control for `global_coverage`."""
    universe = sorted(coverage(corpus), key=_key)
    random.Random(seed).shuffle(universe)
    return _by_rank(corpus, {p: i for i, p in enumerate(universe)})


def global_coverage(corpus: Corpus) -> Plan:
    """Appendix 3 as written: one decreasing-coverage order for everyone."""
    counts = coverage(corpus)
    universe = sorted(counts, key=lambda p: (-counts[p], _key(p)))
    return _by_rank(corpus, {p: i for i, p in enumerate(universe)})


def anti_coverage(corpus: Corpus) -> Plan:
    """Increasing coverage: the worst sane order, bounding what ordering can cost."""
    counts = coverage(corpus)
    universe = sorted(counts, key=lambda p: (counts[p], _key(p)))
    return _by_rank(corpus, {p: i for i, p in enumerate(universe)})


def adaptive_coverage(corpus: Corpus) -> Plan:
    """Figure 20: re-rank predicates by coverage within each branch.

    The obvious implementation recounts the branch on every pick and rescans it
    to partition -- quadratic, 36s at 20k individuals and hours at 280k. Here
    each branch builds an inverted predicate index once, so the chosen
    predicate's members are a lookup rather than a scan, and only those members'
    predicates are touched on the way out. A lazily-revalidated max-heap serves
    the next pick. Iterative over the "lacks this predicate" chain and
    explicit-stack over branches, so depth is bounded by the trie rather than by
    the interpreter's recursion limit.
    """
    remaining = [set(predicates) for _, predicates in corpus]
    plan: list[tuple[Predicate, ...] | None] = [None] * len(remaining)
    stack: list[tuple[tuple[Predicate, ...], list[int]]] = [
        ((), list(range(len(remaining))))
    ]
    while stack:
        prefix, members = stack.pop()
        index: dict[Predicate, set[int]] = {}
        active: set[int] = set()
        for member in members:
            if remaining[member]:
                active.add(member)
                for predicate in remaining[member]:
                    index.setdefault(predicate, set()).add(member)
            else:
                # Out of predicates: this branch is where the individual lives.
                plan[member] = prefix
        heap = [(-len(holders), _key(p), p) for p, holders in index.items()]
        heapq.heapify(heap)
        while active:
            best = None
            while heap:
                negated, sort_key, candidate = heap[0]
                current = len(index.get(candidate, ()))
                if current == -negated and current:
                    best = candidate
                    break
                heapq.heappop(heap)
                if current:
                    heapq.heappush(heap, (-current, sort_key, candidate))
            if best is None:  # pragma: no cover - index and active agree
                for member in active:
                    plan[member] = prefix
                break
            carried = list(index[best])
            for member in carried:
                for predicate in remaining[member]:
                    index[predicate].discard(member)
                remaining[member].discard(best)
                active.discard(member)
            stack.append((prefix + (best,), carried))
    return [
        (individual, plan[position])  # type: ignore[misc]
        for position, (individual, _) in enumerate(corpus)
    ]


def optimal(corpus: Corpus, max_predicates: int = 8) -> Plan:
    """Exhaustive search over every global order. Small corpora only.

    Bounds how much of `global_coverage`'s result is the heuristic and how much
    is simply available in the data.
    """
    universe = sorted(coverage(corpus), key=_key)
    if len(universe) > max_predicates:
        raise ValueError(
            f"optimal ordering is factorial; {len(universe)} predicates exceeds "
            f"max_predicates={max_predicates}"
        )
    best_plan, best_nodes = None, None
    for candidate in permutations(universe):
        rank = {p: i for i, p in enumerate(candidate)}
        current = _by_rank(corpus, rank)
        nodes = len({path[:i + 1] for _, path in current for i in range(len(path))})
        if best_nodes is None or nodes < best_nodes:
            best_plan, best_nodes = current, nodes
    assert best_plan is not None
    return best_plan


ORDERINGS: dict[str, Callable[[Corpus], Plan]] = {
    "lexical": lexical,
    "shuffled": shuffled,
    "global_coverage": global_coverage,
    "adaptive_coverage": adaptive_coverage,
    "anti_coverage": anti_coverage,
}
