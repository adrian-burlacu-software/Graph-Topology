"""The other half of the trie: how fast it tells things apart.

`run_v686.py` measures what the trie costs to *store*. This measures what it
costs to *ask*, on exactly the same structure, and the two turn out to pull in
opposite directions.

Storing wants shared prefixes: put the predicate the most individuals carry
first and everyone walks the same corridor before splitting. Identifying wants
the opposite: the predicate almost nobody carries splits the field on the
first question. `global_coverage` and `anti_coverage` are those two policies,
and they are the ends of one axis rather than a good idea and a bad one.

    identification depth   how many predicates of an individual's path must
                           be read before no other individual shares it. This
                           is "twenty questions" measured on the trie: the
                           prefix that is unique is the point at which the
                           thing has been identified.

McRae et al. (2005) report two per-feature statistics that name these:

    distinctiveness   1 / (number of concepts carrying the feature). Ranking
                      by it is exactly `anti_coverage`.
    cue validity      P(concept | feature), weighted by how many people
                      produced it.

Their file is behind a login, but neither measure needs it: both are derived
statistics over a feature-norm corpus rather than independent human ratings,
and `load_buchanan` carries the production frequencies cue validity is
weighted by, over seven times as many concepts. So the experiment runs on the
norms already here, and `cue_validity` below is the weighted ordering McRae
would have supplied unweighted.
"""
from __future__ import annotations

import collections
import csv
import math
import statistics
from dataclasses import dataclass
from typing import Any

from .ordering import Plan, _by_rank, _key, coverage
from .substrate import Corpus
from . import corpora


@dataclass(frozen=True)
class Identifiability:
    """What one ordering costs to store, and what it costs to ask."""
    ordering: str
    corpus: str
    nodes: int
    reuse: float
    mean_depth: float
    median_depth: int
    p90_depth: int
    never_unique: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def cue_validity(corpus: Corpus) -> Plan:
    """Order by how much a predicate narrows the field, weighted by agreement.

    A predicate carried by one individual identifies it outright; one carried
    by half the corpus barely moves. Production frequency breaks the ties that
    a bare count leaves: of two predicates on three concepts each, the one
    people actually produced is the better question.
    """
    counts = coverage(corpus)
    weight = _production_frequency()
    total = max(1, len(corpus))
    rank_of: dict[Any, float] = {}
    for predicate, carried in counts.items():
        # P(one particular individual | predicate), scaled by agreement
        share = carried / total
        agreement = 1.0 + math.log1p(weight.get(str(predicate), 0.0))
        rank_of[predicate] = share / agreement
    ordered = sorted(counts, key=lambda p: (rank_of[p], _key(p)))
    return _by_rank(corpus, {p: i for i, p in enumerate(ordered)})


_FREQUENCY: dict[str, float] | None = None


def _production_frequency() -> dict[str, float]:
    """feature -> how many participants produced it, summed over concepts."""
    global _FREQUENCY
    if _FREQUENCY is not None:
        return _FREQUENCY
    totals: dict[str, float] = collections.defaultdict(float)
    if corpora.BUCHANAN.is_file():
        with corpora.BUCHANAN.open(encoding="utf-8", errors="replace",
                                   newline="") as handle:
            for row in csv.DictReader(handle):
                value = row.get("frequency_feature", "").strip()
                if value.isdigit():
                    totals[row["translated"].strip()] += int(value)
    _FREQUENCY = dict(totals)
    return _FREQUENCY


def depths(plan: Plan) -> tuple[list[int], int]:
    """For each individual, how much of its path is needed to single it out.

    Returns the depths and the count of individuals that never become unique
    -- two things carrying exactly the same predicates cannot be told apart by
    predicates, and saying so is better than reporting a depth for it.
    """
    shared: collections.Counter = collections.Counter()
    for _, path in plan:
        for cut in range(len(path) + 1):
            shared[path[:cut]] += 1

    found: list[int] = []
    never = 0
    for _, path in plan:
        for cut in range(len(path) + 1):
            if shared[path[:cut]] == 1:
                found.append(cut)
                break
        else:
            never += 1
    return found, never


def measure(corpus: Corpus, name: str, plan: Plan) -> Identifiability:
    found, never = depths(plan)
    nodes = len({path[:cut] for _, path in plan
                 for cut in range(1, len(path) + 1)})
    ordered = sorted(found)
    return Identifiability(
        ordering=name,
        corpus=corpus.name,
        nodes=nodes,
        reuse=round(1 - nodes / corpus.cells, 5) if corpus.cells else 0.0,
        mean_depth=round(statistics.fmean(found), 3) if found else 0.0,
        median_depth=ordered[len(ordered) // 2] if ordered else 0,
        p90_depth=ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
                  if ordered else 0,
        never_unique=never,
    )
