"""Appendix 3's trie, measured on elicited feature norms.

    python -m research.v686.run_v686

V683 ran this same trie and the same six orderings on a scraped ontology.
This runs them on data of the kind the paper actually assumes: a closed
predicate vocabulary, elicited by putting the same question to people about
every concept, so two robins really do carry overlapping predicate sets.

Nothing about the mechanism changes. `research/v683/trie.py` and
`research/v683/ordering.py` are imported unmodified; only the corpus is new.
That is the point -- if the compression is a property of the paper's
structure rather than of one dataset, it has to show up without touching it.

Reported per ordering:

    nodes       what the trie cost to store the corpus
    reuse       1 - nodes/cells: the share of flat storage prefix sharing
                removed. This is the compression figure.
    depth       how far down an individual sits, median and worst case
"""
from __future__ import annotations

import argparse
import time

from ..v683.measure import compare, measure
from ..v683.ordering import ORDERINGS, coverage, optimal
from ..v683.substrate import Corpus
from . import corpora

#: `optimal` is factorial in the predicate count, so it is only reachable on a
#: corpus with a handful of predicates. It is the floor the heuristics are
#: measured against, not a proposal.
OPTIMAL_LIMIT = 8


def bench(corpus: Corpus) -> None:
    print(f"\n  {corpus.name}")
    print(f"    {len(corpus):,} individuals · {corpus.predicates:,} predicates · "
          f"{corpus.cells:,} flat cells")
    plans = dict(ORDERINGS)
    started = time.time()
    results = compare(corpus, plans)
    print(f"    {'ordering':<20}{'nodes':>9}{'reuse':>9}{'median d':>10}"
          f"{'p99 d':>8}{'max d':>8}")
    best = results[0].nodes
    for row in results:
        marker = "  <- best" if row.nodes == best else ""
        print(f"    {row.ordering:<20}{row.nodes:>9,}{row.reuse_rate:>9.1%}"
              f"{row.median_depth:>10.0f}{row.p99_depth:>8}{row.max_depth:>8}"
              f"{marker}")
    print(f"    [{time.time() - started:.1f}s]")


def optimal_floor() -> None:
    """How much of the heuristic's win was simply available in the data.

    `optimal` searches every *global* order exhaustively -- one sequence for
    every individual. It is the floor for that family, and `adaptive_coverage`
    goes under it, because branch-local ordering is a strictly larger search
    space than any single global sequence. That is Figure 20 beating the
    Appendix 3 text, bounded rather than argued.

    The slice is the eight most-carried attributes, tie-broken by name: three
    attributes tie at 39 and `Counter.most_common` picks between them by
    insertion order, which varies with the hash seed. Sorting makes the corpus
    reproducible.
    """
    awa = corpora.load_awa2()
    counts = coverage(awa)
    top = {p for p, _ in sorted(counts.items(),
                                key=lambda kv: (-kv[1], kv[0]))[:OPTIMAL_LIMIT]}
    trimmed = Corpus("awa2/top8", tuple(
        (individual, frozenset(p for p in predicates if p in top))
        for individual, predicates in awa.items))
    trimmed = Corpus("awa2/top8", tuple(x for x in trimmed.items if x[1]))
    plans = dict(ORDERINGS)
    plans["optimal"] = optimal
    print(f"\n  {trimmed.name}  (small enough for {OPTIMAL_LIMIT}! = 40,320 orders)")
    print(f"    {len(trimmed)} individuals · {trimmed.predicates} predicates · "
          f"{trimmed.cells} flat cells")
    for row in compare(trimmed, plans):
        note = ("  <- beats the best global order"
                if row.ordering == "adaptive_coverage" else "")
        print(f"    {row.ordering:<20}{row.nodes:>7,}{row.reuse_rate:>9.1%}{note}")


def scaling() -> None:
    """Does prefix sharing pay off more as the corpus grows?"""
    print("\n  compression against corpus size (adaptive_coverage)")
    for label, corpus in (("awa2", corpora.load_awa2()),
                          ("xcslb", corpora.load_xcslb()),
                          ("buchanan", corpora.load_buchanan())):
        print(f"    {label}")
        for fraction in (0.1, 0.25, 0.5, 1.0):
            count = max(2, int(len(corpus) * fraction))
            head = Corpus(f"{label}[:{count}]", corpus.items[:count])
            result = measure(head, "adaptive_coverage",
                             ORDERINGS["adaptive_coverage"](head))
            print(f"      {count:>4} individuals  {result.flat_cells:>7,} cells  "
                  f"{result.nodes:>7,} nodes  reuse {result.reuse_rate:>6.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="run one corpus by name fragment")
    arguments = parser.parse_args()

    everything = [
        ("awa2", lambda: corpora.load_awa2()),
        ("xcslb", lambda: corpora.load_xcslb()),
        ("buchanan", lambda: corpora.load_buchanan()),
        ("buchanan/surface", lambda: corpora.load_buchanan(root_forms=False)),
        ("xcslb/visual", lambda: corpora.load_xcslb(kinds=("visual perceptual",))),
        ("xcslb/taxonomic", lambda: corpora.load_xcslb(kinds=("taxonomic",))),
        ("xcslb/bird", lambda: corpora.load_xcslb(category="bird")),
        ("xcslb/animal", lambda: corpora.load_xcslb(category="animal")),
        ("xcslb/bird-visual", lambda: corpora.load_xcslb(
            kinds=("visual perceptual",), category="bird")),
    ]
    print("Appendix 3 compression on elicited feature norms")
    print("=" * 64)
    for label, build in everything:
        if arguments.only and arguments.only not in label:
            continue
        bench(build())
    if not arguments.only:
        optimal_floor()
        scaling()


if __name__ == "__main__":
    main()
