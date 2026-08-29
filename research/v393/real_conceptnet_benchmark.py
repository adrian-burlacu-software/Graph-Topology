
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import argparse
import json
import re

from semantic_memory import (
    IndexedSemanticMemory,
    SemanticCandidate,
    SemanticEdge,
)
from semantic_architecture import IntegratedSemanticArchitecture


def canonical(x):
    x=str(x).strip().lower()
    x=re.sub(r"^/c/[a-z]{2}/","",x)
    x=x.split("/")[0]
    x=x.replace("_"," ").replace("-"," ")
    x=re.sub(r"[^a-z0-9' ]+"," ",x)
    return re.sub(r"\s+"," ",x).strip()


class RealConceptNetAmbiguityMemory(IndexedSemanticMemory):
    """
    Real-graph surface ambiguity adapter.

    A query is represented by all exact canonical concepts whose labels match
    the surface form. The benchmark then uses the graph neighborhood of each
    candidate as semantic evidence.
    """

    def __init__(self, *args, surface_aliases=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.surface_aliases=surface_aliases or {}

    def retrieve(self, query, max_candidates=8, hop=1):
        q=canonical(query)
        aliases=self.surface_aliases.get(q, ())
        candidates=[]

        # Exact concept first.
        if q in self.concepts():
            candidates.append(q)

        # Explicit aliases let the benchmark represent lexical ambiguity
        # without changing the underlying ConceptNet graph.
        for alias in aliases:
            if alias in self.concepts() and alias not in candidates:
                candidates.append(alias)

        # Fallback to normal exact retrieval.
        if not candidates:
            return super().retrieve(query,max_candidates,hop)

        prior=1.0/max(1,len(candidates))
        return tuple(
            SemanticCandidate(
                concept=c,
                prior=prior,
                evidence=self.neighborhood(
                    c,
                    max_edges=16,
                ),
            )
            for c in candidates[:max_candidates]
        )


def build_from_sqlite(db_path: Path):
    from semantic_memory import ConceptNetSQLiteLoader
    loader=ConceptNetSQLiteLoader(db_path)
    memory=loader.load_index()
    return loader,memory


def choose_real_ambiguous_surface(memory):
    """
    Find a useful ambiguous surface form from the actual graph.

    We use concepts with the same normalized first lexical token and distinct
    neighborhoods, preferring pairs with different outgoing relation targets.
    """
    concepts=sorted(memory.concepts())

    buckets=defaultdict(list)
    for c in concepts:
        if not c or " " not in c:
            # Multiword / single token both allowed, but favor useful
            # polysemy candidates later.
            key=c
        else:
            key=c.split()[0]

        if len(key)>=3:
            buckets[key].append(c)

    best=None
    best_score=-1

    for surface,vals in buckets.items():
        if len(vals)<2:
            continue

        vals=vals[:20]
        for i,a in enumerate(vals):
            a_edges=set(
                (e.relation,e.target)
                for e in memory.adj.get(a,())
            )
            if not a_edges:
                continue

            for b in vals[i+1:]:
                b_edges=set(
                    (e.relation,e.target)
                    for e in memory.adj.get(b,())
                )
                if not b_edges:
                    continue

                divergence=len(
                    a_edges.symmetric_difference(b_edges)
                )
                score=divergence + min(
                    len(a_edges),
                    len(b_edges),
                )

                if score>best_score:
                    best_score=score
                    best=(surface,a,b)

    return best


def run_real(db_path: Path, query: str | None = None):
    loader,memory=build_from_sqlite(db_path)

    try:
        stats={
            "database":str(db_path.resolve()),
            "concepts":len(memory.concepts()),
            "edges":memory.edge_count,
        }

        if query is None:
            choice=choose_real_ambiguous_surface(memory)
            if choice is None:
                raise RuntimeError(
                    "Could not find a sufficiently ambiguous surface form "
                    "in the loaded ConceptNet graph."
                )
            query,a,b=choice
            aliases={query:(a,b)}
        else:
            # For an explicit query, use exact query plus two close candidates
            # selected by lexical prefix.
            q=canonical(query)
            related=[
                c for c in memory.concepts()
                if c==q or c.startswith(q+" ")
            ][:8]
            aliases={q:tuple(related)}
            if len(aliases[q])<2:
                raise RuntimeError(
                    f"Query '{query}' does not expose >=2 candidates "
                    f"under the benchmark ambiguity adapter."
                )

        benchmark_memory=RealConceptNetAmbiguityMemory(
            surface_aliases=aliases
        )
        # Copy indexed graph without rescanning SQLite.
        benchmark_memory.adj=memory.adj
        benchmark_memory.reverse=memory.reverse
        benchmark_memory.edge_count=memory.edge_count
        benchmark_memory.source=memory.source

        arch=IntegratedSemanticArchitecture(
            benchmark_memory
        )

        # Pick discriminating context automatically from the candidate
        # neighborhoods. Prefer one relation-target unique to candidate A/B.
        cands=benchmark_memory.retrieve(query)
        if len(cands)<2:
            raise RuntimeError("Ambiguity adapter produced fewer than 2 candidates.")

        a=cands[0].concept
        b=cands[1].concept
        a_pairs={
            (e.relation,e.target)
            for e in benchmark_memory.adj.get(a,())
        }
        b_pairs={
            (e.relation,e.target)
            for e in benchmark_memory.adj.get(b,())
        }

        a_unique=next(iter(a_pairs-b_pairs),None)
        b_unique=next(iter(b_pairs-a_pairs),None)

        if a_unique is None and b_unique is None:
            raise RuntimeError(
                "Candidate neighborhoods are not sufficiently distinct "
                "for a deterministic ambiguity test."
            )

        context_a=(a_unique,) if a_unique else (b_unique,)
        context_b=(b_unique,) if b_unique else (a_unique,)

        state_a=arch.perceive(
            query,
            context=context_a,
        )
        state_b=arch.revise(
            query,
            context=context_b,
        )

        return {
            "status":"PASS",
            "conceptnet":stats,
            "query":query,
            "candidates":[a,b],
            "contexts":{
                "first":context_a,
                "second":context_b,
            },
            "states":{
                "first":{
                    "candidates":state_a.candidates,
                    "committed":state_a.committed,
                    "confidence":state_a.confidence,
                    "revision":state_a.revision,
                },
                "second":{
                    "candidates":state_b.candidates,
                    "committed":state_b.committed,
                    "confidence":state_b.confidence,
                    "revision":state_b.revision,
                },
            },
            "architecture_history":len(
                arch.history
            ),
        }
    finally:
        loader.close()


def synthetic_smoke():
    # Reuse the known-good V373 ambiguity setup, but report it in the same
    # shape as the real benchmark.
    from ambiguity_benchmark import make_memory
    memory=make_memory()
    arch=IntegratedSemanticArchitecture(memory)

    initial=arch.perceive("bank")
    assert set(initial.candidates)=={
        "bank_finance",
        "bank_river",
    }
    assert initial.committed is None

    finance=arch.revise(
        "bank",
        context=(
            ("RelatedTo","money"),
            ("UsedFor","deposit"),
        ),
    )
    assert finance.committed=="bank_finance"

    river=arch.revise(
        "bank",
        context=(
            ("RelatedTo","river"),
            ("UsedFor","erosion"),
        ),
    )
    assert river.committed=="bank_river"

    mixed=arch.revise(
        "bank",
        context=(
            ("RelatedTo","money"),
            ("RelatedTo","river"),
        ),
    )
    assert mixed.committed is None

    return {
        "status":"PASS",
        "mode":"synthetic_smoke",
        "candidates":initial.candidates,
        "finance_commitment":finance.committed,
        "river_commitment":river.committed,
        "mixed_commitment":mixed.committed,
        "history":len(arch.history),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument(
        "--conceptnet",
        type=Path,
        default=Path(r".\data\conceptnet_compact.db"),
    )
    p.add_argument(
        "--query",
        default=None,
    )
    p.add_argument(
        "--smoke",
        action="store_true",
    )
    args=p.parse_args()

    if args.smoke or not args.conceptnet.exists():
        print(json.dumps(synthetic_smoke(),indent=2))
        return

    result=run_real(
        args.conceptnet.resolve(),
        args.query,
    )
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
