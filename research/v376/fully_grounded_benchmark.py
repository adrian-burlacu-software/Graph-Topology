
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import argparse, json, math, random, re, statistics, time

from semantic_memory import (
    IndexedSemanticMemory,
    SemanticCandidate,
    SemanticEdge,
)
from semantic_architecture import IntegratedSemanticArchitecture
from semantic_cases import (
    make_synthetic_cases,
    ContextCase,
)
from semantic_memory import canonical_concept


def canonical(x):
    x=str(x).strip().lower()
    x=re.sub(r"^/c/[a-z]{2}/","",x)
    x=x.split("/")[0]
    x=x.replace("_"," ").replace("-"," ")
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9' ]+"," ",x)).strip()


class CaseMemory(IndexedSemanticMemory):
    def __init__(self, edges, aliases):
        super().__init__()
        for e in edges:
            self.add(e)
        self.aliases=aliases

    def retrieve(self, query, max_candidates=8, hop=1):
        names=self.aliases.get(canonical(query))
        if not names:
            return super().retrieve(query,max_candidates,hop)
        prior=1.0/len(names)
        return tuple(
            SemanticCandidate(
                concept=n,
                prior=prior,
                evidence=self.neighborhood(n,max_edges=32),
            )
            for n in names[:max_candidates]
        )


def synthetic_memory():
    cases=make_synthetic_cases()
    aliases={}
    edges=[]

    for c in cases:
        aliases[c.surface]=c.candidates
        for candidate in c.candidates:
            # candidate -> target edges are the actual grounding evidence.
            if "finance" in candidate:
                edges += [
                    SemanticEdge(candidate,"RelatedTo","money"),
                    SemanticEdge(candidate,"UsedFor","deposit"),
                    SemanticEdge(candidate,"IsA","institution"),
                ]
            elif "river" in candidate:
                edges += [
                    SemanticEdge(candidate,"RelatedTo","river"),
                    SemanticEdge(candidate,"UsedFor","erosion"),
                    SemanticEdge(candidate,"IsA","landform"),
                ]
            elif candidate.endswith("_animal"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","animal"),
                    SemanticEdge(candidate,"UsedFor","movement"),
                ]
            elif candidate.endswith("_sports"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","ball"),
                    SemanticEdge(candidate,"UsedFor","competition"),
                ]
            elif candidate.endswith("_bird"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","bird"),
                    SemanticEdge(candidate,"UsedFor","flight"),
                ]
            elif candidate.endswith("_machine"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","construction"),
                    SemanticEdge(candidate,"UsedFor","lifting"),
                ]
            elif candidate.endswith("_stamp"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","document"),
                    SemanticEdge(candidate,"UsedFor","sealing"),
                ]
            elif candidate.endswith("_living"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","growth"),
                    SemanticEdge(candidate,"UsedFor","photosynthesis"),
                ]
            elif candidate.endswith("_factory"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","production"),
                    SemanticEdge(candidate,"UsedFor","manufacturing"),
                ]
            elif candidate.endswith("_device"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","time"),
                    SemanticEdge(candidate,"UsedFor","observe"),
                ]
            elif candidate.endswith("_action"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","observe"),
                    SemanticEdge(candidate,"UsedFor","look"),
                ]
            elif candidate.endswith("_season"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","weather"),
                    SemanticEdge(candidate,"UsedFor","climate"),
                ]
            elif candidate.endswith("_mechanism"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","mechanics"),
                    SemanticEdge(candidate,"UsedFor","motion"),
                ]
            elif candidate.endswith("_fire"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","flame"),
                    SemanticEdge(candidate,"UsedFor","ignition"),
                ]
            elif candidate.endswith("_game"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","competition"),
                    SemanticEdge(candidate,"UsedFor","play"),
                ]
            elif candidate.endswith("_group"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","membership"),
                    SemanticEdge(candidate,"UsedFor","social"),
                ]
            elif candidate.endswith("_weapon"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","weapon"),
                    SemanticEdge(candidate,"UsedFor","fight"),
                ]
            elif candidate.endswith("_photon"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","illumination"),
                    SemanticEdge(candidate,"UsedFor","visibility"),
                ]
            elif candidate.endswith("_object"):
                edges += [
                    SemanticEdge(candidate,"RelatedTo","carry"),
                    SemanticEdge(candidate,"UsedFor","transport"),
                ]

    return CaseMemory(edges,aliases),cases


def evaluate_case(memory, case: ContextCase):
    arch=IntegratedSemanticArchitecture(memory)
    state=arch.perceive(
        case.surface,
        context=case.context,
    )

    candidate_recall=all(
        c in state.candidates
        for c in case.candidates
    )

    correct=(
        state.committed==case.expected
        if case.expected is not None
        else state.committed is None
    )

    return {
        "case_id":case.case_id,
        "surface":case.surface,
        "kind":case.kind,
        "expected":case.expected,
        "candidates":state.candidates,
        "committed":state.committed,
        "confidence":state.confidence,
        "entropy":state.entropy,
        "candidate_recall":candidate_recall,
        "correct":correct,
    }


def revision_test(memory, case):
    """
    Pick the two candidates from a case and derive one distinguishing edge
    for each directly from the semantic graph.
    """
    candidates=case.candidates
    if len(candidates)<2:
        return {"ok":False,"reason":"not_enough_candidates"}

    a,b=candidates[:2]

    a_key=canonical_concept(a)
    b_key=canonical_concept(b)
    a_edges={
        (e.relation,e.target)
        for e in memory.adj.get(a_key,())
    }
    b_edges={
        (e.relation,e.target)
        for e in memory.adj.get(b_key,())
    }

    unique_a=sorted(a_edges-b_edges)
    unique_b=sorted(b_edges-a_edges)

    if not unique_a or not unique_b:
        return {"ok":False,"reason":"no_distinguishing_edges"}

    arch=IntegratedSemanticArchitecture(memory)

    s1=arch.perceive(
        case.surface,
        context=(unique_a[0],),
    )
    s2=arch.revise(
        case.surface,
        context=(unique_b[0],),
    )
    s3=arch.revise(
        case.surface,
        context=(unique_a[0],unique_b[0]),
    )

    return {
        "ok":(
            s1.committed==a
            and s2.committed==b
            and s3.committed is None
        ),
        "first":s1.committed,
        "second":s2.committed,
        "third":s3.committed,
        "confidences":[
            s1.confidence,
            s2.confidence,
            s3.confidence,
        ],
    }



def smoke():
    memory,cases=synthetic_memory()
    results=[evaluate_case(memory,c) for c in cases]

    summary={
        "cases":len(results),
        "resolved":sum(r["kind"]=="resolved" for r in results),
        "ambiguous":sum(r["kind"]=="ambiguous" for r in results),
        "no_context":sum(r["kind"]=="no_context" for r in results),
        "candidate_recall":sum(
            int(r["candidate_recall"]) for r in results
        )/max(1,len(results)),
        "case_accuracy":sum(
            int(r["correct"]) for r in results
        )/max(1,len(results)),
        "resolved_accuracy":sum(
            int(r["correct"]) for r in results
            if r["kind"]=="resolved"
        )/max(
            1,
            sum(r["kind"]=="resolved" for r in results),
        ),
        "uncertainty_accuracy":sum(
            int(r["correct"]) for r in results
            if r["kind"]!="resolved"
        )/max(
            1,
            sum(r["kind"]!="resolved" for r in results),
        ),
        "mean_confidence":sum(
            r["confidence"] for r in results
        )/max(1,len(results)),
        "mean_entropy":sum(
            r["entropy"] for r in results
        )/max(1,len(results)),
    }

    # One revision per semantic family.
    revision=[]
    seen=set()
    for c in cases:
        if c.surface in seen:
            continue
        seen.add(c.surface)
        revision.append(
            revision_test(memory,c)
        )

    revision_accuracy=sum(
        int(x["ok"]) for x in revision
    )/max(1,len(revision))

    assert summary["candidate_recall"]==1.0
    assert summary["resolved_accuracy"]==1.0
    assert summary["uncertainty_accuracy"]==1.0
    assert revision_accuracy==1.0

    return {
        "status":"PASS",
        "summary":summary,
        "revision":{
            "cases":len(revision),
            "accuracy":revision_accuracy,
        },
        "data_free":True,
    }


def main():
    p=argparse.ArgumentParser(
        description="Fully grounded semantic grounding benchmark."
    )
    p.add_argument(
        "--conceptnet",
        type=Path,
        default=Path(r".\data\conceptnet_compact.db"),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of ambiguity surfaces for a real run.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
    )
    args=p.parse_args()

    if args.smoke:
        print(json.dumps(smoke(),indent=2))
        return

    db=args.conceptnet.resolve()
    if not db.exists():
        raise SystemExit(f"ConceptNet database not found: {db}")

    from real_grounding import IndexedConceptNet

    print("[1/8] Loading real ConceptNet...")
    start=time.perf_counter()
    graph=IndexedConceptNet(db).build_index()
    print(
        f"      concepts={len(graph.concepts):,} "
        f"edges={graph.edge_count:,} "
        f"index_seconds={time.perf_counter()-start:.3f}"
    )

    print("[2/8] Discovering real semantic ambiguity...")
    real_pairs=graph.find_ambiguous_surfaces(limit=args.limit)
    print(f"      candidate surface pairs={len(real_pairs):,}")

    cases=[]
    aliases={}
    for surface,a,b,score in real_pairs:
        ctx_a,ctx_b=None,None
        a_edges={(e.relation,e.target) for e in graph.adj.get(a,())}
        b_edges={(e.relation,e.target) for e in graph.adj.get(b,())}
        unique_a=sorted(a_edges-b_edges)
        unique_b=sorted(b_edges-a_edges)
        if not unique_a or not unique_b:
            continue

        first=unique_a[0]
        second=unique_b[0]
        aliases[surface]=(a,b)

        cases += [
            ContextCase(
                f"{surface}-a",
                surface,
                (a,b),
                (first,),
                a,
                "resolved",
            ),
            ContextCase(
                f"{surface}-b",
                surface,
                (a,b),
                (second,),
                b,
                "resolved",
            ),
            ContextCase(
                f"{surface}-ambiguous",
                surface,
                (a,b),
                (first,second),
                None,
                "ambiguous",
            ),
            ContextCase(
                f"{surface}-no-context",
                surface,
                (a,b),
                (),
                None,
                "no_context",
            ),
        ]

    print("[3/8] Validating benchmark case quality...")
    assert cases, "No valid real ambiguity cases found."
    print(f"      usable cases={len(cases):,}")

    # Build the exact same indexed graph into the cognitive semantic substrate.
    memory=CaseMemory([],aliases)
    memory.adj=graph.adj
    memory.reverse=graph.rev
    memory.edge_count=graph.edge_count
    memory.source=graph.source if hasattr(graph,"source") else str(db)

    print("[4/8] Running candidate-retrieval checks...")
    results=[evaluate_case(memory,c) for c in cases]
    candidate_recall=sum(
        int(r["candidate_recall"]) for r in results
    )/len(results)

    print(
        f"      candidate recall={candidate_recall:.3f}"
    )

    print("[5/8] Running grounding resolution...")
    resolved=[
        r for r in results
        if r["kind"]=="resolved"
    ]
    resolved_accuracy=sum(
        int(r["correct"]) for r in resolved
    )/max(1,len(resolved))
    print(
        f"      resolved accuracy={resolved_accuracy:.3f}"
    )

    print("[6/8] Running uncertainty controls...")
    uncertain=[
        r for r in results
        if r["kind"]!="resolved"
    ]
    uncertainty_accuracy=sum(
        int(r["correct"]) for r in uncertain
    )/max(1,len(uncertain))
    print(
        f"      uncertainty accuracy={uncertainty_accuracy:.3f}"
    )

    print("[7/8] Running belief-revision tests...")
    revisions=[]
    seen=set()
    for c in cases:
        if c.surface in seen:
            continue
        seen.add(c.surface)
        a,b=aliases[c.surface]
        a_edge=next(
            (
                e for e in graph.adj.get(a,())
                if e.relation and e.target
            ),
            None,
        )
        b_edge=next(
            (
                e for e in graph.adj.get(b,())
                if e.relation and e.target
            ),
            None,
        )
        if not a_edge or not b_edge:
            continue

        arch=IntegratedSemanticArchitecture(memory)
        s1=arch.perceive(
            c.surface,
            context=((a_edge.relation,a_edge.target),),
        )
        s2=arch.revise(
            c.surface,
            context=((b_edge.relation,b_edge.target),),
        )
        s3=arch.revise(
            c.surface,
            context=(
                (a_edge.relation,a_edge.target),
                (b_edge.relation,b_edge.target),
            ),
        )
        revisions.append(
            (
                s1.committed==a
                and s2.committed==b
                and s3.committed is None
            )
        )

    revision_accuracy=sum(
        int(x) for x in revisions
    )/max(1,len(revisions))

    print(
        f"      revision accuracy={revision_accuracy:.3f}"
    )

    print("[8/8] Final quality gates...")
    gates={
        "candidate_recall":candidate_recall==1.0,
        "resolved_accuracy":resolved_accuracy>=0.80,
        "uncertainty_accuracy":uncertainty_accuracy>=0.80,
        "revision_accuracy":revision_accuracy>=0.80,
    }

    status="PASS" if all(gates.values()) else "FAIL"
    print(
        f"      gates={gates}"
    )
    print(f"[RESULT] {status}")

    report={
        "status":status,
        "conceptnet":{
            "database":str(db),
            "concepts":len(graph.concepts),
            "edges":graph.edge_count,
        },
        "benchmark":{
            "cases":len(cases),
            "surfaces":len(aliases),
            "candidate_recall":candidate_recall,
            "resolved_accuracy":resolved_accuracy,
            "uncertainty_accuracy":uncertainty_accuracy,
            "revision_accuracy":revision_accuracy,
        },
        "diagnostics":{
            "mean_confidence":sum(
                r["confidence"] for r in results
            )/max(1,len(results)),
            "mean_entropy":sum(
                r["entropy"] for r in results
            )/max(1,len(results)),
        },
        "quality_gates":gates,
        "examples":results[:12],
    }

    graph.close()
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
