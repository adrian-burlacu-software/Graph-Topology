
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import argparse, json, math, time, re

from semantic_memory import IndexedSemanticMemory, SemanticCandidate, SemanticEdge
from semantic_architecture import IntegratedSemanticArchitecture
from semantic_memory import canonical_concept


@dataclass(frozen=True)
class GroundingCase:
    case_id: str
    surface: str
    candidates: tuple[str, ...]
    context: tuple[tuple[str, str], ...]
    expected: str | None
    kind: str
    pair_id: str


class CaseMemory(IndexedSemanticMemory):
    """Case-scoped semantic memory. Candidate identity is immutable per case."""

    def __init__(self, base_memory, candidates):
        super().__init__()
        self.adj=base_memory.adj
        self.reverse=getattr(
            base_memory,
            "reverse",
            getattr(base_memory, "rev", None),
        )
        if self.reverse is None:
            raise AttributeError(
                "Semantic graph adapter must expose "
                "'reverse' or 'rev' adjacency."
            )
        self.edge_count=base_memory.edge_count
        self.source=getattr(base_memory,"source","")
        self._concept_set=base_memory.concepts
        self._candidates=tuple(candidates)

    def concepts(self):
        return set(self._concept_set)

    def retrieve(self, query, max_candidates=8, hop=1):
        prior=1.0/max(1,len(self._candidates))
        return tuple(
            SemanticCandidate(
                concept=c,
                prior=prior,
                evidence=tuple(self.neighborhood(c,max_edges=32)),
            )
            for c in self._candidates[:max_candidates]
        )


def _pair_quality(graph, a, b):
    a_key=canonical_concept(a)
    b_key=canonical_concept(b)

    a_edges={
        (e.relation, canonical_concept(e.target))
        for e in graph.adj.get(a_key, ())
    }
    b_edges={
        (e.relation, canonical_concept(e.target))
        for e in graph.adj.get(b_key, ())
    }

    unique_a=a_edges-b_edges
    unique_b=b_edges-a_edges
    overlap=a_edges&b_edges

    return {
        "unique_a": unique_a,
        "unique_b": unique_b,
        "overlap": overlap,
        "a_degree": len(a_edges),
        "b_degree": len(b_edges),
    }


def discover_pairs(graph, surface_limit=25):
    """
    Discover *unique* surface/candidate pairs.

    Crucially, a surface may have many possible phrase extensions. We do not
    overwrite a global alias mapping. Every benchmark case gets its own frozen
    candidate pair.
    """
    buckets=defaultdict(list)
    for concept in graph.concepts:
        c=canonical_concept(concept)
        if not c or len(c)<3:
            continue

        # Only use phrase/sense alternatives where the bare surface is itself
        # a valid concept. This removes the previous "prefix bucket" artifact
        # where arbitrary longer concepts were treated as senses by default.
        surface=c.split(" ",1)[0]
        if surface != c and surface in graph.concepts:
            buckets[surface].append(c)

    candidates=[]
    for surface, concepts in buckets.items():
        unique=sorted(set(concepts))
        if len(unique)<1:
            continue

        # Compare the bare sense against phrase candidates, selecting only
        # candidates with genuinely asymmetric semantic neighborhoods.
        bare=surface
        phrase_pairs=[]
        for c in unique:
            if c==bare:
                continue
            q=_pair_quality(graph,bare,c)
            if not q["unique_a"] or not q["unique_b"]:
                continue

            # Avoid pathological tiny differences and duplicate structural
            # pairs that share almost everything.
            asym=min(len(q["unique_a"]),len(q["unique_b"]))
            if asym<1:
                continue

            score=(
                2.0*asym
                +0.25*(q["a_degree"]+q["b_degree"])
                -0.10*len(q["overlap"])
            )
            phrase_pairs.append((score,bare,c,q))

        phrase_pairs.sort(
            key=lambda x:(-x[0],x[2])
        )

        for rank,(score,a,b,q) in enumerate(
            phrase_pairs[:surface_limit]
        ):
            candidates.append(
                (surface,a,b,score,q)
            )

        if len(candidates)>=surface_limit:
            break

    return candidates[:surface_limit]


def make_cases(graph, pairs):
    cases=[]
    quality=[]

    for pair_index,(surface,a,b,score,q) in enumerate(pairs):
        pair_id=f"{surface}::{a}::{b}"

        unique_a=sorted(q["unique_a"])
        unique_b=sorted(q["unique_b"])

        # Select one context feature from each side.
        ctx_a=unique_a[0]
        ctx_b=unique_b[0]

        # Case-local candidate set is immutable.
        cases.extend([
            GroundingCase(
                case_id=f"{pair_id}::a",
                surface=surface,
                candidates=(a,b),
                context=(ctx_a,),
                expected=a,
                kind="resolved",
                pair_id=pair_id,
            ),
            GroundingCase(
                case_id=f"{pair_id}::b",
                surface=surface,
                candidates=(a,b),
                context=(ctx_b,),
                expected=b,
                kind="resolved",
                pair_id=pair_id,
            ),
        ])

        # Before using an ambiguity control, verify that neither candidate wins
        # from the combined context.
        combined=(ctx_a,ctx_b)
        score_a=_context_score(graph,a,combined)
        score_b=_context_score(graph,b,combined)
        balance=max(score_a,score_b)-min(score_a,score_b)

        # Both senses satisfy one side; combined evidence should not establish
        # a clean winner. Otherwise this is not a valid ambiguity case.
        if (
            abs(score_a-score_b)<=0.5
            and score_a>0
            and score_b>0
        ):
            cases.append(
                GroundingCase(
                    case_id=f"{pair_id}::ambiguous",
                    surface=surface,
                    candidates=(a,b),
                    context=combined,
                    expected=None,
                    kind="ambiguous",
                    pair_id=pair_id,
                )
            )
            quality.append(
                {
                    "pair_id":pair_id,
                    "surface":surface,
                    "ambiguity_balance":balance,
                    "valid_ambiguity":True,
                }
            )

        # No-context control always expects uncertainty because priors are equal.
        cases.append(
            GroundingCase(
                case_id=f"{pair_id}::no-context",
                surface=surface,
                candidates=(a,b),
                context=(),
                expected=None,
                kind="no_context",
                pair_id=pair_id,
            )
        )

    return cases,quality


def _context_score(graph, candidate, context):
    score=0
    for relation,target in context:
        target=canonical_concept(target)
        if any(
            e.relation==relation
            and canonical_concept(e.target)==target
            for e in graph.adj.get(canonical_concept(candidate),())
        ):
            score+=1
    return score


def evaluate_case(base_memory, case):
    # Construct a fresh case-scoped memory so no other pair can contaminate it.
    memory=CaseMemory(
        base_memory,
        case.candidates,
    )
    arch=IntegratedSemanticArchitecture(memory)

    state=arch.perceive(
        case.surface,
        context=case.context,
    )

    candidate_recall=(
        tuple(state.candidates)==tuple(case.candidates)
        or set(case.candidates).issubset(set(state.candidates))
    )

    correct=(
        state.committed==case.expected
        if case.expected is not None
        else state.committed is None
    )

    return {
        "case_id":case.case_id,
        "pair_id":case.pair_id,
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


def revision_case(base_memory, case):
    a,b=case.candidates
    q=_pair_quality(base_memory,a,b)
    if not q["unique_a"] or not q["unique_b"]:
        return {"ok":False,"reason":"no_distinguishing_edges"}

    ctx_a=sorted(q["unique_a"])[0]
    ctx_b=sorted(q["unique_b"])[0]

    memory=CaseMemory(base_memory,(a,b))
    arch=IntegratedSemanticArchitecture(memory)

    s1=arch.perceive(
        case.surface,
        context=(ctx_a,),
    )
    s2=arch.revise(
        case.surface,
        context=(ctx_b,),
    )
    s3=arch.revise(
        case.surface,
        context=(ctx_a,ctx_b),
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


def synthetic_base():
    return IndexedSemanticMemory.from_edges([
        # Realistic sense pairs with distinct semantic neighborhoods.
        SemanticEdge("bank","IsA","institution"),
        SemanticEdge("bank","RelatedTo","money"),
        SemanticEdge("bank","UsedFor","deposit"),
        SemanticEdge("bank_river","IsA","landform"),
        SemanticEdge("bank_river","RelatedTo","river"),
        SemanticEdge("bank_river","UsedFor","erosion"),

        SemanticEdge("bat","IsA","animal"),
        SemanticEdge("bat","RelatedTo","animal"),
        SemanticEdge("bat","UsedFor","flight"),
        SemanticEdge("bat_sport","IsA","equipment"),
        SemanticEdge("bat_sport","RelatedTo","ball"),
        SemanticEdge("bat_sport","UsedFor","competition"),

        SemanticEdge("crane","IsA","bird"),
        SemanticEdge("crane","RelatedTo","bird"),
        SemanticEdge("crane","UsedFor","flight"),
        SemanticEdge("crane_machine","IsA","machine"),
        SemanticEdge("crane_machine","RelatedTo","construction"),
        SemanticEdge("crane_machine","UsedFor","lifting"),
    ])


def smoke():
    graph=synthetic_base()
    pairs=[
        ("bank","bank","bank_river"),
        ("bat","bat","bat_sport"),
        ("crane","crane","crane_machine"),
    ]

    cases=[]
    for surface,a,b in pairs:
        q=_pair_quality(graph,a,b)
        cases.extend([
            GroundingCase(
                f"{surface}::a",
                surface,
                (a,b),
                (sorted(q["unique_a"])[0],),
                a,"resolved",f"{surface}::{a}::{b}"
            ),
            GroundingCase(
                f"{surface}::b",
                surface,
                (a,b),
                (sorted(q["unique_b"])[0],),
                b,"resolved",f"{surface}::{a}::{b}"
            ),
            GroundingCase(
                f"{surface}::ambiguous",
                surface,
                (a,b),
                (
                    sorted(q["unique_a"])[0],
                    sorted(q["unique_b"])[0],
                ),
                None,"ambiguous",f"{surface}::{a}::{b}"
            ),
            GroundingCase(
                f"{surface}::no-context",
                surface,
                (a,b),
                (),
                None,"no_context",f"{surface}::{a}::{b}"
            ),
        ])

    results=[evaluate_case(graph,c) for c in cases]
    revisions=[
        revision_case(
            graph,
            next(c for c in cases if c.pair_id==f"{surface}::{a}::{b}")
        )
        for surface,a,b in pairs
    ]

    # Quality invariants: every case owns exactly its candidate set.
    case_sets={
        r["case_id"]:tuple(r["candidates"])
        for r in results
    }
    assert all(
        set(case.candidates)==set(
            case_sets[case.case_id]
        )
        for case in cases
    )
    assert all(r["candidate_recall"] for r in results)
    assert all(
        r["correct"]
        for r in results
        if r["kind"]=="resolved"
    )
    assert all(
        r["correct"]
        for r in results
        if r["kind"] in ("ambiguous","no_context")
    )
    assert all(r["ok"] for r in revisions)

    return {
        "status":"PASS",
        "cases":len(cases),
        "candidate_recall":sum(
            int(r["candidate_recall"]) for r in results
        )/len(results),
        "resolution_accuracy":sum(
            int(r["correct"]) for r in results
            if r["kind"]=="resolved"
        )/sum(
            int(r["kind"]=="resolved") for r in results
        ),
        "uncertainty_accuracy":sum(
            int(r["correct"]) for r in results
            if r["kind"] in ("ambiguous","no_context")
        )/sum(
            int(r["kind"] in ("ambiguous","no_context"))
            for r in results
        ),
        "revision_accuracy":sum(
            int(r["ok"]) for r in revisions
        )/len(revisions),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument(
        "--conceptnet",
        type=Path,
        default=Path(r".\data\conceptnet_compact.db"),
    )
    p.add_argument("--limit",type=int,default=25)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()

    if args.smoke:
        print(json.dumps(smoke(),indent=2))
        return

    db=args.conceptnet.resolve()
    if not db.exists():
        raise SystemExit(f"ConceptNet database not found: {db}")

    from real_grounding import IndexedConceptNet

    print("[1/9] Loading real ConceptNet...")
    t0=time.perf_counter()
    graph=IndexedConceptNet(db).build_index()
    print(
        f"      concepts={len(graph.concepts):,} "
        f"edges={graph.edge_count:,} "
        f"index_seconds={time.perf_counter()-t0:.3f}"
    )

    print("[2/9] Discovering unique semantic pairs...")
    pairs=discover_pairs(graph,args.limit)
    print(f"      unique candidate pairs={len(pairs):,}")

    print("[3/9] Constructing case-local candidate sets...")
    cases,quality=make_cases(graph,pairs)
    print(f"      usable cases={len(cases):,}")
    print(f"      valid ambiguity controls={len(quality):,}")

    print("[4/9] Checking benchmark integrity...")
    duplicate_ids=len({c.case_id for c in cases})==len(cases)
    immutable_case_sets=all(
        len(c.candidates)==2 and c.candidates[0]!=c.candidates[1]
        for c in cases
    )
    print(f"      unique case IDs={'PASS' if duplicate_ids else 'FAIL'}")
    print(
        f"      immutable candidate pairs="
        f"{'PASS' if immutable_case_sets else 'FAIL'}"
    )

    print("[5/9] Checking semantic adapter contract...")
    adapter_contract = all(
        hasattr(graph, attr)
        for attr in ("adj", "rev", "edge_count")
    )
    print(
        f"      ConceptNet adapter contract="
        f"{'PASS' if adapter_contract else 'FAIL'}"
    )

    print("[6/9] Candidate retrieval...")
    retrieval=[evaluate_case(graph,c) for c in cases]
    candidate_recall=sum(
        int(r["candidate_recall"])
        for r in retrieval
    )/max(1,len(retrieval))
    print(f"      candidate recall={candidate_recall:.3f}")

    print("[7/9] Grounding resolution...")
    resolved=[r for r in retrieval if r["kind"]=="resolved"]
    resolution_accuracy=sum(
        int(r["correct"]) for r in resolved
    )/max(1,len(resolved))
    print(f"      resolution accuracy={resolution_accuracy:.3f}")

    print("[8/9] Uncertainty controls...")
    uncertain=[
        r for r in retrieval
        if r["kind"] in ("ambiguous","no_context")
    ]
    uncertainty_accuracy=sum(
        int(r["correct"]) for r in uncertain
    )/max(1,len(uncertain))
    print(f"      uncertainty accuracy={uncertainty_accuracy:.3f}")

    print("[9/9] Belief revision...")
    revisions=[]
    seen=set()
    for c in cases:
        if c.pair_id in seen: continue
        seen.add(c.pair_id)
        revisions.append(
            revision_case(graph,c)
        )
    revision_accuracy=sum(
        int(r["ok"]) for r in revisions
    )/max(1,len(revisions))
    print(f"      revision accuracy={revision_accuracy:.3f}")

    print("[10/10] Quality gates...")
    gates={
        "candidate_recall":candidate_recall==1.0,
        "resolution_accuracy":resolution_accuracy>=0.80,
        "uncertainty_accuracy":uncertainty_accuracy>=0.80,
        "revision_accuracy":revision_accuracy>=0.80,
        "unique_case_ids":duplicate_ids,
        "immutable_case_pairs":immutable_case_sets,
        "nonzero_ambiguous_controls":len(quality)>0,
        "semantic_adapter_contract":adapter_contract,
    }
    status="PASS" if all(gates.values()) else "FAIL"
    print(f"      gates={gates}")
    print(f"[RESULT] {status}")

    report={
        "status":status,
        "conceptnet":{
            "database":str(db),
            "concepts":len(graph.concepts),
            "edges":graph.edge_count,
        },
        "benchmark":{
            "unique_pairs":len(pairs),
            "cases":len(cases),
            "candidate_recall":candidate_recall,
            "resolution_accuracy":resolution_accuracy,
            "uncertainty_accuracy":uncertainty_accuracy,
            "revision_accuracy":revision_accuracy,
        },
        "quality_gates":gates,
        "ambiguity_controls":quality[:20],
        "examples":retrieval[:20],
    }
    print(json.dumps(report,indent=2,default=str))
    graph.close()


if __name__=="__main__":
    main()
