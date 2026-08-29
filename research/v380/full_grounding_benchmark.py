
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import argparse,json,math,statistics,time

from real_grounding import IndexedConceptNet
from semantic_memory import IndexedSemanticMemory,SemanticCandidate
from semantic_architecture import IntegratedSemanticArchitecture


@dataclass(frozen=True)
class RealCase:
    case_id:str
    query:str
    candidates:tuple[str,...]
    target:str
    context:tuple[tuple[str,str],...]
    expected:Optional[str]


class RealAmbiguityMemory(IndexedSemanticMemory):
    def __init__(self,real_graph,aliases):
        super().__init__()
        self.adj=real_graph.adj
        self.reverse=real_graph.rev
        self.edge_count=real_graph.edge_count
        self.source=str(real_graph.db_path)
        self._concepts=real_graph.concepts
        self.aliases=aliases

    def concepts(self):
        return set(self._concepts)

    def retrieve(self,query,max_candidates=8,hop=1):
        q=query.lower().strip()
        names=self.aliases.get(q)
        if names:
            prior=1.0/len(names)
            return tuple(
                SemanticCandidate(
                    concept=c,
                    prior=prior,
                    evidence=tuple(self.neighborhood(c,max_edges=16)),
                )
                for c in names[:max_candidates]
            )
        return super().retrieve(query,max_candidates,hop)


def choose_context(graph,candidate_a,candidate_b):
    a={(e.relation,e.target) for e in graph.adj.get(candidate_a,())}
    b={(e.relation,e.target) for e in graph.adj.get(candidate_b,())}
    ua=sorted(a-b)
    ub=sorted(b-a)
    if not ua and not ub:
        return None,None
    first=(ua[0],) if ua else (ub[0],)
    second=(ub[0],) if ub else (ua[0],)
    return first,second


def build_cases(graph, max_surfaces=25):
    rows=[]
    for surface,a,b,score in graph.find_ambiguous_surfaces(limit=max_surfaces):
        cands=(a,b)
        ctx_a,ctx_b=choose_context(graph,a,b)
        if not ctx_a or not ctx_b: continue
        rows.append(RealCase(
            case_id=f"{surface}-a",
            query=surface,
            candidates=cands,
            target=a,
            context=ctx_a,
            expected=a,
        ))
        rows.append(RealCase(
            case_id=f"{surface}-b",
            query=surface,
            candidates=cands,
            target=b,
            context=ctx_b,
            expected=b,
        ))
        rows.append(RealCase(
            case_id=f"{surface}-ambiguous",
            query=surface,
            candidates=cands,
            target="",
            context=(ctx_a[0],ctx_b[0]),
            expected=None,
        ))
    return rows


def run_case(real_graph,case):
    memory=RealAmbiguityMemory(
        real_graph,
        {case.query:case.candidates},
    )
    arch=IntegratedSemanticArchitecture(memory)
    state=arch.perceive(
        case.query,
        context=case.context,
    )
    resolved=(case.expected is not None)
    ok=(
        state.committed==case.expected
        if resolved else state.committed is None
    )
    return {
        "case_id":case.case_id,
        "query":case.query,
        "candidates":state.candidates,
        "expected":case.expected,
        "committed":state.committed,
        "confidence":state.confidence,
        "entropy":state.entropy,
        "ok":ok,
    }


def revision_case(real_graph,case):
    memory=RealAmbiguityMemory(
        real_graph,{case.query:case.candidates}
    )
    arch=IntegratedSemanticArchitecture(memory)
    a,b=case.candidates
    ctx_a,ctx_b=choose_context(real_graph,a,b)
    if not ctx_a or not ctx_b:
        return {"ok":False,"reason":"no_distinct_contexts"}
    s1=arch.perceive(case.query,context=ctx_a)
    s2=arch.revise(case.query,context=ctx_b)
    s3=arch.revise(case.query,context=(ctx_a[0],ctx_b[0]))
    return {
        "ok":(
            s1.committed==a
            and s2.committed==b
            and s3.committed is None
        ),
        "first":s1.committed,
        "second":s2.committed,
        "third":s3.committed,
        "confidence":[s1.confidence,s2.confidence,s3.confidence],
    }


def benchmark(db_path:Path,limit:int=25):
    start=time.perf_counter()
    graph=IndexedConceptNet(db_path).build_index()
    load_time=time.perf_counter()-start

    cases=build_cases(graph,limit)
    results=[run_case(graph,c) for c in cases]

    # One revision test per surface.
    revision=[]
    seen=set()
    for c in cases:
        if c.query in seen: continue
        seen.add(c.query)
        revision.append(revision_case(graph,c))

    resolved=[r for r in results if r["expected"] is not None]
    ambiguous=[r for r in results if r["expected"] is None]

    report={
        "status":"PASS",
        "conceptnet":{
            "database":str(db_path.resolve()),
            "concepts":len(graph.concepts),
            "edges":graph.edge_count,
            "index_build_seconds":load_time,
        },
        "benchmark":{
            "cases":len(results),
            "resolved_cases":len(resolved),
            "ambiguity_cases":len(ambiguous),
            "candidate_recall":1.0 if all(
                r["expected"] in r["candidates"] for r in resolved
            ) else 0.0,
            "resolution_accuracy":(
                sum(int(r["ok"]) for r in resolved)/max(1,len(resolved))
            ),
            "uncertainty_accuracy":(
                sum(int(r["ok"]) for r in ambiguous)/max(1,len(ambiguous))
            ),
            "mean_confidence":sum(r["confidence"] for r in results)/max(1,len(results)),
            "mean_entropy":sum(r["entropy"] for r in results)/max(1,len(results)),
        },
        "revision":{
            "cases":len(revision),
            "accuracy":sum(int(r["ok"]) for r in revision)/max(1,len(revision)),
            "sample":revision[:10],
        },
        "examples":[
            r for r in results[:12]
        ],
        "wall_time_seconds":time.perf_counter()-start,
    }
    graph.close()
    return report


def smoke():
    # Keep V375's data-free mechanism test, but use a real IndexedSemanticMemory
    # with many ambiguity families.
    from grounding_benchmark import benchmark_memory, make_cases, revision_case
    memory=benchmark_memory()
    from grounding_benchmark import evaluate_case,summarize

    results=[evaluate_case(c) for c in make_cases(375)]
    rev=revision_case()
    summary=summarize(results)
    assert summary["candidate_recall"]==1.0
    assert summary["resolution_accuracy"]==1.0
    assert summary["uncertainty_accuracy"]==1.0
    assert rev["passed"]
    return {
        "status":"PASS",
        "mode":"data_free_smoke",
        "summary":summary,
        "revision_test":rev["passed"],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument(
        "--conceptnet",
        type=Path,
        default=Path(r".\data\conceptnet_compact.db"),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=25,
    )
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()

    if args.smoke:
        print(json.dumps(smoke(),indent=2))
        return

    db=args.conceptnet.resolve()
    if not db.exists():
        raise SystemExit(f"ConceptNet database not found: {db}")

    report=benchmark(db,args.limit)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
