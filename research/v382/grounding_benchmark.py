
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import math
import random

from semantic_memory import IndexedSemanticMemory, SemanticEdge
from semantic_architecture import IntegratedSemanticArchitecture


@dataclass(frozen=True)
class GroundingCase:
    case_id: str
    query: str
    candidates: Tuple[str, ...]
    context: Tuple[Tuple[str, str], ...]
    expected: Optional[str]


def benchmark_memory() -> IndexedSemanticMemory:
    """
    Deterministic synthetic semantic substrate with many ambiguity families.
    Every family has two candidates with disjoint contextual cues.
    """
    edges = []

    def add_family(surface, a, b, a_context, b_context):
        edges.extend([
            SemanticEdge(a, "IsA", f"{surface}_type_a"),
            SemanticEdge(a, "RelatedTo", a_context),
            SemanticEdge(a, "UsedFor", a_context + "_use"),
            SemanticEdge(b, "IsA", f"{surface}_type_b"),
            SemanticEdge(b, "RelatedTo", b_context),
            SemanticEdge(b, "UsedFor", b_context + "_use"),
        ])

    add_family("bank", "bank_finance", "bank_river", "money", "river")
    add_family("bat", "bat_animal", "bat_sports", "animal", "ball")
    add_family("crane", "crane_bird", "crane_machine", "bird", "construction")
    add_family("seal", "seal_animal", "seal_stamp", "animal", "document")
    add_family("club", "club_group", "club_weapon", "membership", "weapon")
    add_family("match", "match_fire", "match_game", "flame", "competition")
    add_family("plant", "plant_living", "plant_factory", "growth", "production")
    add_family("watch", "watch_device", "watch_action", "time", "observe")
    add_family("light", "light_photon", "light_object", "illumination", "carry")
    add_family("spring", "spring_season", "spring_mechanism", "weather", "mechanics")

    return AmbiguousMemory.from_edges(edges)


class AmbiguousMemory(IndexedSemanticMemory):
    """
    Surface aliases are benchmark metadata. The semantic decisions still use
    the indexed graph and its relation structure.
    """

    aliases: Dict[str, Tuple[str, ...]] = {
        "bank": ("bank_finance", "bank_river"),
        "bat": ("bat_animal", "bat_sports"),
        "crane": ("crane_bird", "crane_machine"),
        "seal": ("seal_animal", "seal_stamp"),
        "club": ("club_group", "club_weapon"),
        "match": ("match_fire", "match_game"),
        "plant": ("plant_living", "plant_factory"),
        "watch": ("watch_device", "watch_action"),
        "light": ("light_photon", "light_object"),
        "spring": ("spring_season", "spring_mechanism"),
    }

    def retrieve(self, query, max_candidates=8, hop=1):
        q=query.strip().lower()
        if q in self.aliases:
            from semantic_memory import SemanticCandidate
            names=self.aliases[q]
            prior=1.0/len(names)
            return tuple(
                SemanticCandidate(
                    concept=n,
                    prior=prior,
                    evidence=self.neighborhood(n,max_edges=16),
                )
                for n in names[:max_candidates]
            )
        return super().retrieve(query,max_candidates,hop)


def make_cases(seed=375):
    rnd=random.Random(seed)
    surfaces=list(AmbiguousMemory.aliases)
    cases=[]

    for i,surface in enumerate(surfaces):
        candidates=AmbiguousMemory.aliases[surface]
        a,b=candidates

        # Balanced positive cases.
        contexts=[
            (
                (("RelatedTo", a.split("_",1)[1]
                  if "_" in a else a),),
                a,
            ),
            (
                (("RelatedTo", b.split("_",1)[1]
                  if "_" in b else b),),
                b,
            ),
        ]

        # Use the actual semantic context nodes represented by the graph.
        relation_targets={
            "bank_finance":"money",
            "bank_river":"river",
            "bat_animal":"animal",
            "bat_sports":"ball",
            "crane_bird":"bird",
            "crane_machine":"construction",
            "seal_animal":"animal",
            "seal_stamp":"document",
            "club_group":"membership",
            "club_weapon":"weapon",
            "match_fire":"flame",
            "match_game":"competition",
            "plant_living":"growth",
            "plant_factory":"production",
            "watch_device":"time",
            "watch_action":"observe",
            "light_photon":"illumination",
            "light_object":"carry",
            "spring_season":"weather",
            "spring_mechanism":"mechanics",
        }

        for j,c in enumerate(candidates):
            target=relation_targets[c]
            cases.append(
                GroundingCase(
                    case_id=f"{surface}-{j}",
                    query=surface,
                    candidates=candidates,
                    context=(("RelatedTo",target),),
                    expected=c,
                )
            )

        # Ambiguous / contradictory cases should withdraw commitment.
        ca=relation_targets[a]
        cb=relation_targets[b]
        cases.append(
            GroundingCase(
                case_id=f"{surface}-ambiguous",
                query=surface,
                candidates=candidates,
                context=(
                    ("RelatedTo",ca),
                    ("RelatedTo",cb),
                ),
                expected=None,
            )
        )

        # Empty context tests prior-only uncertainty.
        cases.append(
            GroundingCase(
                case_id=f"{surface}-none",
                query=surface,
                candidates=candidates,
                context=(),
                expected=None,
            )
        )

    rnd.shuffle(cases)
    return cases


def evaluate_case(case: GroundingCase):
    memory=benchmark_memory()
    arch=IntegratedSemanticArchitecture(memory)

    state=arch.perceive(
        case.query,
        context=case.context,
    )

    correct=(
        case.expected is not None
        and state.committed==case.expected
    )

    uncertainty_correct=(
        case.expected is None
        and state.committed is None
    )

    return {
        "case_id":case.case_id,
        "query":case.query,
        "expected":case.expected,
        "candidates":state.candidates,
        "committed":state.committed,
        "confidence":state.confidence,
        "entropy":state.entropy,
        "correct_resolution":correct,
        "correct_uncertainty":uncertainty_correct,
        "revision":state.revision,
    }


def revision_case():
    memory=benchmark_memory()
    arch=IntegratedSemanticArchitecture(memory)

    first=arch.perceive(
        "bank",
        context=(("RelatedTo","money"),),
    )
    second=arch.revise(
        "bank",
        context=(("RelatedTo","river"),),
    )
    third=arch.revise(
        "bank",
        context=(
            ("RelatedTo","money"),
            ("RelatedTo","river"),
        ),
    )

    return {
        "first":first,
        "second":second,
        "third":third,
        "passed":(
            first.committed=="bank_finance"
            and second.committed=="bank_river"
            and third.committed is None
            and second.revision==2
            and third.revision==3
        ),
    }


def summarize(results):
    n=len(results)
    resolved=[x for x in results if x["expected"] is not None]
    unresolved=[x for x in results if x["expected"] is None]

    return {
        "cases":n,
        "resolved_cases":len(resolved),
        "uncertain_cases":len(unresolved),
        "resolution_accuracy":(
            sum(int(x["correct_resolution"]) for x in resolved)
            /max(1,len(resolved))
        ),
        "uncertainty_accuracy":(
            sum(int(x["correct_uncertainty"]) for x in unresolved)
            /max(1,len(unresolved))
        ),
        "mean_confidence":(
            sum(x["confidence"] for x in results)/max(1,n)
        ),
        "mean_entropy":(
            sum(x["entropy"] for x in results)/max(1,n)
        ),
        "candidate_recall":(
            sum(
                int(x["expected"] in x["candidates"])
                for x in resolved
            )/max(1,len(resolved))
        ),
    }


def smoke():
    cases=make_cases(375)
    results=[evaluate_case(c) for c in cases]

    rev=revision_case()
    assert rev["passed"]

    summary=summarize(results)

    assert summary["candidate_recall"]==1.0
    assert summary["resolution_accuracy"]==1.0
    assert summary["uncertainty_accuracy"]==1.0

    return {
        "status":"PASS",
        "benchmark":"V375_grounding_benchmark",
        "summary":summary,
        "revision_test":{
            "passed":rev["passed"],
            "first_commit":rev["first"].committed,
            "second_commit":rev["second"].committed,
            "third_commit":rev["third"].committed,
        },
        "sample_cases":results[:8],
    }


def main():
    print("V375 — Grounding Benchmark")
    print("===========================")
    result=smoke()
    print(json.dumps(result,indent=2,default=str))


if __name__=="__main__":
    main()
