
from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_memory import IndexedSemanticMemory, SemanticEdge
from semantic_architecture import IntegratedSemanticArchitecture
from roundtrip_cognitive import (
    SemanticFrame,
    BidirectionalRoundTripBenchmark,
)


def smoke_architecture():
    memory=IndexedSemanticMemory.from_edges([
        SemanticEdge("dog","IsA","animal"),
        SemanticEdge("cat","IsA","animal"),
        SemanticEdge("chases","RelatedTo","pursuit"),
        SemanticEdge("eats","RelatedTo","food"),
        SemanticEdge("sees","RelatedTo","vision"),
    ])
    return IntegratedSemanticArchitecture(memory)


def smoke():
    arch=smoke_architecture()
    bench=BidirectionalRoundTripBenchmark(arch)

    seed_sentences=[
        "the dog chases the cat",
        "the cat eats the dog",
        "the dog sees the cat",
    ]

    frames=[
        SemanticFrame(
            predicate="chases",
            arguments=(
                ("agent","dog"),
                ("patient","cat"),
            ),
        ),
        SemanticFrame(
            predicate="eats",
            arguments=(
                ("agent","cat"),
                ("patient","dog"),
            ),
        ),
        SemanticFrame(
            predicate="sees",
            arguments=(
                ("agent","dog"),
                ("patient","cat"),
            ),
        ),
    ]

    p2g=[
        bench.perception_then_generation(s)
        for s in seed_sentences
    ]
    g2p=[
        bench.generation_then_perception(f)
        for f in frames
    ]

    assert all(x["pass"] for x in p2g)
    assert all(x["pass"] for x in g2p)
    assert all(
        x["input"]==x["generated"]
        for x in p2g
    )
    assert all(
        x["generated"]==x["regenerated"]
        for x in g2p
    )
    assert len(arch.history)>=9

    result={
        "status":"PASS",
        "perception_to_generation_to_perception":{
            "cases":len(p2g),
            "accuracy":sum(
                int(x["pass"]) for x in p2g
            )/len(p2g),
        },
        "generation_to_perception_to_generation":{
            "cases":len(g2p),
            "accuracy":sum(
                int(x["pass"]) for x in g2p
            )/len(g2p),
        },
        "semantic_events":len(arch.history),
        "examples":{
            "p2g":p2g,
            "g2p":g2p,
        },
    }

    print("V383 bidirectional roundtrip smoke: PASS")
    print("perception → generation → perception: PASS")
    print("generation → perception → generation: PASS")
    print("semantic equivalence preservation: PASS")
    print("generation stability: PASS")
    print("cognitive semantic architecture active: PASS")
    print(json.dumps(result,indent=2,default=str))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument(
        "--smoke",
        action="store_true",
    )
    p.add_argument(
        "--conceptnet",
        type=Path,
        default=Path(
            r".\data\conceptnet_compact.db"
        ),
    )
    args=p.parse_args()

    # Real BabyLM integration is intentionally a separate next stage: V383
    # establishes the closed-loop invariant without allowing corpus coverage
    # to obscure roundtrip correctness.
    smoke()


if __name__=="__main__":
    main()
