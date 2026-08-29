
from __future__ import annotations

import argparse
import json
from pathlib import Path

from grammar_cognitive import GrammarHypothesisSpace, GrammarGroundingInterface, smoke_examples
from semantic_memory import IndexedSemanticMemory, SemanticEdge
from semantic_architecture import IntegratedSemanticArchitecture


def smoke_memory():
    return IndexedSemanticMemory.from_edges([
        SemanticEdge("dog","IsA","animal"),
        SemanticEdge("cat","IsA","animal"),
        SemanticEdge("chases","RelatedTo","pursuit"),
        SemanticEdge("eats","RelatedTo","food"),
    ])


def smoke():
    memory=smoke_memory()
    semantic=IntegratedSemanticArchitecture(memory)

    space=GrammarHypothesisSpace()
    interface=GrammarGroundingInterface(semantic)

    results=[]
    for example in smoke_examples():
        hypotheses=space.generate(example.sentence)

        assert hypotheses
        belief=interface.interpret_sentence(
            example.sentence,
            hypotheses,
            example.queries,
        )

        assert belief is not None
        assert belief.committed is not None
        assert any(
            h.construction[0:3]
            ==("DET","NOUN","VERB")
            for h in belief.candidates
        )

        results.append(
            {
                "sentence":example.sentence,
                "hypotheses":[
                    h.hypothesis_id
                    for h in belief.candidates
                ],
                "committed":belief.committed,
                "confidence":belief.confidence,
                "entropy":belief.entropy,
            }
        )

    # Architecture-level checks:
    # the semantic subsystem is actually invoked by the grammar interface.
    assert len(semantic.history) >= 4
    assert len(interface.evidence) == 2

    print("V378 grammar integration smoke: PASS")
    print("explicit grammar hypotheses: PASS")
    print("grammar hypothesis competition: PASS")
    print("semantic grounding through cognitive architecture: PASS")
    print("grammar -> semantic evidence: PASS")
    print("architecture history populated: PASS")
    print("explicit belief state: PASS")
    print("results:")
    print(json.dumps(results,indent=2))

    return {
        "status":"PASS",
        "examples":len(results),
        "architecture_events":len(semantic.history),
        "grammar_hypotheses":len(space.hypotheses),
        "results":results,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument(
        "--smoke",
        action="store_true",
    )
    args=p.parse_args()

    smoke()


if __name__=="__main__":
    main()
