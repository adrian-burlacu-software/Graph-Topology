
from __future__ import annotations

from semantic_memory import IndexedSemanticMemory, SemanticEdge, SemanticCandidate
from semantic_architecture import IntegratedSemanticArchitecture


class AmbiguousSemanticMemory(IndexedSemanticMemory):
    """
    Benchmark adapter: surface "bank" has two explicit graph concepts.
    """

    def retrieve(self,query,max_candidates=8,hop=1):
        if query.strip().lower()=="bank":
            return (
                SemanticCandidate(
                    concept="bank_finance",
                    prior=0.5,
                    evidence=self.neighborhood(
                        "bank_finance",
                        max_edges=16,
                    ),
                ),
                SemanticCandidate(
                    concept="bank_river",
                    prior=0.5,
                    evidence=self.neighborhood(
                        "bank_river",
                        max_edges=16,
                    ),
                ),
            )
        return super().retrieve(
            query,max_candidates,hop
        )


def make_memory():
    return AmbiguousSemanticMemory.from_edges([
        # Finance sense.
        SemanticEdge("bank_finance","IsA","institution"),
        SemanticEdge("bank_finance","RelatedTo","money"),
        SemanticEdge("bank_finance","UsedFor","deposit"),
        SemanticEdge("money","RelatedTo","finance"),
        SemanticEdge("deposit","RelatedTo","account"),

        # River sense.
        SemanticEdge("bank_river","IsA","landform"),
        SemanticEdge("bank_river","RelatedTo","river"),
        SemanticEdge("bank_river","UsedFor","erosion"),
        SemanticEdge("river","RelatedTo","water"),
        SemanticEdge("erosion","RelatedTo","sediment"),
        SemanticEdge("water","RelatedTo","flow"),

        # Cross-context disambiguators.
        SemanticEdge("finance","RelatedTo","account"),
        SemanticEdge("sediment","RelatedTo","riverbed"),
    ])


def main():
    memory=make_memory()
    arch=IntegratedSemanticArchitecture(memory)

    # Ambiguous in isolation: equal priors, no useful context.
    ambiguous=arch.perceive("bank")
    assert set(ambiguous.candidates)=={
        "bank_finance",
        "bank_river",
    }
    assert ambiguous.committed is None

    # Financial context should select finance sense.
    finance=arch.revise(
        "bank",
        context=(
            ("RelatedTo","money"),
            ("UsedFor","deposit"),
        ),
    )
    assert finance.committed=="bank_finance"
    assert finance.confidence>=0.80

    # River context should force revision to river sense.
    river=arch.revise(
        "bank",
        context=(
            ("RelatedTo","river"),
            ("UsedFor","erosion"),
        ),
    )
    assert river.committed=="bank_river"
    assert river.confidence>=0.80
    assert river.revision==3

    # Contradictory context to the committed river interpretation must remove
    # commitment rather than keep the stale model.
    conflict=arch.revise(
        "bank",
        context=(
            ("RelatedTo","money"),
            ("RelatedTo","river"),
        ),
    )
    # Mixed evidence should become uncertain instead of asserting certainty.
    assert conflict.committed is None

    explanation=arch.explain("bank")
    assert len(explanation["history"])==4
    assert len(explanation["evidence"])>=8

    print("V373 semantic ambiguity smoke: PASS")
    print("multiple candidate concepts: PASS")
    print("contextual graph consistency: PASS")
    print("belief competition: PASS")
    print("finance disambiguation: PASS")
    print("river disambiguation: PASS")
    print("revision across contexts: PASS")
    print("contradictory-context uncertainty: PASS")
    print("explicit evidence/history: PASS")


if __name__=="__main__":
    main()
