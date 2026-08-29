
from semantic_memory import IndexedSemanticMemory, SemanticEdge
from semantic_architecture import IntegratedSemanticArchitecture


def main():
    memory=IndexedSemanticMemory.from_edges([
        SemanticEdge("/c/en/dog","IsA","animal"),
        SemanticEdge("/c/en/dog","CapableOf","chase"),
        SemanticEdge("animal","RelatedTo","life"),
        SemanticEdge("/c/en/cat","IsA","animal"),
        SemanticEdge("/c/en/cat","CapableOf","chase"),
        SemanticEdge("chase","RelatedTo","pursuit"),
        SemanticEdge("hound","IsA","animal"),
        SemanticEdge("hound","CapableOf","chase"),
        SemanticEdge("hound","HasProperty","loyal"),
    ])

    arch=IntegratedSemanticArchitecture(memory)

    dog=arch.perceive(
        "/c/en/dog",
        context=(("IsA","animal"),),
    )
    assert dog.committed=="dog"
    assert dog.confidence>=0.80

    # The graph itself now serves as the semantic context. A hypothesis that
    # satisfies IsA(animal) but not IsA(vehicle) must lose commitment.
    hound=arch.perceive(
        "hound",
        context=(("IsA","animal"),),
    )
    assert hound.committed=="hound"

    revised=arch.revise(
        "hound",
        context=(("IsA","vehicle"),),
    )
    assert revised.committed is None
    assert revised.revision==2

    # Canonical identity and typed relation lookup.
    assert memory.relation_exists(
        "dog","CapableOf","chase"
    )
    assert memory.relation_exists(
        "/c/en/cat","IsA","animal"
    )

    # Indexed retrieval returns only normalized concept identities.
    retrieved=arch.grounder.retrieve("dog")
    assert retrieved
    assert retrieved[0].concept=="dog"

    explanation=arch.explain("hound")
    assert len(explanation["history"])==2
    assert len(explanation["evidence"])>=2

    # Smoke database is compact and finite.
    assert memory.edge_count==9

    print("V372 semantic integration smoke: PASS")
    print("indexed semantic memory: PASS")
    print("canonical concept identity: PASS")
    print("typed relation retrieval: PASS")
    print("semantic consistency scoring: PASS")
    print("competing interpretation state: PASS")
    print("revision under conflicting evidence: PASS")
    print("native cognitive semantic state: PASS")
    print("explainable provenance/evidence: PASS")
    print("edges:",memory.edge_count)


if __name__=="__main__":
    main()
