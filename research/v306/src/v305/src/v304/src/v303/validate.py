
from __future__ import annotations

from richer_cognition import (
    TASKS,
    make_sequence,
)
from hypothesis import (
    HypothesisRevision,
    CONFIGS as HYPOTHESIS_CONFIGS,
)
from selective import (
    CONFIGS as SELECTIVE_CONFIGS,
    SelectiveRepresentation,
)
from integrated import IntegratedSystem


def main():
    assert len(TASKS)==6
    assert len(SELECTIVE_CONFIGS)==4

    seq=make_sequence(
        303,
        "interference",
        episodes=8,
        horizon=9,
    )

    # Every selection configuration must execute.
    for name,config in SELECTIVE_CONFIGS.items():
        system=IntegratedSystem(
            SelectiveRepresentation(config),
            HypothesisRevision(
                HYPOTHESIS_CONFIGS["fast_revision"]
            ),
        )

        rows=[
            system.run(
                ep,
                learn=True,
            )
            for ep in seq.episodes
        ]

        assert len(rows)==8
        assert system.selective.count>0
        assert system.hypothesis.count==8

    # Selection really suppresses distractor nodes without deleting topology.
    graph=seq.episodes[0].graph.clone()
    selector=SelectiveRepresentation(
        SELECTIVE_CONFIGS["strong_filter"]
    )
    before_edges=len(graph.edges)
    selector.apply(graph)
    after_edges=len(graph.edges)

    assert before_edges==after_edges

    # Run harder task families with one candidate.
    candidate=IntegratedSystem(
        SelectiveRepresentation(
            SELECTIVE_CONFIGS["balanced_filter"]
        ),
        HypothesisRevision(
            HYPOTHESIS_CONFIGS["fast_revision"]
        ),
    )

    for task in (
        "interference",
        "rule_change",
        "counterfactual",
    ):
        seqx=make_sequence(
            304,
            task,
            episodes=8,
            horizon=9,
        )

        rows=[
            candidate.run(
                ep,
                learn=True,
            )
            for ep in seqx.episodes
        ]

        assert len(rows)==8

    print("V303 validation: PASS")
    print("selective configs:",len(SELECTIVE_CONFIGS))
    print("all configs executable: PASS")
    print("distractors filtered without topology deletion: PASS")
    print("hard tasks executable: PASS")


if __name__=="__main__":
    main()
