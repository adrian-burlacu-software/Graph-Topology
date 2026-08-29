
from __future__ import annotations

from richer_cognition import (
    PersistentMemory,
    TransformDynamics,
)
from hypothesis import (
    HypothesisRevision,
    CONFIGS as HYPOTHESIS_CONFIGS,
)
from alternate_settling import (
    ActiveQueryDualSettling,
    CONFIGS as ALTERNATE_CONFIGS,
)


class IntegratedSystem:
    """
    Frozen core plus dual-state settling.
    """

    def __init__(self, name):
        self.memory=PersistentMemory()
        self.dynamics=TransformDynamics()
        self.alternate=ActiveQueryDualSettling(
            ALTERNATE_CONFIGS[name]
        )
        self.hypothesis=HypothesisRevision(
            HYPOTHESIS_CONFIGS["fast_revision"]
        )

    def run(self, episode, learn=True):
        graph=episode.graph.clone()

        self.alternate.inject_state(graph)
        self.hypothesis.inject_state(graph)

        self.memory.observe(
            graph,
            episode.query,
        )

        transient=next(
            (
                name for name in list(graph.nodes)
                if graph.nodes[name].role=="initial_fact"
            ),
            None,
        )

        if transient is not None:
            graph.nodes.pop(transient)

        for step in range(
            1,
            episode.decision_step+1,
        ):
            self.dynamics.step(
                graph,
                step,
            )
            self.memory.maintain(
                graph,
            )

        decision=self.alternate.run(
            graph,
            episode,
        )

        decision=self.hypothesis.transform_decision(
            graph,
            decision,
            episode,
        )

        correct=(
            decision==episode.answer_bit
        )

        if learn:
            self.hypothesis.feedback(
                graph,
                decision,
                episode.answer_bit,
                episode,
            )

        return {
            "correct":correct,
            "decision":decision,
            "answer":episode.answer_bit,
            "actual_state":self.alternate.actual_state,
            "alternate_state":self.alternate.alternate_state,
            "selected_query":self.alternate.selected,
        }
