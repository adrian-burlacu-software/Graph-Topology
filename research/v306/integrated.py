
from __future__ import annotations

from richer_cognition import (
    PersistentMemory,
    TransformDynamics,
)
from hypothesis import (
    HypothesisRevision,
    CONFIGS as HYPOTHESIS_CONFIGS,
)
from hypotheses import (
    CONFIGS as OVERLAY_CONFIGS,
    CLASSES as OVERLAY_CLASSES,
)


class IntegratedSystem:
    """
    V306 architecture:

      persistent memory
      + transform dynamics
      + alternate cognitive overlay
      + hypothesis/revision
    """

    def __init__(
        self,
        overlay_name,
    ):
        config=OVERLAY_CONFIGS[overlay_name]

        self.memory=PersistentMemory()
        self.dynamics=TransformDynamics()
        self.overlay=OVERLAY_CLASSES[
            overlay_name
        ](config)

        self.hypothesis=HypothesisRevision(
            HYPOTHESIS_CONFIGS[
                "fast_revision"
            ]
        )

    def run(self,episode,learn=True):
        graph=episode.graph.clone()

        self.overlay.inject_state(graph)
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
            graph.nodes.pop(
                transient
            )

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

        self.overlay.compute(
            graph,
            episode,
        )

        decision=0

        decision=self.overlay.transform_decision(
            graph,
            decision,
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
            self.overlay.feedback(
                graph,
                decision,
                episode.answer_bit,
                episode,
            )

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
        }
