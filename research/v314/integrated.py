
from __future__ import annotations

from richer_cognition import (
    PersistentMemory,
    TransformDynamics,
)
from hierarchical import (
    HierarchicalAdaptiveRepresentation,
    CONFIGS as REPRESENTATION_CONFIGS,
)
from hypothesis_competition import *  # compatibility not used

from structured_memory import (
    StructuredWorkingMemory,
    CONFIGS as STRUCTURED_CONFIGS,
)


class IntegratedSystem:
    def __init__(self, name):
        self.memory = PersistentMemory()
        self.dynamics = TransformDynamics()

        self.structured = StructuredWorkingMemory(
            STRUCTURED_CONFIGS[name]
        )

        self.representation = (
            HierarchicalAdaptiveRepresentation(
                REPRESENTATION_CONFIGS[
                    "hier_protective"
                ]
            )
        )

        # Tiny internal hypothesis population.  This is intentionally simple:
        # V314's focus is representation, not another hypothesis sweep.
        self.hypothesis_state = 0
        self.hypothesis_confidence = 0.0
        self.hypothesis_count = 0

    def _update_hypothesis(
        self,
        prediction: int,
        answer: int,
        episode,
    ):
        error = int(
            prediction != answer
        )

        changed = (
            episode.task == "rule_change"
            and episode.rule_version == 1
        )

        if changed and error:
            self.hypothesis_state ^= 1
            self.hypothesis_confidence = 1.0

        elif error:
            self.hypothesis_confidence = min(
                1.0,
                self.hypothesis_confidence + 0.10,
            )
            if self.hypothesis_confidence >= 0.60:
                self.hypothesis_state = 1

        else:
            self.hypothesis_confidence = min(
                1.0,
                self.hypothesis_confidence + 0.05,
            )

        self.hypothesis_count += 1

    def run(self, episode, learn=True):
        graph = episode.graph.clone()

        self.structured.inject_state(graph)
        self.representation.inject_state(graph)

        self.memory.observe(
            graph,
            episode.query,
        )

        transient = next(
            (
                name
                for name in list(graph.nodes)
                if graph.nodes[name].role
                == "initial_fact"
            ),
            None,
        )

        if transient is not None:
            graph.nodes.pop(
                transient
            )

        for step in range(
            1,
            episode.decision_step + 1,
        ):
            self.dynamics.step(
                graph,
                step,
            )
            self.memory.maintain(
                graph,
            )

        state=self.structured.construct(
            graph,
            episode,
            self.hypothesis_state,
            self.hypothesis_confidence,
        )

        decision=self.structured.decide(
            episode,
            self.hypothesis_state,
        )

        correct=(
            decision == episode.answer_bit
        )

        if learn:
            self.structured.feedback(
                decision,
                episode.answer_bit,
                episode,
            )
            self._update_hypothesis(
                decision,
                episode.answer_bit,
                episode,
            )

        return {
            "correct":correct,
            "decision":decision,
            "answer":episode.answer_bit,
            "memory":state.memory_bit,
            "selected_cue":state.selected_cue,
            "hypothesis":self.hypothesis_state,
            "hypothesis_confidence":self.hypothesis_confidence,
        }
