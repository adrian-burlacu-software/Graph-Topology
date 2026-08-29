
from __future__ import annotations

from richer_cognition import (
    PersistentMemory,
    TransformDynamics,
)
from structured_memory import (
    StructuredWorkingMemory,
    CONFIGS as STRUCTURED_CONFIGS,
)
from rule_schemas import (
    ExplicitRuleSchemaMemory,
    CONFIGS as SCHEMA_CONFIGS,
)


class IntegratedSystem:
    def __init__(self, name):
        self.memory = PersistentMemory()
        self.dynamics = TransformDynamics()

        self.structured = StructuredWorkingMemory(
            STRUCTURED_CONFIGS[
                "structured_balanced"
            ]
        )

        self.rules = ExplicitRuleSchemaMemory(
            SCHEMA_CONFIGS[name]
        )

    def run(self, episode, learn=True):
        graph = episode.graph.clone()

        self.rules.inject_state(
            graph,
            episode.task,
        )
        self.structured.inject_state(graph)

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
            graph.nodes.pop(transient)

        for step in range(
            1,
            episode.decision_step + 1,
        ):
            self.dynamics.step(
                graph,
                step,
            )
            self.memory.maintain(graph)

        schema_name, confidence = self.rules.active(
            episode.task
        )

        rule_bit = (
            1
            if schema_name
            and schema_name.endswith(":invert")
            else 0
        )

        state = self.structured.construct(
            graph,
            episode,
            rule_bit,
            confidence,
        )

        cues = [
            binding.value
            for binding in state.bindings
            if binding.role in (
                "cue1",
                "cue2",
                "cue3",
            )
        ]

        if len(cues) != 3:
            cues = [
                int(x)
                for x in episode.cue_bits
            ]

        memory = int(state.memory_bit)

        # Direct causal-schema prediction.
        decision = self.rules.apply(
            episode.task,
            memory,
            cues,
        )

        correct = (
            decision == episode.answer_bit
        )

        if learn:
            self.structured.feedback(
                decision,
                episode.answer_bit,
                episode,
            )

            self.rules.observe(
                episode.task,
                memory,
                cues,
                episode.answer_bit,
                episode,
            )

        state_now = self.rules._state(
            episode.task
        )

        return {
            "correct": correct,
            "decision": decision,
            "answer": episode.answer_bit,
            "schema": (
                state_now.active.name
                if state_now.active
                else None
            ),
            "confidence": state_now.active_confidence,
            "pending": state_now.pending,
            "revisions": state_now.revisions,
            "phase": state_now.phase,
        }
